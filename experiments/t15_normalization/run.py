"""
T-15: Normalization Layer Analysis & Replacement

Research questions:
1. How do different normalization methods compare in forward/backward latency on B200?
2. Can we swap RMSNorm for LayerNorm/GroupNorm/BatchNorm/Identity in pretrained Qwen3-4B
   without catastrophic quality loss?
3. Is mean centering redundant (RMSNorm ≈ LayerNorm) when residual stream means are near-zero?
4. Does removing QK-norm cause attention logit explosion?
5. What do activation statistics (mean, variance, max, kurtosis) look like across layers?

Usage: poetry run python experiments/t15_normalization/run.py
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── Configuration ───────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda:0"
SEED = 42
CALIBRATION_PATH = Path("data/text_completions/qwen3-4b-instruct-2507/completions.json")
RESULTS_DIR = Path("experiments/t15_normalization/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark settings
BENCH_HIDDEN_SIZES = [2048, 2560, 4096]
BENCH_SEQ_LENGTHS = [512, 2048, 8192]
BENCH_BATCH_SIZE = 4
BENCH_WARMUP = 10
BENCH_ITERS = 50

# ── Utilities ───────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_calibration_data(path, tokenizer, device, max_samples=20):
    """Load pre-generated completions for perplexity evaluation."""
    with open(path, "r") as f:
        data = json.load(f)
    completions = data["completions"][:max_samples]
    prepared = []
    for c in completions:
        tokens = tokenizer(c["full_text"], return_tensors="pt").to(device)
        prepared.append({
            "tokens": tokens,
            "prompt_len": c["prompt_token_count"],
            "full_text": c["full_text"],
        })
    return prepared


def compute_loss(model, calibration_data):
    """Compute mean next-token cross-entropy loss on completion tokens."""
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for sample in calibration_data:
            tokens = sample["tokens"]
            prompt_len = sample["prompt_len"]
            outputs = model(**tokens, use_cache=False)
            logits = outputs.logits
            # Completion tokens only
            shift_logits = logits[0, prompt_len:-1, :]
            shift_labels = tokens["input_ids"][0, prompt_len + 1:]
            if shift_labels.numel() == 0:
                continue
            loss = F.cross_entropy(
                shift_logits, shift_labels, reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def compute_perplexity(loss):
    """Convert nats loss to perplexity."""
    if np.isnan(loss) or np.isinf(loss):
        return float("nan")
    return np.exp(min(loss, 100))  # Cap to avoid overflow


# ── Norm Layer Definitions ──────────────────────────────────────────────────

class RMSNormBench(nn.Module):
    """Standalone RMSNorm for benchmarking (unfused Python implementation)."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RMSNormCompiled(nn.Module):
    """RMSNorm compiled with torch.compile for fair benchmark comparison."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self._compiled_forward = None

    def _raw_forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

    def forward(self, x):
        if self._compiled_forward is None:
            self._compiled_forward = torch.compile(self._raw_forward)
        return self._compiled_forward(x)


class GroupNormWrapper(nn.Module):
    """GroupNorm wrapper that handles (B, seq, hidden) input by permuting to (B, hidden, seq).

    Note: This changes the normalization semantics. Standard GroupNorm on (B, C, L)
    normalizes over (C/G, L) jointly — grouping channels AND normalizing across the
    sequence dimension. This is fundamentally different from LayerNorm/RMSNorm which
    normalize over the feature dimension only, independently per token.
    """
    def __init__(self, num_groups, hidden_size, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, hidden_size, eps=eps)

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(0, 2, 1)
            x = self.gn(x)
            return x.permute(0, 2, 1)
        return self.gn(x)

    @property
    def weight(self):
        return self.gn.weight

    @weight.setter
    def weight(self, value):
        self.gn.weight = value

    @property
    def bias(self):
        return self.gn.bias


class BatchNormWrapper(nn.Module):
    """BatchNorm1d wrapper for (B, seq, hidden) input.

    Permutes to (B*seq, hidden) for BatchNorm1d, since we want to normalize
    across the batch+sequence dimension per feature. In eval mode, uses running
    statistics (set during warmup forward passes in train mode).
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_size, eps=eps)

    def forward(self, x):
        if x.dim() == 3:
            B, S, H = x.shape
            x = x.reshape(B * S, H)
            x = self.bn(x)
            return x.reshape(B, S, H)
        return self.bn(x)

    @property
    def weight(self):
        return self.bn.weight

    @weight.setter
    def weight(self, value):
        self.bn.weight = value

    @property
    def bias(self):
        return self.bn.bias


# ── Section 1: Normalization Benchmarks ─────────────────────────────────────

def benchmark_norms():
    """Benchmark forward and backward latency for each norm type.

    Includes both unfused Python RMSNorm and torch.compile'd RMSNorm to
    fairly compare against PyTorch's fused LayerNorm CUDA kernel.
    """
    print("\n" + "=" * 70)
    print("SECTION 1: Normalization Forward/Backward Benchmarks")
    print("=" * 70)

    norm_types = ["RMSNorm (unfused)", "RMSNorm (compiled)", "LayerNorm",
                  "BatchNorm", "GroupNorm", "InstanceNorm", "Identity"]
    results = {}

    for hidden_size in BENCH_HIDDEN_SIZES:
        for seq_len in BENCH_SEQ_LENGTHS:
            key = f"h{hidden_size}_s{seq_len}"
            results[key] = {}
            print(f"\n  Hidden={hidden_size}, SeqLen={seq_len}")

            for norm_type in norm_types:
                try:
                    if norm_type == "RMSNorm (unfused)":
                        norm = RMSNormBench(hidden_size)
                    elif norm_type == "RMSNorm (compiled)":
                        norm = RMSNormCompiled(hidden_size)
                    elif norm_type == "LayerNorm":
                        norm = nn.LayerNorm(hidden_size)
                    elif norm_type == "BatchNorm":
                        norm = nn.BatchNorm1d(hidden_size)
                    elif norm_type == "GroupNorm":
                        norm = nn.GroupNorm(min(32, hidden_size), hidden_size)
                    elif norm_type == "InstanceNorm":
                        norm = nn.InstanceNorm1d(hidden_size, affine=True)
                    elif norm_type == "Identity":
                        norm = nn.Identity()
                    else:
                        continue

                    norm = norm.to(DEVICE).to(torch.bfloat16)

                    # BatchNorm/GroupNorm/InstanceNorm need (B, C, L) format
                    if norm_type in ("BatchNorm", "InstanceNorm", "GroupNorm"):
                        x = torch.randn(BENCH_BATCH_SIZE, hidden_size, seq_len,
                                        device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
                    else:
                        x = torch.randn(BENCH_BATCH_SIZE, seq_len, hidden_size,
                                        device=DEVICE, dtype=torch.bfloat16, requires_grad=True)

                    # Extended warmup for compiled variant (triggers compilation)
                    warmup_count = BENCH_WARMUP * 3 if "compiled" in norm_type else BENCH_WARMUP
                    for _ in range(warmup_count):
                        out = norm(x)
                        out.sum().backward()
                        x.grad = None

                    # Forward benchmark
                    torch.cuda.synchronize()
                    fwd_times = []
                    for _ in range(BENCH_ITERS):
                        x_new = x.detach().requires_grad_(True)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        out = norm(x_new)
                        torch.cuda.synchronize()
                        fwd_times.append((time.perf_counter() - t0) * 1000)

                    # Backward benchmark
                    bwd_times = []
                    for _ in range(BENCH_ITERS):
                        x_new = x.detach().requires_grad_(True)
                        out = norm(x_new)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        out.sum().backward()
                        torch.cuda.synchronize()
                        bwd_times.append((time.perf_counter() - t0) * 1000)

                    fwd_ms = np.median(fwd_times)
                    bwd_ms = np.median(bwd_times)
                    results[key][norm_type] = {
                        "forward_ms": round(fwd_ms, 4),
                        "backward_ms": round(bwd_ms, 4),
                        "total_ms": round(fwd_ms + bwd_ms, 4),
                        "param_count": sum(p.numel() for p in norm.parameters()),
                    }
                    print(f"    {norm_type:22s}  fwd={fwd_ms:.4f}ms  bwd={bwd_ms:.4f}ms  total={fwd_ms+bwd_ms:.4f}ms")

                except Exception as e:
                    results[key][norm_type] = {"error": str(e)}
                    print(f"    {norm_type:22s}  ERROR: {e}")

                # Free memory
                if 'norm' in dir():
                    del norm
                torch.cuda.empty_cache()

    return results


# ── Section 2: Activation Statistics Analysis ───────────────────────────────

def analyze_activation_stats(model, tokenizer, calibration_data):
    """Measure activation statistics at norm inputs and outputs across all layers.

    Collects stats on both the *input* to each norm (= residual stream) and the
    *output* of each norm (= what attention/MLP actually receives).
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Activation Statistics")
    print("=" * 70)

    num_layers = len(model.model.layers)
    # Input stats (what enters the norm = residual stream)
    pre_attn_norm_input = {i: [] for i in range(num_layers)}
    pre_mlp_norm_input = {i: [] for i in range(num_layers)}
    # Output stats (what exits the norm = what attention/MLP sees)
    pre_attn_norm_output = {i: [] for i in range(num_layers)}
    pre_mlp_norm_output = {i: [] for i in range(num_layers)}
    # Residual stream after full layer
    residual_stream_stats = {i: [] for i in range(num_layers)}

    hooks = []

    def compute_stats(x):
        """Compute comprehensive stats on a tensor."""
        x_flat = x.cpu().numpy().flatten()
        return {
            "mean": x.mean().item(),
            "std": x.std().item(),
            "abs_max": x.abs().max().item(),
            "kurtosis": float(scipy_stats.kurtosis(x_flat)),
            "skewness": float(scipy_stats.skew(x_flat)),
            "near_zero_frac": (x.abs() < 1e-3).float().mean().item(),
        }

    def make_norm_hook(input_storage, output_storage, layer_idx):
        def hook_fn(module, input, output):
            x_in = input[0].detach().float()
            x_out = output.detach().float()
            input_storage[layer_idx].append(compute_stats(x_in))
            output_storage[layer_idx].append(compute_stats(x_out))
        return hook_fn

    def make_residual_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                x = output[0].detach().float()
            else:
                x = output.detach().float()
            storage[layer_idx].append({
                "mean": x.mean().item(),
                "std": x.std().item(),
                "abs_max": x.abs().max().item(),
            })
        return hook_fn

    for i in range(num_layers):
        layer = model.model.layers[i]
        hooks.append(layer.input_layernorm.register_forward_hook(
            make_norm_hook(pre_attn_norm_input, pre_attn_norm_output, i)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            make_norm_hook(pre_mlp_norm_input, pre_mlp_norm_output, i)))
        hooks.append(layer.register_forward_hook(
            make_residual_hook(residual_stream_stats, i)))

    with torch.no_grad():
        for sample in tqdm(calibration_data[:10], desc="Collecting activation stats"):
            model(**sample["tokens"], use_cache=False)

    for h in hooks:
        h.remove()

    # Aggregate per layer
    def aggregate(storage, has_kurtosis=True):
        result = {}
        for i in range(num_layers):
            if not storage[i]:
                continue
            agg = {
                "mean": np.mean([s["mean"] for s in storage[i]]),
                "std": np.mean([s["std"] for s in storage[i]]),
                "abs_max": np.max([s["abs_max"] for s in storage[i]]),
            }
            if has_kurtosis and "kurtosis" in storage[i][0]:
                agg["kurtosis"] = np.mean([s["kurtosis"] for s in storage[i]])
                agg["skewness"] = np.mean([s["skewness"] for s in storage[i]])
                agg["near_zero_frac"] = np.mean([s["near_zero_frac"] for s in storage[i]])
            result[i] = agg
        return result

    aggregated = {
        "pre_attn_norm_input": aggregate(pre_attn_norm_input),
        "pre_attn_norm_output": aggregate(pre_attn_norm_output),
        "pre_mlp_norm_input": aggregate(pre_mlp_norm_input),
        "pre_mlp_norm_output": aggregate(pre_mlp_norm_output),
        "residual_stream": aggregate(residual_stream_stats, has_kurtosis=False),
    }

    # Print summary
    print(f"\n  {'Layer':>5} | {'NormIn Std':>10} | {'NormOut Std':>11} | {'Kurtosis':>10} | {'Residual Std':>13} | {'Mean/Std':>9}")
    print("  " + "-" * 75)
    for i in range(num_layers):
        pa_in = aggregated["pre_attn_norm_input"].get(i, {})
        pa_out = aggregated["pre_attn_norm_output"].get(i, {})
        rs = aggregated["residual_stream"].get(i, {})
        in_std = pa_in.get("std", 0)
        in_mean = pa_in.get("mean", 0)
        ratio = abs(in_mean) / in_std if in_std > 0 else 0
        print(f"  {i:5d} | {in_std:10.4f} | {pa_out.get('std', 0):11.4f} | "
              f"{pa_in.get('kurtosis', 0):10.1f} | {rs.get('std', 0):13.4f} | {ratio:9.4f}")

    return aggregated


# ── Section 3: QK-Norm Ablation ────────────────────────────────────────────

def ablate_qk_norm(model, tokenizer, calibration_data):
    """Remove QK-norm from attention layers and measure impact on attention logits."""
    print("\n" + "=" * 70)
    print("SECTION 3: QK-Norm Ablation")
    print("=" * 70)

    layer0_attn = model.model.layers[0].self_attn
    has_q_norm = hasattr(layer0_attn, "q_norm")
    has_k_norm = hasattr(layer0_attn, "k_norm")
    print(f"  Model has q_norm: {has_q_norm}, k_norm: {has_k_norm}")

    if not (has_q_norm and has_k_norm):
        print("  No QK-norm found, skipping ablation.")
        return {"has_qk_norm": False}

    baseline_loss = compute_loss(model, calibration_data)
    print(f"  Baseline loss: {baseline_loss:.6f} (PPL={compute_perplexity(baseline_loss):.2f})")

    # Collect post-norm Q/K magnitudes (baseline)
    qk_stats_baseline = collect_qk_stats(model, calibration_data, label="with QK-norm")

    # Save originals and replace with Identity
    num_layers = len(model.model.layers)
    original_q_norms = {}
    original_k_norms = {}
    for i in range(num_layers):
        attn = model.model.layers[i].self_attn
        original_q_norms[i] = attn.q_norm
        original_k_norms[i] = attn.k_norm
        attn.q_norm = nn.Identity()
        attn.k_norm = nn.Identity()

    no_qknorm_loss = compute_loss(model, calibration_data)
    print(f"  Without QK-norm loss: {no_qknorm_loss:.6f} (PPL={compute_perplexity(no_qknorm_loss):.2f})")
    print(f"  Loss ratio: {no_qknorm_loss / baseline_loss:.4f}")

    # Collect post-norm Q/K magnitudes (without QK-norm = raw Q/K proj outputs)
    qk_stats_no_norm = collect_qk_stats(model, calibration_data, label="without QK-norm")

    # Restore
    for i in range(num_layers):
        attn = model.model.layers[i].self_attn
        attn.q_norm = original_q_norms[i]
        attn.k_norm = original_k_norms[i]

    results = {
        "has_qk_norm": True,
        "baseline_loss": baseline_loss,
        "baseline_ppl": compute_perplexity(baseline_loss),
        "no_qknorm_loss": no_qknorm_loss,
        "no_qknorm_ppl": compute_perplexity(no_qknorm_loss),
        "loss_ratio": no_qknorm_loss / baseline_loss,
        "qk_stats_baseline": qk_stats_baseline,
        "qk_stats_no_qknorm": qk_stats_no_norm,
    }

    # Print per-layer comparison
    print(f"\n  {'Layer':>5} | {'Q max (norm)':>12} | {'Q max (raw)':>12} | {'K max (norm)':>12} | {'K max (raw)':>12}")
    print("  " + "-" * 60)
    for i in range(0, num_layers, 4):  # Print every 4th layer
        b = qk_stats_baseline["per_layer"].get(i, {})
        n = qk_stats_no_norm["per_layer"].get(i, {})
        print(f"  {i:5d} | {b.get('q_max', 0):12.2f} | {n.get('q_max', 0):12.2f} | "
              f"{b.get('k_max', 0):12.2f} | {n.get('k_max', 0):12.2f}")

    return results


def collect_qk_stats(model, calibration_data, max_samples=5, label=""):
    """Collect Q/K magnitudes after norm is applied (hooks on q_norm/k_norm output).

    In Qwen3, the forward path is:
      q = q_norm(q_proj(x).view(B, S, H, D))  # then transpose
    So hooking q_norm gives us the post-norm Q vectors that actually enter attention.
    When QK-norm is replaced with Identity, this gives us the raw projection outputs.
    """
    num_layers = len(model.model.layers)
    q_mags = {i: [] for i in range(num_layers)}
    k_mags = {i: [] for i in range(num_layers)}
    hooks = []

    def make_hook(storage, layer_idx):
        def hook_fn(module, input, output):
            # output = normalized Q or K, shape varies
            storage[layer_idx].append(output.detach().float().abs().max().item())
        return hook_fn

    for i in range(num_layers):
        attn = model.model.layers[i].self_attn
        hooks.append(attn.q_norm.register_forward_hook(make_hook(q_mags, i)))
        hooks.append(attn.k_norm.register_forward_hook(make_hook(k_mags, i)))

    with torch.no_grad():
        for sample in calibration_data[:max_samples]:
            model(**sample["tokens"], use_cache=False)

    for h in hooks:
        h.remove()

    per_layer = {}
    all_q_maxes = []
    all_k_maxes = []
    for i in range(num_layers):
        if q_mags[i]:
            q_max = float(np.max(q_mags[i]))
            k_max = float(np.max(k_mags[i]))
            all_q_maxes.append(q_max)
            all_k_maxes.append(k_max)
            per_layer[i] = {
                "q_max": q_max,
                "k_max": k_max,
                "q_mean": float(np.mean(q_mags[i])),
                "k_mean": float(np.mean(k_mags[i])),
            }

    return {
        "max_q": float(max(all_q_maxes)) if all_q_maxes else 0,
        "max_k": float(max(all_k_maxes)) if all_k_maxes else 0,
        "mean_q": float(np.mean(all_q_maxes)) if all_q_maxes else 0,
        "mean_k": float(np.mean(all_k_maxes)) if all_k_maxes else 0,
        "per_layer": per_layer,
    }


# ── Section 4: Norm Replacement Experiments ─────────────────────────────────

def replace_norms_experiment(model, tokenizer, calibration_data):
    """Replace RMSNorm with various norm types and measure quality impact."""
    print("\n" + "=" * 70)
    print("SECTION 4: Norm Replacement in Pretrained Qwen3-4B")
    print("=" * 70)

    hidden_size = model.config.hidden_size

    # Baseline
    baseline_loss = compute_loss(model, calibration_data)
    baseline_ppl = compute_perplexity(baseline_loss)
    print(f"  Baseline: loss={baseline_loss:.6f}, PPL={baseline_ppl:.2f}")

    # Analyze original norm weights
    norm_weight_stats = analyze_norm_weights(model)

    results = {
        "baseline_loss": baseline_loss,
        "baseline_ppl": baseline_ppl,
        "norm_weight_stats": norm_weight_stats,
        "replacements": {},
    }

    # Define replacement experiments
    replacements = [
        ("LayerNorm", lambda hs: nn.LayerNorm(hs)),
        ("BatchNorm", lambda hs: BatchNormWrapper(hs)),
        ("GroupNorm_1", lambda hs: GroupNormWrapper(1, hs)),
        ("GroupNorm_8", lambda hs: GroupNormWrapper(8, hs)),
        ("GroupNorm_32", lambda hs: GroupNormWrapper(32, hs)),
        ("GroupNorm_64", lambda hs: GroupNormWrapper(64, hs)),
        ("Identity", lambda hs: nn.Identity()),
    ]

    for repl_name, make_norm in replacements:
        print(f"\n  Replacing all norms with {repl_name}...")
        t0 = time.perf_counter()

        # Save originals
        originals = save_original_norms(model)

        # Replace all norm layers
        replace_all_norms(model, make_norm, hidden_size, repl_name)

        # Warmup BatchNorm running stats if needed
        if repl_name == "BatchNorm":
            _warmup_batchnorm(model, calibration_data)

        # Measure loss
        repl_loss = compute_loss(model, calibration_data)
        repl_ppl = compute_perplexity(repl_loss)
        elapsed = time.perf_counter() - t0

        # Collect activation stats (skip for Identity which produces NaN)
        act_stats = {}
        if repl_name != "Identity":
            act_stats = quick_activation_stats(model, calibration_data)

        loss_ratio = repl_loss / baseline_loss if not np.isnan(repl_loss) else float("nan")
        results["replacements"][repl_name] = {
            "loss": repl_loss,
            "ppl": repl_ppl,
            "loss_ratio": loss_ratio,
            "time_s": round(elapsed, 2),
            "activation_stats": act_stats,
        }
        print(f"    loss={repl_loss:.6f}, PPL={repl_ppl:.2f}, ratio={loss_ratio:.4f}, time={elapsed:.1f}s")

        # Restore originals
        restore_original_norms(model, originals)

    # Verify restoration
    verify_loss = compute_loss(model, calibration_data)
    print(f"\n  Verification after restore: loss={verify_loss:.6f} (should match baseline {baseline_loss:.6f})")
    results["verify_loss_after_restore"] = verify_loss

    return results


def _warmup_batchnorm(model, calibration_data, steps=5):
    """Run a few forward passes in train mode to accumulate BatchNorm running stats."""
    model.train()
    with torch.no_grad():
        for sample in calibration_data[:steps]:
            model(**sample["tokens"], use_cache=False)
    model.eval()


def analyze_norm_weights(model):
    """Analyze the learned scale parameters in each RMSNorm layer."""
    num_layers = len(model.model.layers)
    stats = {"input_layernorm": [], "post_attention_layernorm": []}

    for i in range(num_layers):
        layer = model.model.layers[i]
        for name in ["input_layernorm", "post_attention_layernorm"]:
            norm = getattr(layer, name)
            w = norm.weight.detach().float()
            stats[name].append({
                "layer": i,
                "mean": w.mean().item(),
                "std": w.std().item(),
                "min": w.min().item(),
                "max": w.max().item(),
                "near_one_frac": ((w - 1.0).abs() < 0.1).float().mean().item(),
            })

    final_w = model.model.norm.weight.detach().float()
    stats["final_norm"] = {
        "mean": final_w.mean().item(),
        "std": final_w.std().item(),
        "min": final_w.min().item(),
        "max": final_w.max().item(),
    }

    return stats


def save_original_norms(model):
    """Save references to all original norm layers."""
    originals = {"layers": [], "final_norm": model.model.norm}
    for layer in model.model.layers:
        originals["layers"].append({
            "input_layernorm": layer.input_layernorm,
            "post_attention_layernorm": layer.post_attention_layernorm,
        })
    if hasattr(model.model.layers[0].self_attn, "q_norm"):
        originals["q_norms"] = [l.self_attn.q_norm for l in model.model.layers]
        originals["k_norms"] = [l.self_attn.k_norm for l in model.model.layers]
    return originals


def restore_original_norms(model, originals):
    """Restore original norm layers."""
    model.model.norm = originals["final_norm"]
    for i, layer in enumerate(model.model.layers):
        layer.input_layernorm = originals["layers"][i]["input_layernorm"]
        layer.post_attention_layernorm = originals["layers"][i]["post_attention_layernorm"]
    if "q_norms" in originals:
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.q_norm = originals["q_norms"][i]
            layer.self_attn.k_norm = originals["k_norms"][i]


def replace_all_norms(model, make_norm_fn, hidden_size, repl_name):
    """Replace all normalization layers in the model."""
    num_layers = len(model.model.layers)
    head_dim = model.model.layers[0].self_attn.head_dim

    for i in range(num_layers):
        layer = model.model.layers[i]
        for norm_name in ["input_layernorm", "post_attention_layernorm"]:
            old_norm = getattr(layer, norm_name)
            new_norm = make_norm_fn(hidden_size).to(DEVICE).to(torch.bfloat16)
            _transfer_norm_weights(old_norm, new_norm, repl_name)
            setattr(layer, norm_name, new_norm)

        # QK-norm replacement: only LayerNorm and Identity work on the 4D shape
        if hasattr(layer.self_attn, "q_norm"):
            if repl_name == "Identity":
                layer.self_attn.q_norm = nn.Identity()
                layer.self_attn.k_norm = nn.Identity()
            elif repl_name == "LayerNorm":
                for qk_name in ["q_norm", "k_norm"]:
                    old_qk = getattr(layer.self_attn, qk_name)
                    new_qk = nn.LayerNorm(head_dim).to(DEVICE).to(torch.bfloat16)
                    _transfer_norm_weights(old_qk, new_qk, repl_name)
                    setattr(layer.self_attn, qk_name, new_qk)
            # For GroupNorm/BatchNorm: keep original QK-norm (can't operate on 4D shape)

    # Replace final norm
    old_final = model.model.norm
    new_final = make_norm_fn(hidden_size).to(DEVICE).to(torch.bfloat16)
    _transfer_norm_weights(old_final, new_final, repl_name)
    model.model.norm = new_final


def _transfer_norm_weights(old_norm, new_norm, repl_name):
    """Transfer learned weights from old norm to new norm where possible."""
    if repl_name == "Identity":
        return
    if not hasattr(old_norm, "weight"):
        return

    old_weight = old_norm.weight.data

    # For wrappers, access the inner module
    target = new_norm
    if isinstance(new_norm, GroupNormWrapper):
        target = new_norm.gn
    elif isinstance(new_norm, BatchNormWrapper):
        target = new_norm.bn

    if hasattr(target, "weight") and target.weight.shape == old_weight.shape:
        target.weight.data.copy_(old_weight)
    if hasattr(target, "bias") and target.bias is not None:
        target.bias.data.zero_()


def quick_activation_stats(model, calibration_data, max_samples=5):
    """Collect residual stream std per layer."""
    num_layers = len(model.model.layers)
    layer_stds = {i: [] for i in range(num_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            val = x.detach().float().std().item()
            if not np.isnan(val):
                layer_stds[layer_idx].append(val)
        return hook_fn

    for i in range(num_layers):
        hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for sample in calibration_data[:max_samples]:
            try:
                model(**sample["tokens"], use_cache=False)
            except Exception:
                break  # Identity replacement may produce NaN/Inf

    for h in hooks:
        h.remove()

    return {i: float(np.mean(layer_stds[i])) for i in range(num_layers) if layer_stds[i]}


# ── Section 5: Per-Layer Norm Replacement ───────────────────────────────────

def per_layer_norm_sensitivity(model, tokenizer, calibration_data):
    """Replace norm in one layer at a time to find which layers are most norm-sensitive."""
    print("\n" + "=" * 70)
    print("SECTION 5: Per-Layer Norm Sensitivity (Identity Replacement)")
    print("=" * 70)

    num_layers = len(model.model.layers)
    baseline_loss = compute_loss(model, calibration_data)
    print(f"  Baseline loss: {baseline_loss:.6f}")

    results = {"baseline_loss": baseline_loss, "per_layer": {}}

    for i in tqdm(range(num_layers), desc="Per-layer knockout"):
        layer = model.model.layers[i]

        # Save originals for this layer
        orig_input_ln = layer.input_layernorm
        orig_post_ln = layer.post_attention_layernorm

        # Replace both norms in this layer with Identity
        layer.input_layernorm = nn.Identity()
        layer.post_attention_layernorm = nn.Identity()

        loss = compute_loss(model, calibration_data)
        ratio = loss / baseline_loss if not np.isnan(loss) else float("nan")

        results["per_layer"][i] = {
            "loss": loss,
            "loss_ratio": ratio,
        }

        # Restore
        layer.input_layernorm = orig_input_ln
        layer.post_attention_layernorm = orig_post_ln

    # Print top-5 most sensitive layers
    sorted_layers = sorted(results["per_layer"].items(),
                           key=lambda x: x[1]["loss_ratio"] if not np.isnan(x[1]["loss_ratio"]) else float("inf"),
                           reverse=True)
    print(f"\n  Top-5 most norm-sensitive layers:")
    for layer_idx, data in sorted_layers[:5]:
        print(f"    Layer {layer_idx}: loss_ratio={data['loss_ratio']:.2f}")

    print(f"\n  Top-5 least norm-sensitive layers:")
    sorted_asc = sorted(results["per_layer"].items(),
                        key=lambda x: x[1]["loss_ratio"] if not np.isnan(x[1]["loss_ratio"]) else float("inf"))
    for layer_idx, data in sorted_asc[:5]:
        print(f"    Layer {layer_idx}: loss_ratio={data['loss_ratio']:.2f}")

    return results


# ── Section 6: Norm Weight Analysis ────────────────────────────────────────

def analyze_norm_scale_distribution(model):
    """Deep analysis of learned RMSNorm scale parameters."""
    print("\n" + "=" * 70)
    print("SECTION 6: Norm Weight Distribution Analysis")
    print("=" * 70)

    num_layers = len(model.model.layers)
    results = {"layers": {}, "summary": {}}

    all_weights_input = []
    all_weights_post = []

    for i in range(num_layers):
        layer = model.model.layers[i]
        w_in = layer.input_layernorm.weight.detach().float().cpu().numpy()
        w_post = layer.post_attention_layernorm.weight.detach().float().cpu().numpy()

        all_weights_input.append(w_in)
        all_weights_post.append(w_post)

        results["layers"][i] = {
            "input_ln": {
                "mean": float(w_in.mean()),
                "std": float(w_in.std()),
                "min": float(w_in.min()),
                "max": float(w_in.max()),
                "deviation_from_one": float(np.abs(w_in - 1.0).mean()),
            },
            "post_attn_ln": {
                "mean": float(w_post.mean()),
                "std": float(w_post.std()),
                "min": float(w_post.min()),
                "max": float(w_post.max()),
                "deviation_from_one": float(np.abs(w_post - 1.0).mean()),
            },
        }

    all_in = np.concatenate(all_weights_input)
    all_post = np.concatenate(all_weights_post)

    results["summary"] = {
        "input_layernorm": {
            "global_mean": float(all_in.mean()),
            "global_std": float(all_in.std()),
            "frac_within_10pct_of_1": float((np.abs(all_in - 1.0) < 0.1).mean()),
            "frac_within_50pct_of_1": float((np.abs(all_in - 1.0) < 0.5).mean()),
        },
        "post_attention_layernorm": {
            "global_mean": float(all_post.mean()),
            "global_std": float(all_post.std()),
            "frac_within_10pct_of_1": float((np.abs(all_post - 1.0) < 0.1).mean()),
            "frac_within_50pct_of_1": float((np.abs(all_post - 1.0) < 0.5).mean()),
        },
    }

    print(f"  Input LayerNorm weights: mean={all_in.mean():.4f}, std={all_in.std():.4f}")
    print(f"    Within 10% of 1.0: {(np.abs(all_in - 1.0) < 0.1).mean()*100:.1f}%")
    print(f"    Within 50% of 1.0: {(np.abs(all_in - 1.0) < 0.5).mean()*100:.1f}%")
    print(f"  Post-Attn LayerNorm weights: mean={all_post.mean():.4f}, std={all_post.std():.4f}")
    print(f"    Within 10% of 1.0: {(np.abs(all_post - 1.0) < 0.1).mean()*100:.1f}%")
    print(f"    Within 50% of 1.0: {(np.abs(all_post - 1.0) < 0.5).mean()*100:.1f}%")

    return results


# ── Section 7: Residual Stream Mean Analysis ────────────────────────────────

def analyze_residual_means(model, tokenizer, calibration_data):
    """Measure residual stream means and relate to std (mean/std ratio)."""
    print("\n" + "=" * 70)
    print("SECTION 7: Residual Stream Mean Analysis (RMSNorm vs LayerNorm)")
    print("=" * 70)

    num_layers = len(model.model.layers)
    layer_means = {i: [] for i in range(num_layers)}
    layer_mean_magnitudes = {i: [] for i in range(num_layers)}
    layer_stds = {i: [] for i in range(num_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            token_means = x.mean(dim=-1)  # (batch, seq_len)
            token_stds = x.std(dim=-1)
            layer_means[layer_idx].append(token_means.mean().item())
            layer_mean_magnitudes[layer_idx].append(token_means.abs().mean().item())
            layer_stds[layer_idx].append(token_stds.mean().item())
        return hook_fn

    for i in range(num_layers):
        hooks.append(model.model.layers[i].input_layernorm.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        for sample in calibration_data[:10]:
            model(**sample["tokens"], use_cache=False)

    for h in hooks:
        h.remove()

    results = {}
    print(f"\n  {'Layer':>5} | {'|Mean|':>8} | {'Std':>10} | {'|Mean|/Std':>10} | {'Centering Needed?':>18}")
    print("  " + "-" * 65)
    for i in range(num_layers):
        avg_mean = np.mean(layer_means[i])
        avg_abs_mean = np.mean(layer_mean_magnitudes[i])
        avg_std = np.mean(layer_stds[i])
        ratio = avg_abs_mean / avg_std if avg_std > 0 else 0
        # The relevant question: is |mean|/std large enough to matter?
        # If ratio < 0.001, centering has negligible effect
        # If ratio > 0.01, centering corrects a non-trivial bias
        redundant = "NO" if ratio < 0.001 else ("MOSTLY" if ratio < 0.01 else "SLIGHTLY")
        results[i] = {
            "mean_of_means": avg_mean,
            "abs_mean_magnitude": avg_abs_mean,
            "std": avg_std,
            "mean_std_ratio": ratio,
            "centering_redundant": redundant,
        }
        print(f"  {i:5d} | {avg_abs_mean:8.4f} | {avg_std:10.4f} | {ratio:10.6f} | {redundant:>18}")

    return results


# ── Section 8: Visualization ───────────────────────────────────────────────

def create_plots(bench_results, replacement_results, activation_stats,
                 qk_results, norm_weights, residual_means, per_layer_sensitivity):
    """Generate all visualization plots."""
    print("\n" + "=" * 70)
    print("SECTION 8: Generating Plots")
    print("=" * 70)

    plot_benchmark_latency(bench_results)
    plot_replacement_losses(replacement_results)
    plot_activation_stats(activation_stats)
    if qk_results.get("has_qk_norm"):
        plot_qk_norm_ablation(qk_results)
    plot_norm_weight_distribution(norm_weights)
    plot_residual_means(residual_means)
    plot_replacement_activation_profiles(replacement_results)
    plot_per_layer_sensitivity(per_layer_sensitivity)
    plot_norm_io_comparison(activation_stats)

    print("  All plots saved.")


def plot_benchmark_latency(bench_results):
    """Plot forward/backward latency comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Normalization Forward+Backward Latency (B200, bf16)", fontsize=14)

    norm_types = list(next(iter(bench_results.values())).keys())
    colors = ["#2196F3", "#1565C0", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#607D8B"]
    colors = colors[:len(norm_types)]

    for ax_idx, hidden_size in enumerate(BENCH_HIDDEN_SIZES):
        ax = axes[ax_idx]
        seq_lens = BENCH_SEQ_LENGTHS
        x = np.arange(len(seq_lens))
        width = 0.12

        for i, (nt, color) in enumerate(zip(norm_types, colors)):
            totals = []
            for sl in seq_lens:
                key = f"h{hidden_size}_s{sl}"
                if key in bench_results and nt in bench_results[key]:
                    entry = bench_results[key][nt]
                    totals.append(entry.get("total_ms", 0))
                else:
                    totals.append(0)
            label = nt.replace(" (unfused)", "\n(unfused)").replace(" (compiled)", "\n(compiled)")
            ax.bar(x + i * width, totals, width, label=label, color=color, alpha=0.85)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Total Latency (ms)")
        ax.set_title(f"Hidden Size = {hidden_size}")
        ax.set_xticks(x + width * (len(norm_types) - 1) / 2)
        ax.set_xticklabels([str(s) for s in seq_lens])
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "benchmark_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved benchmark_latency.png")


def plot_replacement_losses(replacement_results):
    """Plot loss for each norm replacement with log scale for readability."""
    repl = replacement_results["replacements"]
    names = list(repl.keys())
    losses = [repl[n]["loss"] for n in names]
    ratios = [repl[n]["loss_ratio"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Norm Replacement Impact on Qwen3-4B-Instruct", fontsize=14)

    # Color by severity
    def ratio_color(r):
        if np.isnan(r):
            return "#333333"
        return "#4CAF50" if r < 2 else "#FF9800" if r < 10 else "#F44336"

    colors = [ratio_color(r) for r in ratios]

    # Loss comparison (log scale for readability)
    valid_losses = [l for l in losses if not np.isnan(l)]
    bars = ax1.bar(names, [l if not np.isnan(l) else 0 for l in losses], color=colors, alpha=0.85)
    ax1.axhline(y=replacement_results["baseline_loss"], color="blue",
                linestyle="--", linewidth=2, label=f"Baseline ({replacement_results['baseline_loss']:.4f})")
    ax1.set_ylabel("Cross-Entropy Loss (nats)")
    ax1.set_title("Loss After Norm Replacement")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=30)

    # Loss ratio (log scale)
    valid_ratios = [r if not np.isnan(r) else 0 for r in ratios]
    bars2 = ax2.bar(names, valid_ratios, color=colors, alpha=0.85)
    ax2.axhline(y=1.0, color="blue", linestyle="--", linewidth=2, label="Baseline (1.0)")
    ax2.set_ylabel("Loss Ratio (replacement / baseline)")
    ax2.set_title("Loss Ratio (log scale)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=30)

    # Value labels
    for bar, val in zip(bars2, ratios):
        if not np.isnan(val) and val > 0:
            label = f"{val:.1f}x" if val >= 10 else f"{val:.2f}x"
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                     label, ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "replacement_losses.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved replacement_losses.png")


def plot_activation_stats(activation_stats):
    """Plot activation statistics across layers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Activation Statistics Across Layers (Qwen3-4B-Instruct)", fontsize=14)

    # Pre-norm input std
    for stat_name, title, ax in [
        ("pre_attn_norm_input", "Pre-Attention Norm Input", axes[0, 0]),
        ("pre_mlp_norm_input", "Pre-MLP Norm Input", axes[0, 1]),
    ]:
        data = activation_stats[stat_name]
        data_keys = sorted(data.keys(), key=lambda k: int(k))
        layers = [int(k) for k in data_keys]
        means = [data[k]["mean"] for k in data_keys]
        stds = [data[k]["std"] for k in data_keys]

        ax.plot(layers, means, "b-o", markersize=3, label="Mean", alpha=0.8)
        ax.plot(layers, stds, "r-s", markersize=3, label="Std", alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Kurtosis
    ax = axes[1, 0]
    for stat_name, label, color in [
        ("pre_attn_norm_input", "Pre-Attn", "#2196F3"),
        ("pre_mlp_norm_input", "Pre-MLP", "#4CAF50"),
    ]:
        data = activation_stats[stat_name]
        data_keys = sorted(data.keys(), key=lambda k: int(k))
        layers = [int(k) for k in data_keys]
        kurtosis = [data[k].get("kurtosis", 0) for k in data_keys]
        ax.plot(layers, kurtosis, "-o", markersize=3, label=label, color=color, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Kurtosis")
    ax.set_title("Activation Kurtosis (excess)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Residual stream std
    ax = axes[1, 1]
    rs_data = activation_stats["residual_stream"]
    rs_keys = sorted(rs_data.keys(), key=lambda k: int(k))
    layers = [int(k) for k in rs_keys]
    stds = [rs_data[k]["std"] for k in rs_keys]
    ax.plot(layers, stds, "g-o", markersize=3, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Std")
    ax.set_title("Residual Stream Std")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "activation_stats.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved activation_stats.png")


def plot_qk_norm_ablation(qk_results):
    """Plot QK-norm ablation results (post-norm Q/K magnitudes)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("QK-Norm Ablation: Post-Norm Q/K Magnitudes", fontsize=14)

    per_layer_base = qk_results["qk_stats_baseline"]["per_layer"]
    per_layer_no = qk_results["qk_stats_no_qknorm"]["per_layer"]
    layer_keys = sorted(per_layer_base.keys(), key=lambda k: int(k))
    layers_int = [int(k) for k in layer_keys]

    q_base = [per_layer_base[k]["q_max"] for k in layer_keys]
    q_no = [per_layer_no[k]["q_max"] for k in layer_keys]
    ax1.plot(layers_int, q_base, "b-o", markersize=3, label="With QK-norm", alpha=0.8)
    ax1.plot(layers_int, q_no, "r-s", markersize=3, label="Without QK-norm", alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Max |Q| (post-norm)")
    ax1.set_title("Q Vector Magnitudes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    k_base = [per_layer_base[k]["k_max"] for k in layer_keys]
    k_no = [per_layer_no[k]["k_max"] for k in layer_keys]
    ax2.plot(layers_int, k_base, "b-o", markersize=3, label="With QK-norm", alpha=0.8)
    ax2.plot(layers_int, k_no, "r-s", markersize=3, label="Without QK-norm", alpha=0.8)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Max |K| (post-norm)")
    ax2.set_title("K Vector Magnitudes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "qk_norm_ablation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved qk_norm_ablation.png")


def plot_norm_weight_distribution(norm_weights):
    """Plot distribution of learned norm scale parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learned RMSNorm Scale Parameter Distribution", fontsize=14)

    for ax, norm_name, title in [
        (axes[0], "input_layernorm", "Input LayerNorm (Pre-Attention)"),
        (axes[1], "post_attention_layernorm", "Post-Attention LayerNorm (Pre-MLP)"),
    ]:
        layers_data = norm_weights["layers"]
        layer_keys = sorted(layers_data.keys(), key=lambda k: int(k))
        layers = [int(k) for k in layer_keys]
        short = norm_name.replace("input_layernorm", "input_ln").replace("post_attention_layernorm", "post_attn_ln")
        means = [layers_data[k][short]["mean"] for k in layer_keys]
        stds = [layers_data[k][short]["std"] for k in layer_keys]
        devs = [layers_data[k][short]["deviation_from_one"] for k in layer_keys]

        ax.plot(layers, means, "b-o", markersize=3, label="Mean(weight)", alpha=0.8)
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color="blue")
        ax.plot(layers, devs, "r--s", markersize=3, label="|weight - 1| avg", alpha=0.8)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Scale Parameter Value")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "norm_weight_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved norm_weight_distribution.png")


def plot_residual_means(residual_means):
    """Plot residual stream means with mean/std ratio."""
    rm_keys = sorted(residual_means.keys(), key=lambda k: int(k))
    layers = [int(k) for k in rm_keys]
    abs_means = [residual_means[k]["abs_mean_magnitude"] for k in rm_keys]
    ratios = [residual_means[k]["mean_std_ratio"] for k in rm_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Stream Per-Token Means (Is Mean-Centering Redundant?)", fontsize=14)

    ax1.plot(layers, abs_means, "r-o", markersize=3, alpha=0.8)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("|Mean| Magnitude")
    ax1.set_title("Absolute Mean Magnitude")
    ax1.grid(True, alpha=0.3)

    ax2.plot(layers, ratios, "b-o", markersize=3, alpha=0.8)
    ax2.axhline(y=0.001, color="green", linestyle="--", alpha=0.5, label="Negligible (0.001)")
    ax2.axhline(y=0.01, color="orange", linestyle="--", alpha=0.5, label="Small (0.01)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("|Mean| / Std")
    ax2.set_title("|Mean|/Std Ratio (centering importance)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "residual_means.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved residual_means.png")


def plot_replacement_activation_profiles(replacement_results):
    """Plot residual stream std profiles for each replacement."""
    repl = replacement_results["replacements"]
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Residual Stream Std After Norm Replacement", fontsize=14)

    colors = ["#4CAF50", "#E91E63", "#2196F3", "#9C27B0", "#FF9800", "#F44336", "#607D8B"]
    for (name, data), color in zip(repl.items(), colors):
        act = data.get("activation_stats", {})
        if act:
            act_keys = sorted(act.keys(), key=lambda k: int(k))
            layers = [int(k) for k in act_keys]
            stds = [act[k] for k in act_keys]
            ax.plot(layers, stds, "-o", markersize=3, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Residual Stream Std")
    ax.set_title("Activation Magnitude Profile by Norm Type")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "replacement_activation_profiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved replacement_activation_profiles.png")


def plot_per_layer_sensitivity(per_layer_results):
    """Plot per-layer norm sensitivity (which layers need norm the most)."""
    per_layer = per_layer_results["per_layer"]
    layer_keys = sorted(per_layer.keys(), key=lambda k: int(k))
    layers = [int(k) for k in layer_keys]
    ratios = [per_layer[k]["loss_ratio"] for k in layer_keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if r < 2 else "#FF9800" if r < 5 else "#F44336"
              for r in ratios]
    ax.bar(layers, ratios, color=colors, alpha=0.85)
    ax.axhline(y=1.0, color="blue", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss Ratio (norm removed / baseline)")
    ax.set_title("Per-Layer Norm Sensitivity (Identity Replacement)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_layer_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved per_layer_sensitivity.png")


def plot_norm_io_comparison(activation_stats):
    """Plot norm input vs output statistics to show what normalization actually does."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Normalization Effect: Input vs Output Std", fontsize=14)

    for ax, in_key, out_key, title in [
        (axes[0], "pre_attn_norm_input", "pre_attn_norm_output", "Pre-Attention RMSNorm"),
        (axes[1], "pre_mlp_norm_input", "pre_mlp_norm_output", "Pre-MLP RMSNorm"),
    ]:
        in_data = activation_stats[in_key]
        out_data = activation_stats[out_key]
        in_keys = sorted(in_data.keys(), key=lambda k: int(k))
        layers = [int(k) for k in in_keys]

        in_stds = [in_data[k]["std"] for k in in_keys]
        out_stds = [out_data[k]["std"] for k in in_keys]

        ax.plot(layers, in_stds, "r-o", markersize=3, label="Input Std", alpha=0.8)
        ax.plot(layers, out_stds, "b-s", markersize=3, label="Output Std", alpha=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Std")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "norm_io_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved norm_io_comparison.png")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    set_seed(SEED)
    total_start = time.perf_counter()
    all_results = {"config": {
        "model": MODEL_NAME,
        "device": DEVICE,
        "seed": SEED,
        "calibration": str(CALIBRATION_PATH),
    }}

    # ── Section 1: Benchmark norms ──
    t0 = time.perf_counter()
    bench_results = benchmark_norms()
    all_results["benchmarks"] = bench_results
    all_results["timing_benchmarks_s"] = round(time.perf_counter() - t0, 1)

    # ── Load model ──
    print("\n" + "=" * 70)
    print("Loading model and calibration data...")
    print("=" * 70)
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    calibration_data = load_calibration_data(CALIBRATION_PATH, tokenizer, DEVICE)
    load_time = time.perf_counter() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  Layers: {len(model.model.layers)}, Hidden: {model.config.hidden_size}, "
          f"Head dim: {model.model.layers[0].self_attn.head_dim}")
    all_results["timing_load_s"] = round(load_time, 1)
    all_results["config"]["num_layers"] = len(model.model.layers)
    all_results["config"]["hidden_size"] = model.config.hidden_size
    all_results["config"]["head_dim"] = model.model.layers[0].self_attn.head_dim

    # ── Section 2: Activation statistics ──
    t0 = time.perf_counter()
    activation_stats = analyze_activation_stats(model, tokenizer, calibration_data)
    act_stats_json = {}
    for k, v in activation_stats.items():
        act_stats_json[k] = {str(kk): vv for kk, vv in v.items()}
    all_results["activation_stats"] = act_stats_json
    all_results["timing_activation_stats_s"] = round(time.perf_counter() - t0, 1)

    # ── Section 3: QK-norm ablation ──
    t0 = time.perf_counter()
    qk_results = ablate_qk_norm(model, tokenizer, calibration_data)
    if "qk_stats_baseline" in qk_results:
        for stat_key in ["qk_stats_baseline", "qk_stats_no_qknorm"]:
            if "per_layer" in qk_results[stat_key]:
                qk_results[stat_key]["per_layer"] = {
                    str(k): v for k, v in qk_results[stat_key]["per_layer"].items()
                }
    all_results["qk_norm_ablation"] = qk_results
    all_results["timing_qk_ablation_s"] = round(time.perf_counter() - t0, 1)

    # ── Section 4: Norm replacement ──
    t0 = time.perf_counter()
    replacement_results = replace_norms_experiment(model, tokenizer, calibration_data)
    for name, data in replacement_results["replacements"].items():
        if "activation_stats" in data:
            data["activation_stats"] = {str(k): v for k, v in data["activation_stats"].items()}
    all_results["norm_replacement"] = replacement_results
    all_results["timing_replacement_s"] = round(time.perf_counter() - t0, 1)

    # ── Section 5: Per-layer sensitivity ──
    t0 = time.perf_counter()
    per_layer_results = per_layer_norm_sensitivity(model, tokenizer, calibration_data)
    per_layer_results["per_layer"] = {str(k): v for k, v in per_layer_results["per_layer"].items()}
    all_results["per_layer_sensitivity"] = per_layer_results
    all_results["timing_per_layer_s"] = round(time.perf_counter() - t0, 1)

    # ── Section 6: Norm weight analysis ──
    t0 = time.perf_counter()
    norm_weights = analyze_norm_scale_distribution(model)
    norm_weights["layers"] = {str(k): v for k, v in norm_weights["layers"].items()}
    all_results["norm_weight_analysis"] = norm_weights
    all_results["timing_norm_weights_s"] = round(time.perf_counter() - t0, 1)

    # ── Section 7: Residual stream means ──
    t0 = time.perf_counter()
    residual_means = analyze_residual_means(model, tokenizer, calibration_data)
    residual_means_json = {str(k): v for k, v in residual_means.items()}
    all_results["residual_means"] = residual_means_json
    all_results["timing_residual_means_s"] = round(time.perf_counter() - t0, 1)

    # ── Section 8: Plots ──
    t0 = time.perf_counter()
    create_plots(bench_results, replacement_results, activation_stats,
                 qk_results, norm_weights, residual_means, per_layer_results)
    all_results["timing_plots_s"] = round(time.perf_counter() - t0, 1)

    # ── Save results ──
    total_time = time.perf_counter() - total_start
    all_results["timing_total_s"] = round(total_time, 1)

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'=' * 70}")
    print(f"All results saved to {RESULTS_DIR / 'results.json'}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
