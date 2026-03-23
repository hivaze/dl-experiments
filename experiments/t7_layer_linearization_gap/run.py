"""
Layer Linearization Gap (T-7)
==============================
How nonlinear is each layer's computation on real inputs?

For each transformer layer, we measure the gap between the actual nonlinear
layer output and its best linear (Jacobian-based) approximation. The layer
function is g(x) = layer(x) - x (the non-residual part). We use finite-
difference methods to estimate J@d for various perturbation directions d,
then measure how well the linear prediction tracks the actual output.

Methods:
  1. Perturbation gap: Perturb input by ε in random directions, compare
     actual displacement to linear (Jacobian-based) prediction. The ratio
     of 2nd-order to 1st-order error is the primary nonlinearity metric.
  2. Homogeneity gap: Scale input by (1±ε), compare g(x) to J@x. Dominated
     by RMSNorm scale-invariance — included for completeness.
  3. Attention vs MLP decomposition: Perturbation gap applied separately
     to the attention and MLP sublayers.
  4. Jacobian spectral properties: Spectral norm and mean amplification
     via finite-difference power iteration.
  5. Multi-scale analysis: Perturbation gap at multiple ε values to
     determine the dominant nonlinearity order per layer.
  6. Jacobian consistency: Measure how much the Jacobian varies across
     different inputs. High consistency = globally linear (one linear map
     works for all inputs). Low consistency = only locally linear.

Cross-references:
  - T-2 (layer knockout): correlate linearization gap with layer criticality
  - Method 7 (layer replacement): Jacobian consistency predicts whether
    a global linear replacement can capture a layer's function
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42
DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).parent / "results"
CALIBRATION_PATH = (
    Path(__file__).parents[2]
    / "data" / "text_completions" / "qwen3-4b-instruct-2507" / "completions.json"
)
T2_RESULTS_PATH = (
    Path(__file__).parents[2]
    / "experiments" / "t2_layer_knockout" / "results" / "results.json"
)

NUM_PERTURBATION_DIRS = 16       # random directions for perturbation gap
PERTURBATION_EPS = 0.05          # perturbation magnitude (bf16-safe: >= 0.01)
MAX_SEQ_LEN = 128                # truncate sequences for memory
MULTI_SCALE_EPS = [0.01, 0.02, 0.05, 0.1, 0.2]  # for nonlinearity order fitting
MULTI_SCALE_DIRS = 8             # fewer dirs per ε for speed
JACOBIAN_CONSISTENCY_PROMPTS = 10  # subset of prompts for Jacobian consistency
JACOBIAN_CONSISTENCY_DIRS = 8      # shared random directions

# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_calibration_data():
    """Load pre-generated completions from vLLM."""
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)
    completions = data["completions"]
    print(f"Loaded {len(completions)} completions from {CALIBRATION_PATH.name}")
    return completions


def load_t2_criticality():
    """Load T-2 layer knockout results for cross-referencing."""
    if not T2_RESULTS_PATH.exists():
        print("T-2 results not found, skipping cross-reference")
        return None
    with open(T2_RESULTS_PATH) as f:
        data = json.load(f)
    knockouts = data["single_knockouts"]["knockouts"]
    # Return dict: layer_idx -> loss_delta (criticality)
    return {int(k): v["loss_delta"] for k, v in knockouts.items()}


# ============================================================================
# Layer Function Wrappers
# ============================================================================

def make_layer_fn(layer, position_embeddings):
    """Create a function that maps hidden_state -> layer transform (non-residual part).

    The layer transform is: g(h) = layer(h)[0] - h
    where layer(h) is the full decoder layer forward pass.
    """
    def fn(hidden_states):
        # Decoder layer returns a plain tensor [batch, seq, hidden], not a tuple
        out = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        return out - hidden_states
    return fn


def make_attn_fn(layer, position_embeddings):
    """Function for just the attention sublayer contribution."""
    def fn(hidden_states):
        normed = layer.input_layernorm(hidden_states)
        attn_out, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=None,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        return attn_out  # just the attention delta
    return fn


def make_mlp_fn(layer, position_embeddings):
    """Function for just the MLP sublayer contribution.
    MLP input = h + attn_out, so we need to run attention first.
    """
    def fn(hidden_states):
        normed = layer.input_layernorm(hidden_states)
        attn_out, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=None,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        h_after_attn = hidden_states + attn_out
        normed2 = layer.post_attention_layernorm(h_after_attn)
        mlp_out = layer.mlp(normed2)
        return mlp_out  # just the MLP delta
    return fn


# ============================================================================
# Linearization Gap Computation
# ============================================================================

def compute_linearization_gap(fn, hidden_states, eps=0.05):
    """Compute linearization gap: ||g(x) - J@x|| / ||g(x)||.

    For a linear function g(x) = Ax, we have g(x) = J@x exactly.
    We estimate J@x via finite-difference: J@x ≈ [g(x+εv) - g(x-εv)] / (2ε)
    where v = x (so we perturb along the input direction itself).

    Uses float32 intermediates to avoid bfloat16 precision loss.
    The eps must be large enough to survive bf16 quantization: bf16 has
    ~7.8e-3 relative precision, so eps >= 0.05 is safe.
    """
    with torch.no_grad():
        g_x = fn(hidden_states).float()

        # Additive perturbation scaled by input norm to ensure it's representable in bf16
        # v = x, perturb: x ± eps*x = x*(1 ± eps)
        # Use separate scaling factors that survive bf16
        scale_plus = torch.tensor(1.0 + eps, dtype=hidden_states.dtype, device=hidden_states.device)
        scale_minus = torch.tensor(1.0 - eps, dtype=hidden_states.dtype, device=hidden_states.device)

        g_plus = fn(hidden_states * scale_plus).float()
        g_minus = fn(hidden_states * scale_minus).float()
        jx_est = (g_plus - g_minus) / (2.0 * eps)

    diff = g_x - jx_est
    gap_per_token = torch.norm(diff, dim=-1) / (torch.norm(g_x, dim=-1) + 1e-10)
    return gap_per_token.squeeze(0).detach()


def compute_perturbation_gap(fn, hidden_states, num_dirs, eps):
    """Compute linearization gap via finite-difference perturbation in random directions.

    For random unit directions d_i, perturb by eps * ||x|| * d_i (scaled to
    survive bf16 quantization). Measures ratio of 2nd-order to 1st-order
    response: gap = ||g(x+d) - g(x) - J@d|| / ||g(x+d) - g(x)||.
    For a linear function, gap = 0. For quadratic nonlinearity, gap ~ eps.
    """
    with torch.no_grad():
        g_x = fn(hidden_states).float()

    total_gap = torch.zeros(hidden_states.shape[1], device=hidden_states.device)
    input_scale = torch.norm(hidden_states, dim=-1, keepdim=True) + 1e-10

    for _ in range(num_dirs):
        d = torch.randn_like(hidden_states)
        # Scale d to have magnitude eps * ||x|| / sqrt(dim) per token
        d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-10)
        d = d * input_scale * eps

        with torch.no_grad():
            g_plus = fn(hidden_states + d).float()
            g_minus = fn(hidden_states - d).float()

        jd_est = (g_plus - g_minus) / 2.0
        actual_delta = g_plus - g_x
        linear_delta = jd_est  # eps already baked into d

        err = torch.norm(actual_delta - linear_delta, dim=-1)
        scale = torch.norm(actual_delta, dim=-1) + 1e-10
        total_gap += (err / scale).squeeze(0)

    return total_gap / num_dirs


def compute_jacobian_stats(fn, hidden_states, num_dirs=16, eps=0.05):
    """Estimate Jacobian spectral properties via finite differences.

    Uses bf16-safe perturbation magnitudes.
    """
    batch, seq, dim = hidden_states.shape
    device = hidden_states.device
    input_scale = torch.norm(hidden_states, dim=-1, keepdim=True) + 1e-10

    # Power iteration for spectral norm (5 iters)
    v = torch.randn(batch, seq, dim, device=device, dtype=hidden_states.dtype)
    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)

    for _ in range(5):
        # Scale perturbation to be representable in bf16
        delta = v * input_scale * eps
        with torch.no_grad():
            jv = (fn(hidden_states + delta).float() - fn(hidden_states - delta).float()) / 2.0
        # Undo the eps*input_scale factor to get true J@v
        jv = jv / (input_scale.float() * eps)
        norms = torch.norm(jv, dim=-1, keepdim=True) + 1e-10
        v = (jv / norms).to(hidden_states.dtype)

    delta = v * input_scale * eps
    with torch.no_grad():
        jv = (fn(hidden_states + delta).float() - fn(hidden_states - delta).float()) / 2.0
    jv = jv / (input_scale.float() * eps)
    spectral_norm_est = torch.norm(jv, dim=-1) / (torch.norm(v.float(), dim=-1) + 1e-10)

    # Mean amplification
    amp_samples = []
    for _ in range(num_dirs):
        d = torch.randn_like(hidden_states)
        d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-10)
        delta = d * input_scale * eps
        with torch.no_grad():
            jd = (fn(hidden_states + delta).float() - fn(hidden_states - delta).float()) / 2.0
        jd = jd / (input_scale.float() * eps)
        amp = torch.norm(jd, dim=-1)  # d is unit norm
        amp_samples.append(amp)

    mean_amp = torch.stack(amp_samples).mean(0)
    return spectral_norm_est.squeeze(0).detach(), mean_amp.squeeze(0).detach()


def compute_multiscale_gap(fn, hidden_states, eps_values, num_dirs):
    """Compute perturbation gap at multiple ε scales to determine nonlinearity order.

    For a layer with dominant degree-k nonlinearity, the perturbation gap
    scales as gap ~ ε^(k-1). Fitting log(gap) vs log(ε) gives the slope
    (k-1), so we can identify whether the nonlinearity is quadratic (slope=1),
    cubic (slope=2), etc.

    Returns dict with per-ε gap values and fitted slope.
    """
    input_scale = torch.norm(hidden_states, dim=-1, keepdim=True) + 1e-10
    gaps_by_eps = {}

    for eps in eps_values:
        with torch.no_grad():
            g_x = fn(hidden_states).float()

        total_gap = torch.zeros(hidden_states.shape[1], device=hidden_states.device)

        for _ in range(num_dirs):
            d = torch.randn_like(hidden_states)
            d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-10)
            d = d * input_scale * eps

            with torch.no_grad():
                g_plus = fn(hidden_states + d).float()
                g_minus = fn(hidden_states - d).float()

            jd_est = (g_plus - g_minus) / 2.0
            actual_delta = g_plus - g_x
            linear_delta = jd_est

            err = torch.norm(actual_delta - linear_delta, dim=-1)
            scale = torch.norm(actual_delta, dim=-1) + 1e-10
            total_gap += (err / scale).squeeze(0)

        gaps_by_eps[eps] = float((total_gap / num_dirs).mean())

    # Fit log-log slope: log(gap) = α + β*log(ε)
    log_eps = np.log(list(gaps_by_eps.keys()))
    log_gap = np.log(np.clip(list(gaps_by_eps.values()), 1e-10, None))
    if len(log_eps) >= 2:
        slope, intercept = np.polyfit(log_eps, log_gap, 1)
    else:
        slope, intercept = 0.0, 0.0

    return {
        "gaps_by_eps": gaps_by_eps,
        "nonlinearity_order": float(slope + 1),  # gap ~ ε^(k-1) → slope = k-1
        "log_log_slope": float(slope),
        "log_log_r2": float(np.corrcoef(log_eps, log_gap)[0, 1] ** 2) if len(log_eps) >= 2 else 0.0,
    }


# ============================================================================
# Part 6: Jacobian Consistency Across Inputs
# ============================================================================

def compute_jacobian_consistency(model, tokenizer, calibration_data, num_layers):
    """Measure how much each layer's Jacobian varies across different inputs.

    For each layer, computes JVPs (Jacobian-vector products) in shared random
    directions at multiple different inputs, then measures cosine similarity
    of the normalized JVP outputs across inputs.

    High consistency (→1.0): the Jacobian is approximately the same matrix
    at all inputs → the layer is globally linearizable (a single W works).
    Low consistency (→0.0): the Jacobian varies strongly with input content
    → the layer is only locally linear (each input needs a different W).

    This directly predicts the success of Method 7's global linear
    replacement experiment.
    """
    hidden_dim = model.config.hidden_size
    device = DEVICE

    # Generate shared random directions (fixed across all prompts)
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + 7777)
    shared_dirs = []
    for _ in range(JACOBIAN_CONSISTENCY_DIRS):
        d = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.bfloat16,
                         generator=gen)
        d = d / (d.norm() + 1e-10)
        shared_dirs.append(d)

    prompts_subset = calibration_data[:JACOBIAN_CONSISTENCY_PROMPTS]

    # Collect JVPs for each layer across prompts
    # layer_jvps[layer_idx] = list of (num_dirs, hidden_dim) tensors, one per prompt
    layer_jvps = {i: [] for i in range(num_layers)}

    for entry in tqdm(prompts_subset, desc="Jacobian consistency"):
        full_text = entry["full_text"]
        prompt_len = entry["prompt_token_count"]
        tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(device)
        seq_len = tokens["input_ids"].shape[1]

        # Forward pass to collect hidden states
        hidden_states_by_layer = {}

        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs):
                if args:
                    hidden_states_by_layer[layer_idx] = args[0].detach().clone()
            return hook_fn

        hooks = []
        for i in range(num_layers):
            h = model.model.layers[i].register_forward_pre_hook(
                make_hook(i), with_kwargs=True)
            hooks.append(h)

        with torch.no_grad():
            model(**tokens, use_cache=False)
        for h in hooks:
            h.remove()

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            h = hidden_states_by_layer[layer_idx]
            h_comp = h[:, prompt_len:, :]
            pos_comp = position_ids[:, prompt_len:]
            n_comp = h_comp.shape[1]
            if n_comp < 2:
                continue

            pos_emb = model.model.rotary_emb(h_comp, position_ids=pos_comp)
            layer_fn = make_layer_fn(layer, pos_emb)
            input_scale = h_comp.norm(dim=-1, keepdim=True) + 1e-10
            last_pos = n_comp - 1

            prompt_jvps = []
            for d_unit in shared_dirs:
                d_scaled = d_unit.expand_as(h_comp) * input_scale * PERTURBATION_EPS
                with torch.no_grad():
                    g_plus = layer_fn(h_comp + d_scaled).float()
                    g_minus = layer_fn(h_comp - d_scaled).float()
                jvp = (g_plus - g_minus) / 2.0
                jvp_last = jvp[0, last_pos, :]
                jvp_last = jvp_last / (jvp_last.norm() + 1e-10)
                prompt_jvps.append(jvp_last.cpu())

            layer_jvps[layer_idx].append(torch.stack(prompt_jvps))

        del hidden_states_by_layer
        torch.cuda.empty_cache()

    # Compute consistency: pairwise cosine similarity of JVPs across prompts
    results = {}
    for layer_idx in range(num_layers):
        jvps_list = layer_jvps[layer_idx]
        if len(jvps_list) < 2:
            results[layer_idx] = {
                "consistency_mean": float("nan"),
                "consistency_std": float("nan"),
            }
            continue

        K = len(jvps_list)
        D = jvps_list[0].shape[0]

        per_dir_consistency = []
        for d_idx in range(D):
            jvp_vectors = torch.stack([jvps_list[k][d_idx] for k in range(K)])
            # Pairwise cosine similarity (vectors are already normalized)
            cos_sims = []
            for i in range(K):
                for j in range(i + 1, K):
                    cos = torch.dot(jvp_vectors[i], jvp_vectors[j]).item()
                    cos_sims.append(cos)
            per_dir_consistency.append(float(np.mean(cos_sims)))

        results[layer_idx] = {
            "consistency_mean": float(np.mean(per_dir_consistency)),
            "consistency_std": float(np.std(per_dir_consistency)),
        }

    return results


# ============================================================================
# Main Analysis
# ============================================================================

def analyze_model(model, tokenizer, calibration_data):
    """Run linearization gap analysis for all layers on calibration data."""
    num_layers = len(model.model.layers)

    # Results accumulators per layer
    all_results = {i: {
        "homogeneity_gaps": [],      # per-prompt mean homogeneity gap
        "perturb_gaps": [],          # per-prompt mean perturbation gap
        "attn_gaps": [],             # attention sublayer perturbation gap
        "mlp_gaps": [],              # MLP sublayer perturbation gap
        "spectral_norms": [],        # Jacobian spectral norm estimate
        "mean_amps": [],             # mean Jacobian amplification
        "transform_norms": [],       # ||g(x)|| (layer transform magnitude)
        "input_norms": [],           # ||x|| (input magnitude)
        "multiscale": [],            # multi-scale analysis dicts
        "num_tokens": [],
    } for i in range(num_layers)}

    for entry_idx, entry in enumerate(tqdm(calibration_data, desc="Processing prompts")):
        full_text = entry["full_text"]
        prompt_len = entry["prompt_token_count"]

        tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(DEVICE)
        seq_len = tokens["input_ids"].shape[1]

        # Forward pass to collect all hidden states via hooks
        hidden_states_by_layer = {}

        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs):
                # Input to the layer is the first positional arg
                if args:
                    hidden_states_by_layer[layer_idx] = args[0].detach().clone()
            return hook_fn

        hooks = []
        for i in range(num_layers):
            h = model.model.layers[i].register_forward_pre_hook(
                make_hook(i), with_kwargs=True
            )
            hooks.append(h)

        with torch.no_grad():
            outputs = model(**tokens, use_cache=False)

        for h in hooks:
            h.remove()

        # Compute rotary position embeddings for completion tokens
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

        # Analyze each layer
        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            h = hidden_states_by_layer[layer_idx]  # [1, seq, hidden]

            # Only analyze completion tokens
            h_comp = h[:, prompt_len:, :]
            pos_comp = position_ids[:, prompt_len:]
            n_comp = h_comp.shape[1]
            if n_comp < 2:
                continue

            # Compute position embeddings (cos, sin) for completion positions
            pos_emb = model.model.rotary_emb(h_comp, position_ids=pos_comp)

            # Create layer functions for completion tokens
            layer_fn = make_layer_fn(layer, pos_emb)
            attn_fn = make_attn_fn(layer, pos_emb)
            mlp_fn = make_mlp_fn(layer, pos_emb)

            # 1. Homogeneity gap (scaling test)
            homogeneity_gap = compute_linearization_gap(layer_fn, h_comp)

            # 2. Perturbation gap (primary nonlinearity metric)
            perturb_gap = compute_perturbation_gap(
                layer_fn, h_comp, NUM_PERTURBATION_DIRS, PERTURBATION_EPS
            )

            # 3. Attention vs MLP decomposition (perturbation-based)
            attn_gap = compute_perturbation_gap(
                attn_fn, h_comp, NUM_PERTURBATION_DIRS, PERTURBATION_EPS
            )
            mlp_gap = compute_perturbation_gap(
                mlp_fn, h_comp, NUM_PERTURBATION_DIRS, PERTURBATION_EPS
            )

            # 4. Jacobian spectral properties
            spec_norm, mean_amp = compute_jacobian_stats(layer_fn, h_comp)

            # 5. Transform and input norms
            with torch.no_grad():
                g_x = layer_fn(h_comp)
            transform_norm = torch.norm(g_x, dim=-1).squeeze(0)
            input_norm = torch.norm(h_comp, dim=-1).squeeze(0)

            # 6. Multi-scale analysis (nonlinearity order)
            ms = compute_multiscale_gap(
                layer_fn, h_comp, MULTI_SCALE_EPS, MULTI_SCALE_DIRS
            )

            # Store means
            res = all_results[layer_idx]
            res["homogeneity_gaps"].append(float(homogeneity_gap.mean()))
            res["perturb_gaps"].append(float(perturb_gap.mean()))
            res["attn_gaps"].append(float(attn_gap.mean()))
            res["mlp_gaps"].append(float(mlp_gap.mean()))
            res["spectral_norms"].append(float(spec_norm.mean()))
            res["mean_amps"].append(float(mean_amp.mean()))
            res["transform_norms"].append(float(transform_norm.mean()))
            res["input_norms"].append(float(input_norm.mean()))
            res["multiscale"].append(ms)
            res["num_tokens"].append(n_comp)

        del hidden_states_by_layer
        torch.cuda.empty_cache()

    # Aggregate across prompts
    summary = {}
    for layer_idx in range(num_layers):
        res = all_results[layer_idx]
        if not res["homogeneity_gaps"]:
            continue
        # Weighted mean by number of tokens
        weights = np.array(res["num_tokens"], dtype=np.float64)
        weights /= weights.sum()

        def wmean(vals):
            return float(np.average(vals, weights=weights))

        def wstd(vals):
            m = np.average(vals, weights=weights)
            return float(np.sqrt(np.average((np.array(vals) - m) ** 2, weights=weights)))

        # Aggregate multi-scale results: weighted mean of per-prompt values
        ms_agg = {}
        for eps_val in MULTI_SCALE_EPS:
            eps_gaps = [m["gaps_by_eps"][eps_val] for m in res["multiscale"]]
            ms_agg[eps_val] = wmean(eps_gaps)
        ms_orders = [m["nonlinearity_order"] for m in res["multiscale"]]
        ms_slopes = [m["log_log_slope"] for m in res["multiscale"]]
        ms_r2s = [m["log_log_r2"] for m in res["multiscale"]]

        summary[layer_idx] = {
            "homogeneity_gap_mean": wmean(res["homogeneity_gaps"]),
            "homogeneity_gap_std": wstd(res["homogeneity_gaps"]),
            "perturb_gap_mean": wmean(res["perturb_gaps"]),
            "perturb_gap_std": wstd(res["perturb_gaps"]),
            "attn_gap_mean": wmean(res["attn_gaps"]),
            "attn_gap_std": wstd(res["attn_gaps"]),
            "mlp_gap_mean": wmean(res["mlp_gaps"]),
            "mlp_gap_std": wstd(res["mlp_gaps"]),
            "spectral_norm_mean": wmean(res["spectral_norms"]),
            "mean_amplification": wmean(res["mean_amps"]),
            "transform_norm_mean": wmean(res["transform_norms"]),
            "input_norm_mean": wmean(res["input_norms"]),
            "transform_to_input_ratio": wmean(res["transform_norms"]) / (wmean(res["input_norms"]) + 1e-10),
            "multiscale_gaps": ms_agg,
            "nonlinearity_order_mean": wmean(ms_orders),
            "nonlinearity_order_std": wstd(ms_orders),
            "log_log_slope_mean": wmean(ms_slopes),
            "log_log_r2_mean": wmean(ms_r2s),
            "total_tokens": int(sum(res["num_tokens"])),
            "per_prompt_homogeneity_gaps": res["homogeneity_gaps"],
            "per_prompt_perturb_gaps": res["perturb_gaps"],
        }

    return summary, num_layers


# ============================================================================
# Method 7: Global Linear Replacement
# ============================================================================

def compute_loss(model, tokenizer, calibration_data, device):
    """Compute mean cross-entropy loss on completion tokens only."""
    total_loss = 0.0
    total_tokens = 0
    for entry in calibration_data:
        full_text = entry["full_text"]
        prompt_token_count = entry["prompt_token_count"]
        tokens = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]
        if seq_len < 2 or prompt_token_count >= seq_len:
            continue
        with torch.no_grad():
            outputs = model(**tokens, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        start = max(prompt_token_count - 1, 0)
        comp_logits = logits[:, start:, :]
        comp_targets = targets[:, start:]
        if comp_targets.numel() == 0:
            continue
        loss = F.cross_entropy(comp_logits.reshape(-1, comp_logits.size(-1)),
                               comp_targets.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += comp_targets.numel()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def collect_layer_activations(model, tokenizer, layer_idx, calibration_data, device):
    """Collect (input, output) activation pairs for a given layer across
    calibration data. Returns input and output tensors of shape (total_tokens, hidden_dim)."""
    inputs_list = []
    outputs_list = []

    def capture_input_hook(module, inp, out):
        h_in = inp[0].detach()
        if isinstance(out, tuple):
            h_out = out[0].detach()
        else:
            h_out = out.detach()
        inputs_list.append(h_in.squeeze(0))
        outputs_list.append(h_out.squeeze(0))

    hook = model.model.layers[layer_idx].register_forward_hook(capture_input_hook)

    with torch.no_grad():
        for entry in calibration_data:
            tokens = tokenizer(entry["full_text"], return_tensors="pt").to(device)
            model(**tokens, use_cache=False)

    hook.remove()

    all_inputs = torch.cat(inputs_list, dim=0)
    all_outputs = torch.cat(outputs_list, dim=0)
    return all_inputs, all_outputs


def fit_linear_replacement(inputs, outputs, rank=None):
    """Fit a linear map W such that outputs ≈ W @ inputs via least-squares.

    For full-rank: solves W = argmin ||Y - WX||_F^2 directly.
    For low-rank: fits the RESIDUAL component R = Y - X, so the replacement
    is Y ≈ X + W_r @ X where W_r is truncated to the target rank. This
    preserves the skip connection (identity).

    All computation done in float32 for numerical stability.
    Returns W in bfloat16, shape (hidden, hidden).
    """
    X = inputs.float()
    Y = outputs.float()

    if rank is not None and rank < X.shape[1]:
        R = Y - X
        result = torch.linalg.lstsq(X, R)
        W_r = result.solution.T

        U, S, Vh = torch.linalg.svd(W_r, full_matrices=False)
        W_r_lr = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]

        W = torch.eye(X.shape[1], device=X.device) + W_r_lr

        Y_pred = X @ W.T
        residual = (Y - Y_pred).norm() / Y.norm()
    else:
        result = torch.linalg.lstsq(X, Y)
        W = result.solution.T

        Y_pred = X @ W.T
        residual = (Y - Y_pred).norm() / Y.norm()

    return W.to(torch.bfloat16), residual.item()


def evaluate_with_replacement(model, tokenizer, layer_idx, W_replacement,
                               calibration_data, device):
    """Replace a layer with a linear map and measure loss."""
    def replacement_hook(module, inp, out):
        h_in = inp[0]
        h_out = torch.einsum("ij,bsj->bsi", W_replacement, h_in)
        if isinstance(out, tuple):
            return (h_out,) + out[1:]
        return h_out

    hook = model.model.layers[layer_idx].register_forward_hook(replacement_hook)
    loss = compute_loss(model, tokenizer, calibration_data, device)
    hook.remove()
    return loss


def run_layer_replacements(model, tokenizer, num_layers, single_results,
                           calibration_data):
    """For each layer, train a linear surrogate and measure loss recovery
    vs knockout. Tests full-rank and low-rank (rank 64, 256) replacements."""
    print("\n" + "=" * 70)
    print("METHOD 7: Global Linear Replacement")
    print("=" * 70)

    # Recompute baseline on current calibration data for consistency
    print("  Computing baseline loss on current calibration data...")
    baseline_loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Select layers to replace: top-5 critical, bottom-5 least critical, plus mid-range
    sorted_by_impact = sorted(single_results["knockouts"].items(),
                              key=lambda x: x[1]["loss_delta"], reverse=True)
    top5 = [int(idx) for idx, _ in sorted_by_impact[:5]]
    bot5 = [int(idx) for idx, _ in sorted_by_impact[-5:]]
    mid_layers = [int(idx) for idx, _ in sorted_by_impact[15:20]]
    target_layers = sorted(set(top5 + bot5 + mid_layers))

    ranks_to_test = [64, 256, None]  # None = full rank
    results = {"baseline_loss": baseline_loss, "replacements": {}}

    # Recompute knockout losses for target layers on current calibration data
    print(f"  Recomputing knockout losses for {len(target_layers)} target layers...")
    original_layers = list(model.model.layers)
    knockout_losses = {}
    for skip_idx in tqdm(target_layers, desc="  Knockout re-eval"):
        remaining = [l for i, l in enumerate(original_layers) if i != skip_idx]
        model.model.layers = torch.nn.ModuleList(remaining)
        knockout_losses[skip_idx] = compute_loss(model, tokenizer, calibration_data, DEVICE)
    model.model.layers = torch.nn.ModuleList(original_layers)

    for layer_idx in tqdm(target_layers, desc="Layer replacements"):
        print(f"\n  Layer {layer_idx}:")

        t0 = time.time()
        inputs, outputs = collect_layer_activations(
            model, tokenizer, layer_idx, calibration_data, DEVICE)
        t_collect = time.time() - t0
        print(f"    Collected {inputs.shape[0]} token activations ({t_collect:.1f}s)")

        knockout_loss = knockout_losses[layer_idx]
        knockout_delta = knockout_loss - baseline_loss

        layer_results = {
            "knockout_loss": knockout_loss,
            "knockout_delta": knockout_delta,
            "replacements": {},
        }

        for rank in ranks_to_test:
            rank_label = f"rank_{rank}" if rank else "full_rank"

            t0 = time.time()
            W, fit_residual = fit_linear_replacement(
                inputs.to("cpu"), outputs.to("cpu"), rank=rank)
            W = W.to(DEVICE)
            t_fit = time.time() - t0

            replacement_loss = evaluate_with_replacement(
                model, tokenizer, layer_idx, W, calibration_data, DEVICE)
            replacement_delta = replacement_loss - baseline_loss
            recovery = 1.0 - (replacement_delta / knockout_delta) if knockout_delta > 0 else 0.0

            layer_results["replacements"][rank_label] = {
                "loss": replacement_loss,
                "loss_delta": replacement_delta,
                "loss_ratio": replacement_loss / baseline_loss,
                "recovery_vs_knockout": recovery,
                "fit_residual_norm": fit_residual,
                "rank": rank if rank else inputs.shape[1],
                "fit_time_s": t_fit,
            }

            rank_str = f"rank-{rank}" if rank else "full-rank"
            print(f"    {rank_str}: loss={replacement_loss:.4f} "
                  f"(Δ={replacement_delta:+.4f}, "
                  f"recovery={recovery:.1%}, "
                  f"fit_residual={fit_residual:.4f})")

        results["replacements"][layer_idx] = layer_results

        del inputs, outputs, W
        torch.cuda.empty_cache()

    # Summary
    print(f"\n--- Layer Replacement Summary ---")
    print(f"{'Layer':>6} {'Knockout':>10} {'FullRank':>10} {'Rank256':>10} "
          f"{'Rank64':>10} {'Recovery(FR)':>12}")
    for layer_idx in target_layers:
        lr = results["replacements"][layer_idx]
        ko = lr["knockout_delta"]
        fr = lr["replacements"].get("full_rank", {}).get("loss_delta", float("nan"))
        r256 = lr["replacements"].get("rank_256", {}).get("loss_delta", float("nan"))
        r64 = lr["replacements"].get("rank_64", {}).get("loss_delta", float("nan"))
        rec = lr["replacements"].get("full_rank", {}).get("recovery_vs_knockout", 0)
        print(f"  L{layer_idx:>3} {ko:>10.4f} {fr:>10.4f} {r256:>10.4f} "
              f"{r64:>10.4f} {rec:>11.1%}")

    return results


# ============================================================================
# Visualization
# ============================================================================

def create_plots(summary, num_layers, t2_criticality, jacobian_consistency=None):
    """Generate all visualization plots."""
    layers = sorted(summary.keys())
    x = np.array(layers)

    def get_vals(key):
        return [summary[l][key] for l in layers]

    # --- Plot 1: Main Linearization Gap (3-panel) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Layer Linearization Gap — Qwen3-4B-Instruct-2507", fontsize=14)

    # A: Perturbation gap (primary nonlinearity metric)
    ax = axes[0]
    means_p = get_vals("perturb_gap_mean")
    stds_p = get_vals("perturb_gap_std")
    ax.plot(x, means_p, "r-o", markersize=4, label="Perturbation gap")
    ax.fill_between(x, np.array(means_p) - np.array(stds_p),
                     np.array(means_p) + np.array(stds_p), alpha=0.2, color="red")
    ax.set_xlabel("Layer")
    ax.set_ylabel("2nd-order / 1st-order Error Ratio")
    ax.set_title("A. Nonlinearity (Perturbation Gap)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # B: Homogeneity gap (scaling test, dominated by RMSNorm)
    ax = axes[1]
    means = get_vals("homogeneity_gap_mean")
    stds = get_vals("homogeneity_gap_std")
    ax.plot(x, means, "b-s", markersize=4, label="Homogeneity gap")
    ax.fill_between(x, np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds), alpha=0.2, color="blue")
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="g(x)=0 baseline")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||g(x) - J@x|| / ||g(x)||")
    ax.set_title("B. Homogeneity Gap (scale-invariance)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # C: Attention vs MLP nonlinearity (perturbation-based)
    ax = axes[2]
    ax.plot(x, get_vals("attn_gap_mean"), "g-o", markersize=4, label="Attention")
    ax.plot(x, get_vals("mlp_gap_mean"), "m-s", markersize=4, label="MLP")
    ax.plot(x, means_p, "r-^", markersize=3, alpha=0.5, label="Full layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Perturbation Gap")
    ax.set_title("C. Attention vs MLP Nonlinearity")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "linearization_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved linearization_gap.png")

    # --- Plot 2: Jacobian Properties ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Jacobian Properties Across Depth", fontsize=14)

    # A: Spectral norm
    ax = axes[0]
    ax.plot(x, get_vals("spectral_norm_mean"), "k-o", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Estimated Spectral Norm")
    ax.set_title("A. Jacobian Spectral Norm")
    ax.grid(True, alpha=0.3)

    # B: Mean amplification
    ax = axes[1]
    ax.plot(x, get_vals("mean_amplification"), "b-o", markersize=4)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="Unity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean ||J@d|| / ||d||")
    ax.set_title("B. Mean Jacobian Amplification")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # C: Transform norm / input norm ratio
    ax = axes[2]
    ax.plot(x, get_vals("transform_to_input_ratio"), "r-o", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("||g(x)|| / ||x||")
    ax.set_title("C. Layer Transform Magnitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "jacobian_properties.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved jacobian_properties.png")

    # --- Plot 3: Cross-reference with T-2 criticality ---
    if t2_criticality is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Linearization Gap vs Layer Criticality (T-2)", fontsize=14)

        # Get matching layers
        common_layers = sorted(set(layers) & set(t2_criticality.keys()))
        gaps = [summary[l]["perturb_gap_mean"] for l in common_layers]
        crits = [t2_criticality[l] for l in common_layers]

        # A: Overlay plot
        ax = axes[0]
        ax2 = ax.twinx()
        ax.plot(common_layers, gaps, "b-o", markersize=4, label="Linearization gap")
        ax2.plot(common_layers, crits, "r-s", markersize=4, label="Knockout loss delta")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Linearization Gap", color="b")
        ax2.set_ylabel("Knockout Loss Delta", color="r")
        ax.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax.set_title("A. Gap & Criticality Across Depth")
        ax.grid(True, alpha=0.3)

        # B: Scatter plot
        ax = axes[1]
        ax.scatter(gaps, crits, c=common_layers, cmap="viridis", s=40, zorder=5)
        for l, g, c in zip(common_layers, gaps, crits):
            ax.annotate(str(l), (g, c), fontsize=6, alpha=0.7)
        ax.set_xlabel("Linearization Gap")
        ax.set_ylabel("Knockout Loss Delta (Criticality)")
        ax.set_title("B. Gap vs Criticality Scatter")
        ax.grid(True, alpha=0.3)

        # Compute correlation
        corr = np.corrcoef(gaps, crits)[0, 1]
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
                fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "gap_vs_criticality.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved gap_vs_criticality.png")

    # --- Plot 4: Per-prompt variability heatmap ---
    if "per_prompt_perturb_gaps" in summary.get(layers[0], {}):
        fig, ax = plt.subplots(figsize=(14, 6))
        per_prompt_matrix = []
        for l in layers:
            per_prompt_matrix.append(summary[l]["per_prompt_perturb_gaps"])
        per_prompt_matrix = np.array(per_prompt_matrix)

        im = ax.imshow(per_prompt_matrix, aspect="auto", cmap="YlOrRd",
                        extent=[-0.5, per_prompt_matrix.shape[1] - 0.5,
                                layers[-1] + 0.5, layers[0] - 0.5])
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Layer")
        ax.set_title("Perturbation Gap by Layer x Prompt — Qwen3-4B-Instruct-2507")
        plt.colorbar(im, ax=ax, label="Perturbation Gap", fraction=0.046, pad=0.04)
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "gap_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved gap_heatmap.png")
    else:
        print("Skipped gap_heatmap.png (per-prompt data not available)")

    # --- Plot 5: Multi-scale analysis ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Multi-Scale Nonlinearity Analysis — Qwen3-4B-Instruct-2507", fontsize=14)

    # A: Gap vs epsilon for selected layers
    ax = axes[0]
    sample_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_layers)))
    for sl, color in zip(sample_layers, colors):
        if sl not in summary:
            continue
        ms = summary[sl]["multiscale_gaps"]
        eps_vals = sorted(ms.keys())
        gap_vals = [ms[e] for e in eps_vals]
        ax.loglog(eps_vals, gap_vals, "o-", color=color, markersize=5,
                  label=f"Layer {sl} (order={summary[sl]['nonlinearity_order_mean']:.1f})")
    # Reference slopes
    eps_ref = np.array([0.01, 0.2])
    ax.loglog(eps_ref, 0.3 * eps_ref, "k--", alpha=0.3, label="slope=1 (quadratic)")
    ax.loglog(eps_ref, 3.0 * eps_ref**2, "k:", alpha=0.3, label="slope=2 (cubic)")
    ax.set_xlabel("Perturbation scale (ε)")
    ax.set_ylabel("Perturbation gap")
    ax.set_title("A. Gap vs ε (log-log)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # B: Nonlinearity order across depth
    ax = axes[1]
    orders = get_vals("nonlinearity_order_mean")
    order_stds = get_vals("nonlinearity_order_std")
    ax.plot(x, orders, "b-o", markersize=4)
    ax.fill_between(x, np.array(orders) - np.array(order_stds),
                     np.array(orders) + np.array(order_stds), alpha=0.2, color="blue")
    ax.axhline(y=2.0, color="k", linestyle="--", alpha=0.3, label="Quadratic (order=2)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Estimated Nonlinearity Order")
    ax.set_title("B. Nonlinearity Order Across Depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # C: R² of log-log fit (how well power-law holds)
    ax = axes[2]
    r2s = get_vals("log_log_r2_mean")
    ax.plot(x, r2s, "g-o", markersize=4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² of log(gap) vs log(ε)")
    ax.set_title("C. Power-Law Fit Quality")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "multiscale_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved multiscale_analysis.png")

    # --- Plot 6: Jacobian Consistency ---
    if jacobian_consistency is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle("Jacobian Consistency Across Inputs — Qwen3-4B-Instruct-2507",
                      fontsize=14)

        jc_layers = sorted(jacobian_consistency.keys())
        jc_means = [jacobian_consistency[l]["consistency_mean"] for l in jc_layers]
        jc_stds = [jacobian_consistency[l].get("consistency_std", 0) for l in jc_layers]

        # A: Consistency across depth
        ax = axes[0]
        ax.plot(jc_layers, jc_means, "b-o", markersize=4, linewidth=1.5)
        ax.fill_between(jc_layers,
                         np.array(jc_means) - np.array(jc_stds),
                         np.array(jc_means) + np.array(jc_stds),
                         alpha=0.2, color="blue")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Pairwise Cosine Similarity of JVPs")
        ax.set_title("A. Jacobian Consistency Across Depth")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="Perfect consistency")

        # Annotate extremes
        if jc_means:
            top3 = sorted(range(len(jc_means)), key=lambda i: jc_means[i], reverse=True)[:3]
            bot3 = sorted(range(len(jc_means)), key=lambda i: jc_means[i])[:3]
            for idx in top3:
                ax.annotate(f"L{jc_layers[idx]}", (jc_layers[idx], jc_means[idx]),
                            fontsize=7, ha="center", va="bottom", color="#1B5E20")
            for idx in bot3:
                ax.annotate(f"L{jc_layers[idx]}", (jc_layers[idx], jc_means[idx]),
                            fontsize=7, ha="center", va="top", color="#B71C1C")
        ax.legend(fontsize=8)

        # B: Consistency vs perturbation gap (local vs global linearity)
        ax = axes[1]
        common = sorted(set(jc_layers) & set(summary.keys()))
        if common:
            gaps = [summary[l]["perturb_gap_mean"] for l in common]
            cons = [jacobian_consistency[l]["consistency_mean"] for l in common]
            scatter = ax.scatter(gaps, cons, c=common, cmap="viridis", s=50, zorder=5)
            for l, g, c in zip(common, gaps, cons):
                ax.annotate(str(l), (g, c), fontsize=6, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label="Layer index", shrink=0.8)
            corr = np.corrcoef(gaps, cons)[0, 1]
            ax.text(0.05, 0.05, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
                    fontsize=10, verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))
        ax.set_xlabel("Perturbation Gap (Local Nonlinearity)")
        ax.set_ylabel("Jacobian Consistency (Global Linearity)")
        ax.set_title("B. Local Nonlinearity vs Global Consistency")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(RESULTS_DIR / "jacobian_consistency.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved jacobian_consistency.png")


def create_replacement_plot(replacement_results):
    """Generate layer replacement visualization (Method 7)."""
    if replacement_results is None:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = replacement_results["replacements"]
    rep_layers = sorted(rep.keys(), key=lambda k: int(k))
    model_short = MODEL_NAME.split("/")[-1]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    x = np.arange(len(rep_layers))
    width = 0.2
    knockout_deltas = [rep[l]["knockout_delta"] for l in rep_layers]
    fr_deltas = [rep[l]["replacements"].get("full_rank", {}).get("loss_delta", 0)
                 for l in rep_layers]
    r256_deltas = [rep[l]["replacements"].get("rank_256", {}).get("loss_delta", 0)
                   for l in rep_layers]
    r64_deltas = [rep[l]["replacements"].get("rank_64", {}).get("loss_delta", 0)
                  for l in rep_layers]

    ax = axes[0]
    ax.bar(x - 1.5*width, knockout_deltas, width, label="Knockout (skip)",
           color="#E53935", alpha=0.85, edgecolor="white")
    ax.bar(x - 0.5*width, fr_deltas, width, label="Linear (full rank)",
           color="#1976D2", alpha=0.85, edgecolor="white")
    ax.bar(x + 0.5*width, r256_deltas, width, label="Linear (rank 256)",
           color="#FFA726", alpha=0.85, edgecolor="white")
    ax.bar(x + 1.5*width, r64_deltas, width, label="Linear (rank 64)",
           color="#66BB6A", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in rep_layers], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss Delta vs Baseline")
    ax.set_title("Knockout vs Linear Replacement: Loss Impact")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax = axes[1]
    fr_rec = [rep[l]["replacements"].get("full_rank", {}).get(
                  "recovery_vs_knockout", 0) * 100 for l in rep_layers]
    r256_rec = [rep[l]["replacements"].get("rank_256", {}).get(
                    "recovery_vs_knockout", 0) * 100 for l in rep_layers]
    r64_rec = [rep[l]["replacements"].get("rank_64", {}).get(
                   "recovery_vs_knockout", 0) * 100 for l in rep_layers]

    ax.bar(x - width, fr_rec, width, label="Full rank",
           color="#1976D2", alpha=0.85, edgecolor="white")
    ax.bar(x, r256_rec, width, label="Rank 256",
           color="#FFA726", alpha=0.85, edgecolor="white")
    ax.bar(x + width, r64_rec, width, label="Rank 64",
           color="#66BB6A", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in rep_layers], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery vs Knockout (%)")
    ax.set_title("Layer Replacement: Recovery from Knockout Damage")
    ax.axhline(y=100, color="black", linestyle="--", alpha=0.3, label="100% = full recovery")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)

    fig.suptitle(f"Global Linear Replacement (Method 7) — {model_short}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "layer_replacement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved layer_replacement.png")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Load data
    calibration_data = load_calibration_data()
    t2_criticality = load_t2_criticality()

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    t_model = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    num_layers = len(model.model.layers)
    print(f"Model loaded in {time.time() - t_model:.1f}s — {num_layers} layers")

    # Run analysis
    print("\n--- Computing Linearization Gaps ---")
    t_analyze = time.time()
    summary, num_layers = analyze_model(model, tokenizer, calibration_data)
    analyze_time = time.time() - t_analyze
    print(f"Analysis took {analyze_time:.1f}s")

    # Print summary table
    print("\n" + "=" * 110)
    print(f"{'Layer':>5} | {'Perturb':>9} | {'Attn':>9} | {'MLP':>9} | "
          f"{'NL Order':>8} | {'R²':>6} | {'SpecNorm':>9} | {'Amp':>7} | {'||g||/||x||':>11}")
    print("-" * 120)
    for l in sorted(summary.keys()):
        s = summary[l]
        print(f"{l:>5} | {s['perturb_gap_mean']:>9.4f} | {s['attn_gap_mean']:>9.4f} | "
              f"{s['mlp_gap_mean']:>9.4f} | {s['nonlinearity_order_mean']:>8.2f} | "
              f"{s['log_log_r2_mean']:>6.3f} | {s['spectral_norm_mean']:>9.4f} | "
              f"{s['mean_amplification']:>7.4f} | {s['transform_to_input_ratio']:>11.6f}")
    print("=" * 120)

    # Part 6: Jacobian consistency
    print("\n--- Computing Jacobian Consistency ---")
    t_jc = time.time()
    jacobian_consistency = compute_jacobian_consistency(
        model, tokenizer, calibration_data, num_layers)
    jc_time = time.time() - t_jc
    print(f"Jacobian consistency took {jc_time:.1f}s")

    # Print consistency summary
    print(f"\n{'Layer':>5} | {'Consistency':>12} | {'Std':>8}")
    print("-" * 35)
    for l in sorted(jacobian_consistency.keys()):
        jc = jacobian_consistency[l]
        print(f"{l:>5} | {jc['consistency_mean']:>12.4f} | {jc.get('consistency_std', 0):>8.4f}")

    # Method 7: Global Linear Replacement
    replacement_results = None
    if t2_criticality:
        print("\n--- Running Global Linear Replacement (Method 7) ---")
        t_repl = time.time()
        # Build single_results format for run_layer_replacements
        single_results = {"knockouts": {
            int(k): {"loss_delta": v} for k, v in t2_criticality.items()
        }}
        replacement_results = run_layer_replacements(
            model, tokenizer, num_layers, single_results, calibration_data)
        repl_time = time.time() - t_repl
        print(f"Layer replacement took {repl_time:.1f}s")
    else:
        print("\nSkipping Method 7 (no T-2 criticality data available)")
        repl_time = 0.0

    # Free GPU before plotting
    del model
    torch.cuda.empty_cache()

    # Generate plots
    print("\n--- Generating Plots ---")
    create_plots(summary, num_layers, t2_criticality, jacobian_consistency)
    create_replacement_plot(replacement_results)

    # Save results
    json_results = {}
    for l, s in summary.items():
        s_copy = {k: v for k, v in s.items() if not k.startswith("per_prompt_")}
        # Convert float keys in multiscale_gaps to strings for JSON
        if "multiscale_gaps" in s_copy:
            s_copy["multiscale_gaps"] = {str(k): v for k, v in s_copy["multiscale_gaps"].items()}
        json_results[f"layer_{l}"] = s_copy

    # Add Jacobian consistency to per-layer results
    for l, jc in jacobian_consistency.items():
        key = f"layer_{l}"
        if key in json_results:
            json_results[key]["jacobian_consistency_mean"] = jc["consistency_mean"]
            json_results[key]["jacobian_consistency_std"] = jc.get("consistency_std", 0)

    output = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": DEVICE,
            "num_layers": num_layers,
            "num_prompts": len(calibration_data),
            "max_seq_len": MAX_SEQ_LEN,
            "num_perturbation_dirs": NUM_PERTURBATION_DIRS,
            "perturbation_eps": PERTURBATION_EPS,
            "multi_scale_eps": MULTI_SCALE_EPS,
            "multi_scale_dirs": MULTI_SCALE_DIRS,
            "jacobian_consistency_prompts": JACOBIAN_CONSISTENCY_PROMPTS,
            "jacobian_consistency_dirs": JACOBIAN_CONSISTENCY_DIRS,
        },
        "per_layer": json_results,
        "cross_reference": {
            "t2_criticality": {str(k): v for k, v in t2_criticality.items()} if t2_criticality else None,
        },
        "layer_replacements": replacement_results,
        "timing": {
            "total_s": time.time() - t_start,
            "analysis_s": analyze_time,
            "jacobian_consistency_s": jc_time,
            "replacement_s": repl_time,
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR / 'summary.json'}")
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
