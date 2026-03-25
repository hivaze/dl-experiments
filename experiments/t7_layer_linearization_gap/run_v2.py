"""
T-7 Layer Linearization Gap — V2 Redesign
==========================================

Addresses logical issues in the original experiment:
1. Layer 0 paradox: Methods 1-5 show highest nonlinearity, Method 7 shows
   highest linearizability. NEW Method 8 (PCA-aligned perturbation gap) tests
   whether nonlinearity is on-manifold or off-manifold.
2. Metric incoherence: Methods 1-5 analyze g(x), original Method 7 fits full
   output Y=WX. REDESIGNED Method 7 fits the residual g(x)=Y-X directly and
   reports Residual R² as the primary metric.
3. Ridge lambda maxes out: Extended grid to [0.001..1000].
4. Sub-quadratic orders: Restricted eps range to [0.005..0.1], flag R²<0.5.
5. Insufficient data: Uses extended calibration (200+ prompts, 512 max tokens).
6. Homogeneity gap removed (uninformative due to RMSNorm).
7. Enhanced Jacobian consistency: 30 prompts, 16 dirs, all token positions.

Methods in this script:
  1. Perturbation gap (unchanged algorithm, more data)
  3. Attn vs MLP decomposition (unchanged)
  4. Jacobian spectral properties (unchanged)
  5. Multi-scale analysis (FIXED: restricted eps, R² flagging)
  6. Jacobian consistency (ENHANCED: more prompts, all tokens)
  7. Global linear replacement (REDESIGNED: residual fitting + Residual R²)
  8. Data-aligned perturbation gap (NEW: PCA on-manifold vs off-manifold)

Usage:
  poetry run python experiments/t7_layer_linearization_gap/run_v2.py
"""

import json
import time
import random
import sys
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

# Extended calibration data (falls back to original if not available)
CALIBRATION_PATH = (
    Path(__file__).parents[2]
    / "data" / "text_completions" / "qwen3-4b-instruct-2507" / "completions_extended.json"
)
CALIBRATION_PATH_FALLBACK = (
    Path(__file__).parents[2]
    / "data" / "text_completions" / "qwen3-4b-instruct-2507" / "completions.json"
)
T2_RESULTS_PATH = (
    Path(__file__).parents[2]
    / "experiments" / "t2_layer_knockout" / "results" / "results.json"
)

# Methods 1, 3, 4 — use subset of prompts (gap is stable across prompts)
NUM_PERTURBATION_DIRS = 16
PERTURBATION_EPS = 0.05
MAX_SEQ_LEN = 256  # up from 128
ANALYSIS_PROMPTS = 50  # prompts for Methods 1,3,4,5,8 (gap is stable with 50)

# Method 5: Multi-scale (FIXED — drops eps=0.2, adds eps=0.005)
MULTI_SCALE_EPS = [0.005, 0.01, 0.02, 0.05, 0.1]
MULTI_SCALE_DIRS = 8

# Method 6: Jacobian consistency (ENHANCED)
JACOBIAN_CONSISTENCY_PROMPTS = 30   # up from 10
JACOBIAN_CONSISTENCY_DIRS = 16      # up from 8

# Method 7: Residual fitting (REDESIGNED)
RIDGE_LAMBDAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RANKS_TO_TEST = [16, 32, 64, 128, 256, 512]
TRAIN_FRACTION = 0.8

# Method 8: Data-aligned perturbation gap (NEW)
PCA_TOP_K = 20        # top PCA directions to test
PCA_RANDOM_K = 20     # random directions for comparison
PCA_EPS = 0.05


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_calibration_data():
    """Load calibration completions, preferring extended set."""
    path = CALIBRATION_PATH if CALIBRATION_PATH.exists() else CALIBRATION_PATH_FALLBACK
    with open(path) as f:
        data = json.load(f)
    completions = data["completions"]
    print(f"Loaded {len(completions)} completions from {path.name}")
    return completions


def load_t2_criticality():
    """Load T-2 layer knockout results for cross-referencing."""
    if not T2_RESULTS_PATH.exists():
        print("T-2 results not found, skipping cross-reference")
        return None
    with open(T2_RESULTS_PATH) as f:
        data = json.load(f)
    knockouts = data["single_knockouts"]["knockouts"]
    return {int(k): v["loss_delta"] for k, v in knockouts.items()}


# ============================================================================
# Layer Function Wrappers
# ============================================================================

def make_layer_fn(layer, position_embeddings):
    """g(h) = layer(h) - h (non-residual part)."""
    def fn(hidden_states):
        out = layer(
            hidden_states,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        return out - hidden_states
    return fn


def make_attn_fn(layer, position_embeddings):
    """Attention sublayer contribution only."""
    def fn(hidden_states):
        normed = layer.input_layernorm(hidden_states)
        attn_out, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=None,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        return attn_out
    return fn


def make_mlp_fn(layer, position_embeddings):
    """MLP sublayer contribution (includes attention dependency)."""
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
        return mlp_out
    return fn


# ============================================================================
# Method 1: Perturbation Gap
# ============================================================================

def compute_perturbation_gap(fn, hidden_states, num_dirs, eps):
    """Perturbation gap: ||actual - linear_pred|| / ||actual||.
    Uses central differences for Jacobian-vector products."""
    with torch.no_grad():
        g_x = fn(hidden_states).float()

    total_gap = torch.zeros(hidden_states.shape[1], device=hidden_states.device)
    input_scale = torch.norm(hidden_states, dim=-1, keepdim=True) + 1e-10

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

    return total_gap / num_dirs


# ============================================================================
# Method 4: Jacobian Spectral Properties
# ============================================================================

def compute_jacobian_stats(fn, hidden_states, num_dirs=16, eps=0.05):
    """Spectral norm (power iteration) and mean amplification."""
    batch, seq, dim = hidden_states.shape
    device = hidden_states.device
    input_scale = torch.norm(hidden_states, dim=-1, keepdim=True) + 1e-10

    # Power iteration for spectral norm (5 iters)
    v = torch.randn(batch, seq, dim, device=device, dtype=hidden_states.dtype)
    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-10)

    for _ in range(5):
        delta = v * input_scale * eps
        with torch.no_grad():
            jv = (fn(hidden_states + delta).float() - fn(hidden_states - delta).float()) / 2.0
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
        amp = torch.norm(jd, dim=-1)
        amp_samples.append(amp)

    mean_amp = torch.stack(amp_samples).mean(0)
    return spectral_norm_est.squeeze(0).detach(), mean_amp.squeeze(0).detach()


# ============================================================================
# Method 5: Multi-Scale Analysis (FIXED)
# ============================================================================

def compute_multiscale_gap(fn, hidden_states, eps_values, num_dirs):
    """Gap at multiple eps, with R² flagging. Restricted to eps<=0.1."""
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

    log_eps = np.log(list(gaps_by_eps.keys()))
    log_gap = np.log(np.clip(list(gaps_by_eps.values()), 1e-10, None))
    if len(log_eps) >= 2:
        slope, intercept = np.polyfit(log_eps, log_gap, 1)
        r2 = float(np.corrcoef(log_eps, log_gap)[0, 1] ** 2)
    else:
        slope, r2 = 0.0, 0.0

    return {
        "gaps_by_eps": gaps_by_eps,
        "nonlinearity_order": float(slope + 1),
        "log_log_slope": float(slope),
        "log_log_r2": r2,
        "reliable": r2 >= 0.5,
    }


# ============================================================================
# Method 8: Data-Aligned Perturbation Gap (NEW)
# ============================================================================

def compute_pca_aligned_gap(fn, hidden_states, top_k, random_k, eps):
    """Compare perturbation gap along PCA (on-manifold) vs random (off-manifold) directions.

    If on-manifold gap << off-manifold gap, the nonlinearity is concentrated
    in directions the data doesn't actually visit, explaining why global linear
    fits (Method 7) can work despite high random-direction gap (Method 1).
    """
    batch, seq, dim = hidden_states.shape
    device = hidden_states.device

    # Flatten to (n_tokens, dim) for PCA
    tokens = hidden_states.reshape(-1, dim).float()
    n_tokens = tokens.shape[0]

    # Center and compute covariance
    mean = tokens.mean(0, keepdim=True)
    centered = tokens - mean
    # Use SVD on centered data for numerical stability (avoids forming d×d cov)
    # centered is (n, d); SVD gives U(n,k) S(k) V(d,k)
    k_compute = min(top_k, n_tokens - 1, dim)
    if k_compute < 1:
        return {
            "on_manifold_gap_mean": float("nan"),
            "off_manifold_gap_mean": float("nan"),
            "gap_ratio": float("nan"),
            "effective_rank": 0,
            "top_eigenvalues": [],
        }

    # Low-rank SVD for efficiency
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S[:k_compute] ** 2) / (n_tokens - 1)
    pca_dirs = Vh[:k_compute]  # (k, dim) — top PCA directions

    # Effective rank: eigenvalues needed for 95% variance
    cumvar = torch.cumsum(eigenvalues, 0) / eigenvalues.sum()
    effective_rank = int((cumvar < 0.95).sum().item()) + 1

    # Compute gap along PCA directions (on-manifold)
    input_scale = torch.norm(hidden_states, dim=-1, keepdim=True) + 1e-10
    with torch.no_grad():
        g_x = fn(hidden_states).float()

    on_manifold_gaps = []
    n_pca = min(top_k, k_compute)
    for i in range(n_pca):
        d = pca_dirs[i].unsqueeze(0).unsqueeze(0).to(hidden_states.dtype)  # (1, 1, dim)
        d = d / (d.norm() + 1e-10)
        d = d.expand_as(hidden_states) * input_scale * eps

        with torch.no_grad():
            g_plus = fn(hidden_states + d).float()
            g_minus = fn(hidden_states - d).float()

        jd_est = (g_plus - g_minus) / 2.0
        actual_delta = g_plus - g_x
        err = torch.norm(actual_delta - jd_est, dim=-1)
        scale = torch.norm(actual_delta, dim=-1) + 1e-10
        gap = (err / scale).mean().item()
        on_manifold_gaps.append(gap)

    # Compute gap along random directions (off-manifold)
    off_manifold_gaps = []
    for _ in range(random_k):
        d = torch.randn_like(hidden_states)
        d = d / (torch.norm(d, dim=-1, keepdim=True) + 1e-10)
        d = d * input_scale * eps

        with torch.no_grad():
            g_plus = fn(hidden_states + d).float()
            g_minus = fn(hidden_states - d).float()

        jd_est = (g_plus - g_minus) / 2.0
        actual_delta = g_plus - g_x
        err = torch.norm(actual_delta - jd_est, dim=-1)
        scale = torch.norm(actual_delta, dim=-1) + 1e-10
        gap = (err / scale).mean().item()
        off_manifold_gaps.append(gap)

    on_mean = float(np.mean(on_manifold_gaps))
    off_mean = float(np.mean(off_manifold_gaps))
    ratio = on_mean / (off_mean + 1e-10)

    return {
        "on_manifold_gap_mean": on_mean,
        "on_manifold_gap_std": float(np.std(on_manifold_gaps)),
        "off_manifold_gap_mean": off_mean,
        "off_manifold_gap_std": float(np.std(off_manifold_gaps)),
        "gap_ratio": ratio,
        "effective_rank": effective_rank,
        "top_eigenvalues": eigenvalues[:min(20, len(eigenvalues))].tolist(),
        "n_pca_dirs": n_pca,
        "n_random_dirs": random_k,
    }


# ============================================================================
# Method 6: Enhanced Jacobian Consistency
# ============================================================================

def compute_jacobian_consistency_v2(model, tokenizer, calibration_data, num_layers):
    """Enhanced Jacobian consistency: 30 prompts, 16 dirs, all token positions."""
    hidden_dim = model.config.hidden_size
    device = DEVICE

    # Shared random directions
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + 7777)
    shared_dirs = []
    for _ in range(JACOBIAN_CONSISTENCY_DIRS):
        d = torch.randn(1, 1, hidden_dim, device=device, dtype=torch.bfloat16,
                         generator=gen)
        d = d / (d.norm() + 1e-10)
        shared_dirs.append(d)

    prompts_subset = calibration_data[:JACOBIAN_CONSISTENCY_PROMPTS]

    # layer_jvps[layer_idx] = list of (num_dirs, hidden_dim) tensors
    # Each tensor is the mean-over-tokens JVP for one prompt
    layer_jvps = {i: [] for i in range(num_layers)}

    for entry in tqdm(prompts_subset, desc="Jacobian consistency v2"):
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

            # Compute JVPs at ALL token positions, then average
            prompt_jvps = []
            for d_unit in shared_dirs:
                d_scaled = d_unit.expand_as(h_comp) * input_scale * PERTURBATION_EPS
                with torch.no_grad():
                    g_plus = layer_fn(h_comp + d_scaled).float()
                    g_minus = layer_fn(h_comp - d_scaled).float()
                jvp = (g_plus - g_minus) / 2.0  # (1, n_comp, dim)
                # Average JVP magnitude across token positions, then normalize
                jvp_mean = jvp.squeeze(0).mean(dim=0)  # (dim,)
                jvp_mean = jvp_mean / (jvp_mean.norm() + 1e-10)
                prompt_jvps.append(jvp_mean.cpu())

            layer_jvps[layer_idx].append(torch.stack(prompt_jvps))

        del hidden_states_by_layer
        torch.cuda.empty_cache()

    # Compute consistency: pairwise cosine similarity
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
# Method 7: Redesigned Global Linear Replacement
# ============================================================================

def compute_loss(model, tokenizer, calibration_data, device):
    """Mean cross-entropy loss on completion tokens."""
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
    """Collect (input, output) activation pairs for a given layer."""
    inputs_list = []
    outputs_list = []

    def capture_hook(module, inp, out):
        h_in = inp[0].detach()
        h_out = out[0].detach() if isinstance(out, tuple) else out.detach()
        inputs_list.append(h_in.squeeze(0))
        outputs_list.append(h_out.squeeze(0))

    hook = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        for entry in calibration_data:
            tokens = tokenizer(entry["full_text"], return_tensors="pt").to(device)
            model(**tokens, use_cache=False)
    hook.remove()

    return torch.cat(inputs_list, dim=0), torch.cat(outputs_list, dim=0)


def split_train_test(inputs, outputs, train_frac=0.8, seed=42):
    """Split activation pairs into train/test."""
    n = inputs.shape[0]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(n, generator=gen).to(inputs.device)
    n_train = int(n * train_frac)
    return (inputs[perm[:n_train]], outputs[perm[:n_train]],
            inputs[perm[n_train:]], outputs[perm[n_train:]])


def fit_ridge(X, Y, ridge_lambda=0.0):
    """Fit W = argmin ||Y - WX||² + λ||W||² via normal equations. float32."""
    d = X.shape[1]
    XtX = X.T @ X
    if ridge_lambda > 0:
        XtX = XtX + ridge_lambda * torch.eye(d, device=X.device)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY).T  # (d, d)
    return W


def activation_mse(W, X, Y, bias=None):
    """MSE: ||Y - WX - b||² / N."""
    pred = X @ W.T
    if bias is not None:
        pred = pred + bias.unsqueeze(0)
    return ((Y - pred) ** 2).mean().item()


def compute_r2(W, X, Y, bias=None):
    """R² = 1 - ||Y - WX||² / ||Y - mean(Y)||²."""
    pred = X @ W.T
    if bias is not None:
        pred = pred + bias.unsqueeze(0)
    ss_res = ((Y - pred) ** 2).sum().item()
    ss_tot = ((Y - Y.mean(0, keepdim=True)) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-10)


def evaluate_with_replacement(model, tokenizer, layer_idx, W, bias,
                               calibration_data, device):
    """Replace a layer with linear/affine map and measure loss."""
    def hook_fn(module, inp, out):
        h_in = inp[0]
        h_out = torch.einsum("ij,bsj->bsi", W, h_in)
        if bias is not None:
            h_out = h_out + bias
        return (h_out,) + out[1:] if isinstance(out, tuple) else h_out

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    loss = compute_loss(model, tokenizer, calibration_data, device)
    hook.remove()
    return loss


def identity_replacement_loss(model, tokenizer, layer_idx, calibration_data, device):
    """Replace layer with identity (skip connection only, no correction)."""
    def hook_fn(module, inp, out):
        h_in = inp[0]
        return (h_in,) + out[1:] if isinstance(out, tuple) else h_in

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    loss = compute_loss(model, tokenizer, calibration_data, device)
    hook.remove()
    return loss


def run_method7_v2(model, tokenizer, num_layers, calibration_data):
    """Redesigned Method 7: residual fitting + full-output fitting + identity baseline."""
    print("\n" + "=" * 70)
    print("METHOD 7 v2: Residual + Full-Output Linear Replacement")
    print("=" * 70)

    device = next(model.parameters()).device

    # Baseline loss
    print("  Computing baseline loss...")
    baseline_loss = compute_loss(model, tokenizer, calibration_data, device)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Knockout losses for all layers
    print(f"  Computing knockout losses for all {num_layers} layers...")
    original_layers = list(model.model.layers)
    knockout_losses = {}
    for skip_idx in tqdm(range(num_layers), desc="  Knockout"):
        remaining = [l for i, l in enumerate(original_layers) if i != skip_idx]
        model.model.layers = torch.nn.ModuleList(remaining)
        knockout_losses[skip_idx] = compute_loss(model, tokenizer, calibration_data, device)
    model.model.layers = torch.nn.ModuleList(original_layers)

    # Identity replacement losses for all layers
    print(f"  Computing identity replacement losses...")
    identity_losses = {}
    for li in tqdm(range(num_layers), desc="  Identity"):
        identity_losses[li] = identity_replacement_loss(
            model, tokenizer, li, calibration_data, device)

    results = {
        "baseline_loss": baseline_loss,
        "config": {
            "ranks": RANKS_TO_TEST,
            "ridge_lambdas": RIDGE_LAMBDAS,
            "train_fraction": TRAIN_FRACTION,
        },
        "replacements": {},
    }

    for layer_idx in tqdm(range(num_layers), desc="Layer replacements"):
        t0 = time.time()

        inputs, outputs = collect_layer_activations(
            model, tokenizer, layer_idx, calibration_data, device)
        n_tokens = inputs.shape[0]

        # Train/test split in float32
        X_tr, Y_tr, X_te, Y_te = split_train_test(
            inputs.float(), outputs.float(), TRAIN_FRACTION, SEED)

        # Compute residuals R = Y - X = g(x)
        R_tr = Y_tr - X_tr
        R_te = Y_te - X_te

        knockout_delta = knockout_losses[layer_idx] - baseline_loss
        identity_delta = identity_losses[layer_idx] - baseline_loss

        layer_results = {
            "knockout_loss": knockout_losses[layer_idx],
            "knockout_delta": knockout_delta,
            "identity_loss": identity_losses[layer_idx],
            "identity_delta": identity_delta,
            "n_tokens": n_tokens,
        }
        replacements = {}

        # ============================================================
        # A. RESIDUAL FITTING: R = g(x) ≈ W_r · X
        # ============================================================
        best_lam_r, best_mse_r = 0.0, float("inf")
        for lam in RIDGE_LAMBDAS:
            W_r = fit_ridge(X_tr, R_tr, lam)
            mse = activation_mse(W_r, X_te, R_te)
            if mse < best_mse_r:
                best_lam_r, best_mse_r = lam, mse
                W_r_best = W_r

        # Residual R² on test set
        residual_r2 = compute_r2(W_r_best, X_te, R_te)

        # Build full replacement: W_full = I + W_r
        d = X_tr.shape[1]
        W_full_from_residual = torch.eye(d, device=X_tr.device) + W_r_best
        W_dev = W_full_from_residual.to(torch.bfloat16).to(device)
        res_loss = evaluate_with_replacement(
            model, tokenizer, layer_idx, W_dev, None, calibration_data, device)
        res_delta = res_loss - baseline_loss
        res_rec = (1.0 - res_delta / knockout_delta) if knockout_delta > 0 else 0.0

        replacements["residual_ridge"] = {
            "loss": res_loss, "loss_delta": res_delta,
            "recovery_vs_knockout": res_rec,
            "residual_r2": residual_r2,
            "best_lambda": best_lam_r, "test_mse": best_mse_r,
        }
        del W_dev

        # ============================================================
        # B. FULL-OUTPUT FITTING: Y ≈ W · X (existing approach)
        # ============================================================
        best_lam_f, best_mse_f = 0.0, float("inf")
        for lam in RIDGE_LAMBDAS:
            W = fit_ridge(X_tr, Y_tr, lam)
            mse = activation_mse(W, X_te, Y_te)
            if mse < best_mse_f:
                best_lam_f, best_mse_f = lam, mse
                W_f_best = W

        full_r2 = compute_r2(W_f_best, X_te, Y_te)

        W_dev = W_f_best.to(torch.bfloat16).to(device)
        full_loss = evaluate_with_replacement(
            model, tokenizer, layer_idx, W_dev, None, calibration_data, device)
        full_delta = full_loss - baseline_loss
        full_rec = (1.0 - full_delta / knockout_delta) if knockout_delta > 0 else 0.0

        replacements["full_output_ridge"] = {
            "loss": full_loss, "loss_delta": full_delta,
            "recovery_vs_knockout": full_rec,
            "full_output_r2": full_r2,
            "best_lambda": best_lam_f, "test_mse": best_mse_f,
        }
        del W_dev

        # ============================================================
        # C. LOW-RANK RESIDUAL-PRESERVING FITS
        # ============================================================
        for rank in RANKS_TO_TEST:
            W_r_lr = fit_ridge(X_tr, R_tr, ridge_lambda=0.1)
            U, S, Vh = torch.linalg.svd(W_r_lr, full_matrices=False)
            W_r_trunc = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            W_lr = torch.eye(d, device=X_tr.device) + W_r_trunc
            lr_test_mse = activation_mse(W_lr, X_te, Y_te)
            lr_residual_r2 = compute_r2(W_r_trunc, X_te, R_te)

            W_dev = W_lr.to(torch.bfloat16).to(device)
            lr_loss = evaluate_with_replacement(
                model, tokenizer, layer_idx, W_dev, None, calibration_data, device)
            lr_delta = lr_loss - baseline_loss
            lr_rec = (1.0 - lr_delta / knockout_delta) if knockout_delta > 0 else 0.0

            replacements[f"rank_{rank}"] = {
                "loss": lr_loss, "loss_delta": lr_delta,
                "recovery_vs_knockout": lr_rec,
                "residual_r2": lr_residual_r2,
                "test_mse": lr_test_mse, "rank": rank,
            }
            del W_dev

        layer_results["replacements"] = replacements
        results["replacements"][layer_idx] = layer_results

        elapsed = time.time() - t0
        rr = replacements["residual_ridge"]
        fr = replacements["full_output_ridge"]
        print(f"  L{layer_idx:>2} ({elapsed:>4.1f}s) "
              f"KO={knockout_delta:+.3f} Id={identity_delta:+.3f}  "
              f"ResR²={rr['residual_r2']:>.3f} ResRec={rr['recovery_vs_knockout']:>6.1%}  "
              f"FullRec={fr['recovery_vs_knockout']:>6.1%}")
        sys.stdout.flush()

        del inputs, outputs, X_tr, Y_tr, X_te, Y_te, R_tr, R_te
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Main Analysis (Methods 1, 3, 4, 5, 8)
# ============================================================================

def analyze_model_v2(model, tokenizer, calibration_data):
    """Run per-prompt analysis: Methods 1, 3, 4, 5, 8.
    Uses ANALYSIS_PROMPTS subset — gap is very stable across prompts (10-20% std).
    The full dataset is used only for Methods 6/7 which benefit from more data."""
    num_layers = len(model.model.layers)
    calibration_data = calibration_data[:ANALYSIS_PROMPTS]
    print(f"  Using {len(calibration_data)} prompts for per-prompt analysis")

    all_results = {i: {
        "perturb_gaps": [],
        "attn_gaps": [],
        "mlp_gaps": [],
        "spectral_norms": [],
        "mean_amps": [],
        "transform_norms": [],
        "input_norms": [],
        "multiscale": [],
        "num_tokens": [],
    } for i in range(num_layers)}

    # Method 8 needs all hidden states pooled, so collect per-layer
    pca_hidden_states = {i: [] for i in range(num_layers)}

    for entry_idx, entry in enumerate(tqdm(calibration_data, desc="Methods 1,3,4,5")):
        full_text = entry["full_text"]
        prompt_len = entry["prompt_token_count"]

        tokens = tokenizer(full_text, return_tensors="pt", truncation=True,
                           max_length=MAX_SEQ_LEN).to(DEVICE)
        seq_len = tokens["input_ids"].shape[1]

        # Forward pass to collect all hidden states via hooks
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

        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)

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
            attn_fn = make_attn_fn(layer, pos_emb)
            mlp_fn = make_mlp_fn(layer, pos_emb)

            # Method 1: Perturbation gap
            perturb_gap = compute_perturbation_gap(
                layer_fn, h_comp, NUM_PERTURBATION_DIRS, PERTURBATION_EPS)

            # Method 3: Attn vs MLP decomposition
            attn_gap = compute_perturbation_gap(
                attn_fn, h_comp, NUM_PERTURBATION_DIRS, PERTURBATION_EPS)
            mlp_gap = compute_perturbation_gap(
                mlp_fn, h_comp, NUM_PERTURBATION_DIRS, PERTURBATION_EPS)

            # Method 4: Jacobian spectral properties
            spec_norm, mean_amp = compute_jacobian_stats(layer_fn, h_comp)

            # Transform and input norms
            with torch.no_grad():
                g_x = layer_fn(h_comp)
            transform_norm = torch.norm(g_x, dim=-1).squeeze(0)
            input_norm = torch.norm(h_comp, dim=-1).squeeze(0)

            # Method 5: Multi-scale (fixed)
            ms = compute_multiscale_gap(
                layer_fn, h_comp, MULTI_SCALE_EPS, MULTI_SCALE_DIRS)

            # Store per-prompt means
            res = all_results[layer_idx]
            res["perturb_gaps"].append(float(perturb_gap.mean()))
            res["attn_gaps"].append(float(attn_gap.mean()))
            res["mlp_gaps"].append(float(mlp_gap.mean()))
            res["spectral_norms"].append(float(spec_norm.mean()))
            res["mean_amps"].append(float(mean_amp.mean()))
            res["transform_norms"].append(float(transform_norm.mean()))
            res["input_norms"].append(float(input_norm.mean()))
            res["multiscale"].append(ms)
            res["num_tokens"].append(n_comp)

            # Collect hidden states for Method 8 PCA
            pca_hidden_states[layer_idx].append(h_comp.detach())

        del hidden_states_by_layer
        torch.cuda.empty_cache()

    # Aggregate across prompts
    summary = {}
    for layer_idx in range(num_layers):
        res = all_results[layer_idx]
        if not res["perturb_gaps"]:
            continue
        weights = np.array(res["num_tokens"], dtype=np.float64)
        weights /= weights.sum()

        def wmean(vals):
            return float(np.average(vals, weights=weights))

        def wstd(vals):
            m = np.average(vals, weights=weights)
            return float(np.sqrt(np.average((np.array(vals) - m) ** 2, weights=weights)))

        ms_agg = {}
        for eps_val in MULTI_SCALE_EPS:
            eps_gaps = [m["gaps_by_eps"][eps_val] for m in res["multiscale"]]
            ms_agg[eps_val] = wmean(eps_gaps)
        ms_orders = [m["nonlinearity_order"] for m in res["multiscale"]]
        ms_r2s = [m["log_log_r2"] for m in res["multiscale"]]

        summary[layer_idx] = {
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
            "log_log_r2_mean": wmean(ms_r2s),
            "nonlinearity_order_reliable": wmean(ms_r2s) >= 0.5,
            "total_tokens": int(sum(res["num_tokens"])),
            "per_prompt_perturb_gaps": res["perturb_gaps"],
        }

    return summary, num_layers, pca_hidden_states


def run_method8(model, pca_hidden_states, num_layers):
    """Method 8: Data-aligned perturbation gap on pooled hidden states."""
    print("\n" + "=" * 70)
    print("METHOD 8: Data-Aligned Perturbation Gap (PCA)")
    print("=" * 70)

    results = {}
    for layer_idx in tqdm(range(num_layers), desc="Method 8 (PCA gap)"):
        hs_list = pca_hidden_states[layer_idx]
        if not hs_list:
            continue

        # Pool all hidden states for this layer: (1, total_tokens, dim)
        all_h = torch.cat(hs_list, dim=1)

        layer = model.model.layers[layer_idx]
        # Use a simple position embedding for the pooled tokens
        # (PCA gap doesn't depend on position for the gap measurement)
        seq_len = all_h.shape[1]
        position_ids = torch.arange(min(seq_len, MAX_SEQ_LEN),
                                     device=DEVICE).unsqueeze(0)
        # Truncate to MAX_SEQ_LEN for memory
        all_h_trunc = all_h[:, :MAX_SEQ_LEN, :]
        pos_emb = model.model.rotary_emb(all_h_trunc, position_ids=position_ids)
        layer_fn = make_layer_fn(layer, pos_emb)

        pca_result = compute_pca_aligned_gap(
            layer_fn, all_h_trunc, PCA_TOP_K, PCA_RANDOM_K, PCA_EPS)
        results[layer_idx] = pca_result

        ratio = pca_result["gap_ratio"]
        on = pca_result["on_manifold_gap_mean"]
        off = pca_result["off_manifold_gap_mean"]
        erank = pca_result["effective_rank"]
        print(f"  L{layer_idx:>2}: on={on:.4f} off={off:.4f} ratio={ratio:.3f} eff_rank={erank}")

        # Free memory
        del all_h, all_h_trunc
        torch.cuda.empty_cache()

    return results


# ============================================================================
# Plotting
# ============================================================================

def create_plots_v2(summary, num_layers, t2_criticality, consistency_results,
                    method7_results, method8_results):
    """Generate all v2 plots."""
    layers = sorted(summary.keys())

    # ---- Plot 1: Perturbation gap + attn/MLP decomposition ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    ax = axes[0]
    gaps = [summary[l]["perturb_gap_mean"] for l in layers]
    stds = [summary[l]["perturb_gap_std"] for l in layers]
    ax.errorbar(layers, gaps, yerr=stds, fmt='o-', markersize=4, capsize=2,
                color='#1976D2', label='Perturbation gap')
    ax.set_xlabel("Layer"); ax.set_ylabel("Gap (relative)")
    ax.set_title("Perturbation Gap Across Depth")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    attn_gaps = [summary[l]["attn_gap_mean"] for l in layers]
    mlp_gaps = [summary[l]["mlp_gap_mean"] for l in layers]
    ax.plot(layers, attn_gaps, 'o-', markersize=3, color='#E53935', label='Attention')
    ax.plot(layers, mlp_gaps, 's-', markersize=3, color='#43A047', label='MLP')
    ax.set_xlabel("Layer"); ax.set_ylabel("Gap (relative)")
    ax.set_title("Attention vs MLP Nonlinearity")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("V2: Linearization Gap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "linearization_gap_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Jacobian properties ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    spec_norms = [summary[l]["spectral_norm_mean"] for l in layers]
    mean_amps = [summary[l]["mean_amplification"] for l in layers]
    ratios = [summary[l]["transform_to_input_ratio"] for l in layers]

    ax = axes[0]
    ax.plot(layers, spec_norms, 'o-', markersize=4, color='#7B1FA2')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Spectral norm")
    ax.set_title("Spectral Norm"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(layers, mean_amps, 'o-', markersize=4, color='#00897B')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean amplification")
    ax.set_title("Mean Amplification"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(layers, ratios, 'o-', markersize=4, color='#FF6F00')
    ax.set_xlabel("Layer"); ax.set_ylabel("||g(x)|| / ||x||")
    ax.set_title("Transform-to-Input Ratio"); ax.grid(True, alpha=0.3)

    fig.suptitle("V2: Jacobian Properties", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "jacobian_properties_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Gap vs T-2 criticality ----
    if t2_criticality:
        fig, ax = plt.subplots(figsize=(8, 6))
        crit_layers = [l for l in layers if l in t2_criticality]
        g = [summary[l]["perturb_gap_mean"] for l in crit_layers]
        c = [t2_criticality[l] for l in crit_layers]
        ax.scatter(g, c, c=crit_layers, cmap='viridis', s=50, zorder=3)
        for l in crit_layers:
            ax.annotate(f"L{l}", (summary[l]["perturb_gap_mean"], t2_criticality[l]),
                       fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("Perturbation Gap"); ax.set_ylabel("T-2 Knockout Loss Delta")
        ax.set_title("Nonlinearity vs Layer Criticality")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "gap_vs_criticality_v2.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Plot 4: Gap heatmap ----
    n_prompts = len(summary[0]["per_prompt_perturb_gaps"])
    gap_matrix = np.array([summary[l]["per_prompt_perturb_gaps"] for l in layers])
    fig, ax = plt.subplots(figsize=(max(12, n_prompts * 0.12), 7))
    im = ax.imshow(gap_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel("Prompt index"); ax.set_ylabel("Layer")
    ax.set_yticks(range(0, len(layers), 5))
    ax.set_yticklabels([str(layers[i]) for i in range(0, len(layers), 5)])
    ax.set_title("Per-Prompt Perturbation Gap")
    plt.colorbar(im, ax=ax, label="Gap")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gap_heatmap_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 5: Multi-scale (fixed) ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    interesting = [0, 3, 8, 16, 24, 35]
    interesting = [l for l in interesting if l in summary]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(interesting)))
    for i, l in enumerate(interesting):
        eps_vals = list(summary[l]["multiscale_gaps"].keys())
        gap_vals = list(summary[l]["multiscale_gaps"].values())
        ax.plot(eps_vals, gap_vals, 'o-', color=cmap[i], label=f"L{l}", markersize=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("ε"); ax.set_ylabel("Gap")
    ax.set_title("Gap vs ε (log-log)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    orders = [summary[l]["nonlinearity_order_mean"] for l in layers]
    reliable = [summary[l].get("nonlinearity_order_reliable", True) for l in layers]
    colors = ['#1976D2' if r else '#BDBDBD' for r in reliable]
    markers = ['o' if r else 'x' for r in reliable]
    for i, l in enumerate(layers):
        ax.scatter(l, orders[i], c=colors[i], marker=markers[i], s=30, zorder=3)
    ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.3, label="Expected (quadratic)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Nonlinearity order k")
    ax.set_title("Fitted Nonlinearity Order (x = unreliable R²<0.5)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2]
    r2s = [summary[l]["log_log_r2_mean"] for l in layers]
    ax.bar(layers, r2s, color=['#1976D2' if r >= 0.5 else '#EF9A9A' for r in r2s])
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label="R² = 0.5 threshold")
    ax.set_xlabel("Layer"); ax.set_ylabel("R²")
    ax.set_title("Log-Log Fit Quality"); ax.legend(fontsize=7)

    fig.suptitle("V2: Multi-Scale Analysis (ε ≤ 0.1)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multiscale_analysis_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 6: Jacobian consistency ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cons_layers = [l for l in layers if l in consistency_results and
                   not np.isnan(consistency_results[l]["consistency_mean"])]
    cons_vals = [consistency_results[l]["consistency_mean"] for l in cons_layers]
    cons_stds = [consistency_results[l]["consistency_std"] for l in cons_layers]

    ax = axes[0]
    ax.errorbar(cons_layers, cons_vals, yerr=cons_stds, fmt='o-', markersize=4,
                capsize=2, color='#00897B')
    ax.set_xlabel("Layer"); ax.set_ylabel("Jacobian consistency")
    ax.set_title("Jacobian Consistency Across Depth (All Tokens)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    perturb = [summary[l]["perturb_gap_mean"] for l in cons_layers]
    ax.scatter(perturb, cons_vals, c=cons_layers, cmap='viridis', s=50, zorder=3)
    for l in cons_layers:
        if l in [0, 1, 6, 16, 35]:
            ax.annotate(f"L{l}", (summary[l]["perturb_gap_mean"],
                        consistency_results[l]["consistency_mean"]),
                       fontsize=7, textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Perturbation Gap (local nonlinearity)")
    ax.set_ylabel("Jacobian Consistency (global)")
    ax.set_title("Local Nonlinearity vs Global Consistency")
    ax.grid(True, alpha=0.3)

    fig.suptitle("V2: Enhanced Jacobian Consistency", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "jacobian_consistency_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 7: PCA-aligned gap (NEW) ----
    if method8_results:
        m8_layers = sorted(method8_results.keys())
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        on_gaps = [method8_results[l]["on_manifold_gap_mean"] for l in m8_layers]
        off_gaps = [method8_results[l]["off_manifold_gap_mean"] for l in m8_layers]
        ratios_m8 = [method8_results[l]["gap_ratio"] for l in m8_layers]
        eranks = [method8_results[l]["effective_rank"] for l in m8_layers]

        ax = axes[0]
        ax.plot(m8_layers, on_gaps, 'o-', markersize=4, color='#43A047',
                label='On-manifold (PCA)')
        ax.plot(m8_layers, off_gaps, 's-', markersize=4, color='#E53935',
                label='Off-manifold (random)')
        ax.set_xlabel("Layer"); ax.set_ylabel("Perturbation gap")
        ax.set_title("On- vs Off-Manifold Nonlinearity")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(m8_layers, ratios_m8, 'o-', markersize=4, color='#7B1FA2')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                   label="ratio=1 (isotropic)")
        ax.set_xlabel("Layer"); ax.set_ylabel("On/Off gap ratio")
        ax.set_title("Gap Ratio (< 1 = more linear on manifold)")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.bar(m8_layers, eranks, color='#1976D2', alpha=0.7)
        ax.set_xlabel("Layer"); ax.set_ylabel("Effective rank (95% var)")
        ax.set_title("Input Manifold Dimensionality")
        ax.grid(True, alpha=0.3)

        fig.suptitle("V2: Data-Aligned Perturbation Gap (Method 8)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "pca_aligned_gap.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Plot 8: Residual fitting ----
    if method7_results:
        rep = method7_results["replacements"]
        m7_layers = sorted(int(k) for k in rep.keys())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        res_r2 = [rep[l]["replacements"]["residual_ridge"]["residual_r2"] for l in m7_layers]
        full_rec = [rep[l]["replacements"]["full_output_ridge"]["recovery_vs_knockout"] * 100
                    for l in m7_layers]
        res_rec = [rep[l]["replacements"]["residual_ridge"]["recovery_vs_knockout"] * 100
                   for l in m7_layers]

        ax = axes[0]
        ax.plot(m7_layers, res_r2, 'o-', markersize=4, color='#1976D2',
                label='Residual R²')
        ax.set_xlabel("Layer"); ax.set_ylabel("R²")
        ax.set_title("Residual R² — Fraction of g(x) Explained by Linear Fit")
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(res_r2, full_rec, c=m7_layers, cmap='viridis', s=50, zorder=3,
                  label='Full-output recovery')
        ax.scatter(res_r2, res_rec, c=m7_layers, cmap='viridis', s=50, zorder=3,
                  marker='s', alpha=0.5, label='Residual recovery')
        for l in m7_layers:
            if l in [0, 6, 16, 35]:
                ax.annotate(f"L{l}", (rep[l]["replacements"]["residual_ridge"]["residual_r2"],
                            rep[l]["replacements"]["full_output_ridge"]["recovery_vs_knockout"] * 100),
                           fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("Residual R²"); ax.set_ylabel("CE Recovery (%)")
        ax.set_title("Residual R² vs CE Recovery")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        fig.suptitle("V2: Residual Fitting Analysis (Method 7)", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "residual_fitting_v2.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Plot 9: Layer replacement with identity baseline ----
    if method7_results:
        rep = method7_results["replacements"]
        m7_layers = sorted(int(k) for k in rep.keys())

        fig, axes = plt.subplots(2, 1, figsize=(18, 10))

        x = np.arange(len(m7_layers))
        width = 0.15

        ax = axes[0]
        methods_plot = [
            ("residual_ridge", "Residual Ridge", "#7B1FA2"),
            ("full_output_ridge", "Full-Output Ridge", "#1976D2"),
            ("rank_512", "R-512", "#00897B"),
            ("rank_256", "R-256", "#FFA726"),
            ("rank_64", "R-64", "#66BB6A"),
        ]
        for i, (key, label, color) in enumerate(methods_plot):
            recs = [rep[l]["replacements"].get(key, {}).get("recovery_vs_knockout", 0) * 100
                    for l in m7_layers]
            recs_clip = [max(r, -200) for r in recs]
            offset = (i - len(methods_plot)/2 + 0.5) * width
            ax.bar(x + offset, recs_clip, width, label=label, color=color, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in m7_layers], fontsize=7)
        ax.set_xlabel("Layer"); ax.set_ylabel("Recovery vs Knockout (%)")
        ax.set_title("Recovery Comparison — All Methods")
        ax.axhline(y=100, color="black", linestyle="--", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.legend(fontsize=7, ncol=5, loc="lower left")
        ax.set_ylim(bottom=-210)

        # Bottom: loss deltas including identity baseline
        ax = axes[1]
        ko_deltas = [rep[l]["knockout_delta"] for l in m7_layers]
        id_deltas = [rep[l]["identity_delta"] for l in m7_layers]
        res_deltas = [rep[l]["replacements"]["residual_ridge"]["loss_delta"] for l in m7_layers]
        full_deltas = [rep[l]["replacements"]["full_output_ridge"]["loss_delta"] for l in m7_layers]

        w2 = 0.2
        ax.bar(x - 1.5*w2, ko_deltas, w2, label="Knockout", color="#E53935", alpha=0.85)
        ax.bar(x - 0.5*w2, id_deltas, w2, label="Identity (skip only)", color="#9E9E9E", alpha=0.85)
        ax.bar(x + 0.5*w2, res_deltas, w2, label="Residual Ridge", color="#7B1FA2", alpha=0.85)
        ax.bar(x + 1.5*w2, full_deltas, w2, label="Full Ridge", color="#1976D2", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in m7_layers], fontsize=7)
        ax.set_xlabel("Layer"); ax.set_ylabel("Loss Delta vs Baseline")
        ax.set_title("Loss Impact: Knockout vs Identity vs Linear Replacement")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.legend(fontsize=8)

        fig.suptitle("V2: Global Linear Replacement (Method 7)", fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "layer_replacement_v2.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Plot 10: Paradox resolution ----
    if method8_results and method7_results:
        rep = method7_results["replacements"]
        m8_layers = sorted(method8_results.keys())
        m7_layers_set = set(int(k) for k in rep.keys())
        common = [l for l in m8_layers if l in m7_layers_set]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Panel A: Layer 0 deep-dive
        ax = axes[0]
        if 0 in method8_results and 0 in rep:
            l0_m8 = method8_results[0]
            l0_m7 = rep[0]["replacements"]
            metrics = {
                "Off-manifold\ngap": l0_m8["off_manifold_gap_mean"],
                "On-manifold\ngap": l0_m8["on_manifold_gap_mean"],
                "Gap ratio\n(on/off)": l0_m8["gap_ratio"],
                "Residual R²": l0_m7["residual_ridge"]["residual_r2"],
                "CE Recovery\n(%)": l0_m7["residual_ridge"]["recovery_vs_knockout"],
            }
            bars = ax.bar(range(len(metrics)), list(metrics.values()),
                         color=['#E53935', '#43A047', '#7B1FA2', '#1976D2', '#FF6F00'])
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(list(metrics.keys()), fontsize=8)
            ax.set_title("Layer 0 Deep-Dive: Resolving the Paradox")
            ax.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, metrics.values()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha='center', fontsize=8)

        # Panel B: On-manifold gap vs Residual R²
        ax = axes[1]
        on_gaps_common = [method8_results[l]["on_manifold_gap_mean"] for l in common]
        res_r2_common = [rep[l]["replacements"]["residual_ridge"]["residual_r2"] for l in common]
        ax.scatter(on_gaps_common, res_r2_common, c=common, cmap='viridis', s=50, zorder=3)
        for l in common:
            if l in [0, 1, 6, 8, 16, 35]:
                ax.annotate(f"L{l}",
                           (method8_results[l]["on_manifold_gap_mean"],
                            rep[l]["replacements"]["residual_ridge"]["residual_r2"]),
                           fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("On-Manifold Perturbation Gap")
        ax.set_ylabel("Residual R²")
        ax.set_title("On-Manifold Gap vs Residual R² (should anti-correlate)")
        ax.grid(True, alpha=0.3)

        fig.suptitle("V2: Layer 0 Paradox Resolution", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "paradox_resolution.png", dpi=150, bbox_inches="tight")
        plt.close()

    print("All v2 plots saved.")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # Load data
    calibration_data = load_calibration_data()
    t2_criticality = load_t2_criticality()

    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} decoder layers on {DEVICE}")

    # Phase 1: Methods 1, 3, 4, 5 + collect hidden states for Method 8
    print("\n" + "=" * 70)
    print("PHASE 1: Methods 1, 3, 4, 5 (per-prompt analysis)")
    print("=" * 70)
    t1 = time.time()
    summary, num_layers, pca_hidden_states = analyze_model_v2(
        model, tokenizer, calibration_data)
    print(f"Phase 1 done in {time.time() - t1:.1f}s")

    # Phase 2: Method 8 (PCA-aligned gap)
    t2 = time.time()
    method8_results = run_method8(model, pca_hidden_states, num_layers)
    print(f"Method 8 done in {time.time() - t2:.1f}s")

    # Free PCA hidden states
    del pca_hidden_states
    torch.cuda.empty_cache()

    # Phase 3: Method 6 (Enhanced Jacobian consistency)
    print("\n" + "=" * 70)
    print("PHASE 3: Method 6 (Enhanced Jacobian Consistency)")
    print("=" * 70)
    t3 = time.time()
    consistency_results = compute_jacobian_consistency_v2(
        model, tokenizer, calibration_data, num_layers)
    print(f"Method 6 done in {time.time() - t3:.1f}s")

    # Phase 4: Method 7 (Redesigned residual fitting)
    t4 = time.time()
    method7_results = run_method7_v2(model, tokenizer, num_layers, calibration_data)
    print(f"Method 7 done in {time.time() - t4:.1f}s")

    del model
    torch.cuda.empty_cache()

    # Phase 5: Plotting
    print("\nGenerating plots...")
    create_plots_v2(summary, num_layers, t2_criticality, consistency_results,
                    method7_results, method8_results)

    # Save results
    all_results = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": DEVICE,
            "num_layers": num_layers,
            "num_prompts_total": len(calibration_data),
            "num_prompts_analysis": ANALYSIS_PROMPTS,
            "max_seq_len": MAX_SEQ_LEN,
            "num_perturbation_dirs": NUM_PERTURBATION_DIRS,
            "perturbation_eps": PERTURBATION_EPS,
            "multi_scale_eps": MULTI_SCALE_EPS,
            "multi_scale_dirs": MULTI_SCALE_DIRS,
            "jacobian_consistency_prompts": JACOBIAN_CONSISTENCY_PROMPTS,
            "jacobian_consistency_dirs": JACOBIAN_CONSISTENCY_DIRS,
            "pca_top_k": PCA_TOP_K,
            "pca_random_k": PCA_RANDOM_K,
            "ridge_lambdas": RIDGE_LAMBDAS,
            "train_fraction": TRAIN_FRACTION,
        },
        "per_layer": {f"layer_{k}": v for k, v in summary.items()},
        "jacobian_consistency": {f"layer_{k}": v for k, v in consistency_results.items()},
        "pca_aligned_gap": {f"layer_{k}": v for k, v in method8_results.items()},
        "layer_replacements": method7_results,
    }

    summary_path = RESULTS_DIR / "summary_v2.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved {summary_path}")

    elapsed = time.time() - t_total
    print(f"\nTotal v2 runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Print key summary
    print("\n" + "=" * 70)
    print("KEY V2 RESULTS SUMMARY")
    print("=" * 70)
    for l in [0, 6, 8, 16, 35]:
        if l in summary and l in method8_results:
            s = summary[l]
            m8 = method8_results[l]
            rep = method7_results["replacements"].get(l, {})
            rr = rep.get("replacements", {}).get("residual_ridge", {})
            fr = rep.get("replacements", {}).get("full_output_ridge", {})
            print(f"  L{l:>2}: gap(random)={s['perturb_gap_mean']:.3f} "
                  f"gap(PCA)={m8['on_manifold_gap_mean']:.3f} "
                  f"ratio={m8['gap_ratio']:.3f} "
                  f"ResR²={rr.get('residual_r2', 0):.3f} "
                  f"CE_rec={fr.get('recovery_vs_knockout', 0):.1%}")


if __name__ == "__main__":
    main()
