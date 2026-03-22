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

Cross-references:
  - T-2 (layer knockout): correlate linearization gap with layer criticality
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
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
# Visualization
# ============================================================================

def create_plots(summary, num_layers, t2_criticality):
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

    # Free GPU before plotting
    del model
    torch.cuda.empty_cache()

    # Generate plots
    print("\n--- Generating Plots ---")
    create_plots(summary, num_layers, t2_criticality)

    # Save results
    json_results = {}
    for l, s in summary.items():
        s_copy = {k: v for k, v in s.items() if not k.startswith("per_prompt_")}
        # Convert float keys in multiscale_gaps to strings for JSON
        if "multiscale_gaps" in s_copy:
            s_copy["multiscale_gaps"] = {str(k): v for k, v in s_copy["multiscale_gaps"].items()}
        json_results[f"layer_{l}"] = s_copy

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
        },
        "per_layer": json_results,
        "cross_reference": {
            "t2_criticality": {str(k): v for k, v in t2_criticality.items()} if t2_criticality else None,
        },
        "timing": {
            "total_s": time.time() - t_start,
            "analysis_s": analyze_time,
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR / 'summary.json'}")
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
