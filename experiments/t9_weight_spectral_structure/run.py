"""
T-9: Weight Matrix Spectral Structure
======================================
Compute SVD of each weight matrix (Q, K, V, O, gate, up, down) per layer.
Analyze singular value distributions, effective rank, power-law fits,
and cross-reference with linearization gap (T-7) results.
"""

import json
import time
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda:0"
SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"

# Weight matrices to analyze per layer
WEIGHT_NAMES = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}

# Cross-reference paths
T7_RESULTS_PATH = Path(__file__).parents[1] / "t7_layer_linearization_gap" / "results"
T2_RESULTS_PATH = Path(__file__).parents[1] / "t2_layer_knockout" / "results" / "results.json"
T4_RESULTS_PATH = Path(__file__).parents[1] / "t4_residual_stream_geometry" / "results" / "summary.json"
SHUFFLE_RESULTS_PATH = Path(__file__).parents[1] / "layer_shuffle_recovery" / "results" / "results.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_spectral_stats(weight_matrix: torch.Tensor) -> dict:
    """Compute SVD-based spectral statistics for a weight matrix."""
    # Move to float32 for numerical stability in SVD
    W = weight_matrix.float()

    # Full SVD (we only need singular values for most stats)
    S = torch.linalg.svdvals(W)
    S_np = S.detach().cpu().numpy()

    # Basic stats
    rank = int((S > 1e-6).sum().item())
    total_energy = float((S ** 2).sum().item())

    # Effective rank via participation ratio
    # PR = (sum(s_i^2))^2 / sum(s_i^4)
    s2 = S_np ** 2
    s4 = S_np ** 4
    effective_rank = float(s2.sum() ** 2 / s4.sum()) if s4.sum() > 0 else 0.0

    # Normalized effective rank (0 to 1)
    max_possible_rank = min(W.shape)
    effective_rank_ratio = effective_rank / max_possible_rank

    # Cumulative energy: how many singular values to capture 50%, 90%, 99%
    cumulative = np.cumsum(s2) / s2.sum()
    rank_50 = int(np.searchsorted(cumulative, 0.50) + 1)
    rank_90 = int(np.searchsorted(cumulative, 0.90) + 1)
    rank_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    # Power-law fit on singular values (log-log linear regression)
    # Only fit on non-negligible singular values
    mask = S_np > 1e-6
    S_fit = S_np[mask]
    if len(S_fit) > 2:
        log_idx = np.log(np.arange(1, len(S_fit) + 1))
        log_sv = np.log(S_fit)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_idx, log_sv)
        power_law = {
            "exponent": float(-slope),  # positive exponent for decay
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
        }
    else:
        power_law = {"exponent": 0.0, "r_squared": 0.0, "p_value": 1.0}

    # Spectral entropy (normalized)
    s_norm = s2 / s2.sum()
    s_norm = s_norm[s_norm > 0]
    spectral_entropy = float(-np.sum(s_norm * np.log(s_norm)) / np.log(max_possible_rank))

    # Condition number (ratio of largest to smallest non-zero SV)
    condition_number = float(S_np[0] / S_np[rank - 1]) if rank > 0 and S_np[rank - 1] > 0 else float("inf")

    # Stable rank: ||W||_F^2 / ||W||_2^2 = sum(s^2) / s_max^2
    stable_rank = float(s2.sum() / s2[0]) if s2[0] > 0 else 0.0
    stable_rank_ratio = stable_rank / max_possible_rank

    # Top-10 singular values for inspection
    top_svs = S_np[:10].tolist()

    # Full SV spectrum (downsampled to 200 points for storage)
    if len(S_np) > 200:
        indices = np.linspace(0, len(S_np) - 1, 200, dtype=int)
        sv_spectrum_sampled = S_np[indices].tolist()
    else:
        sv_spectrum_sampled = S_np.tolist()

    return {
        "shape": list(W.shape),
        "actual_rank": rank,
        "max_rank": max_possible_rank,
        "effective_rank": round(effective_rank, 2),
        "effective_rank_ratio": round(effective_rank_ratio, 4),
        "stable_rank": round(stable_rank, 2),
        "stable_rank_ratio": round(stable_rank_ratio, 4),
        "rank_for_50pct": rank_50,
        "rank_for_90pct": rank_90,
        "rank_for_99pct": rank_99,
        "total_frobenius_sq": round(total_energy, 4),
        "spectral_entropy": round(spectral_entropy, 4),
        "condition_number": round(condition_number, 2),
        "power_law": power_law,
        "top_10_singular_values": [round(v, 6) for v in top_svs],
        "sv_spectrum_sampled": [round(v, 6) for v in sv_spectrum_sampled],
    }


def load_t7_results():
    """Load T-7 linearization gap results for cross-referencing."""
    candidates = list(T7_RESULTS_PATH.glob("*.json")) if T7_RESULTS_PATH.exists() else []
    for p in candidates:
        try:
            with open(p) as f:
                data = json.load(f)
            if "per_layer" in data or "layers" in data:
                return data
        except (json.JSONDecodeError, KeyError):
            continue
    return None


def load_t2_criticality():
    """Load T-2 layer knockout results — per-layer loss delta."""
    if not T2_RESULTS_PATH.exists():
        return None
    with open(T2_RESULTS_PATH) as f:
        data = json.load(f)
    knockouts = data.get("single_knockouts", {}).get("knockouts", {})
    return {int(k): v["loss_delta"] for k, v in knockouts.items()}


def load_t4_geometry():
    """Load T-4 residual stream geometry — per-layer isotropy/participation ratio."""
    if not T4_RESULTS_PATH.exists():
        return None
    with open(T4_RESULTS_PATH) as f:
        data = json.load(f)
    per_layer = data.get("per_layer", {})
    result = {}
    for k, v in per_layer.items():
        try:
            layer_idx = int(k.replace("layer_", ""))
            result[layer_idx] = {
                "participation_ratio": v.get("participation_ratio"),
                "isoscore": v.get("isoscore"),
                "spectral_flatness": v.get("spectral_flatness"),
            }
        except (ValueError, AttributeError):
            continue
    return result if result else None


def load_shuffle_weight_norms():
    """Load weight norms from layer shuffle experiment."""
    if not SHUFFLE_RESULTS_PATH.exists():
        return None
    with open(SHUFFLE_RESULTS_PATH) as f:
        data = json.load(f)
    return data.get("weight_norms")


def analyze_model(model, num_layers):
    """Compute spectral stats for every weight matrix in every layer."""
    per_layer = {}

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        layer_results = {}

        for name, attr_path in WEIGHT_NAMES.items():
            # Navigate to the weight tensor
            obj = layer
            for part in attr_path.split("."):
                obj = getattr(obj, part)

            t0 = time.time()
            stats_dict = compute_spectral_stats(obj)
            stats_dict["compute_time_s"] = round(time.time() - t0, 3)
            layer_results[name] = stats_dict

        per_layer[layer_idx] = layer_results
        print(f"  Layer {layer_idx:2d}: "
              f"eff_rank(Q)={layer_results['q_proj']['effective_rank']:.0f} "
              f"eff_rank(V)={layer_results['v_proj']['effective_rank']:.0f} "
              f"eff_rank(down)={layer_results['down_proj']['effective_rank']:.0f} "
              f"plaw_exp(Q)={layer_results['q_proj']['power_law']['exponent']:.3f}")

    return per_layer


def compute_aggregates(per_layer, num_layers):
    """Compute cross-layer aggregate statistics."""
    agg = {}

    for name in WEIGHT_NAMES:
        eff_ranks = [per_layer[i][name]["effective_rank"] for i in range(num_layers)]
        eff_ratios = [per_layer[i][name]["effective_rank_ratio"] for i in range(num_layers)]
        plaw_exps = [per_layer[i][name]["power_law"]["exponent"] for i in range(num_layers)]
        plaw_r2s = [per_layer[i][name]["power_law"]["r_squared"] for i in range(num_layers)]
        rank_90s = [per_layer[i][name]["rank_for_90pct"] for i in range(num_layers)]
        entropies = [per_layer[i][name]["spectral_entropy"] for i in range(num_layers)]

        agg[name] = {
            "effective_rank_mean": round(float(np.mean(eff_ranks)), 2),
            "effective_rank_std": round(float(np.std(eff_ranks)), 2),
            "effective_rank_ratio_mean": round(float(np.mean(eff_ratios)), 4),
            "power_law_exponent_mean": round(float(np.mean(plaw_exps)), 4),
            "power_law_exponent_std": round(float(np.std(plaw_exps)), 4),
            "power_law_r2_mean": round(float(np.mean(plaw_r2s)), 4),
            "rank_90pct_mean": round(float(np.mean(rank_90s)), 1),
            "spectral_entropy_mean": round(float(np.mean(entropies)), 4),
        }

    # Compare attention (Q/K) vs value (V/O) effective rank
    qk_ranks = []
    vo_ranks = []
    for i in range(num_layers):
        qk_ranks.append(np.mean([per_layer[i]["q_proj"]["effective_rank_ratio"],
                                  per_layer[i]["k_proj"]["effective_rank_ratio"]]))
        vo_ranks.append(np.mean([per_layer[i]["v_proj"]["effective_rank_ratio"],
                                  per_layer[i]["o_proj"]["effective_rank_ratio"]]))

    agg["qk_vs_vo_comparison"] = {
        "qk_mean_eff_rank_ratio": round(float(np.mean(qk_ranks)), 4),
        "vo_mean_eff_rank_ratio": round(float(np.mean(vo_ranks)), 4),
        "qk_higher_than_vo_layers": int(sum(1 for q, v in zip(qk_ranks, vo_ranks) if q > v)),
        "interpretation": "Q/K < V/O means routing is lower-rank than value extraction"
    }

    # MLP vs attention comparison
    attn_ranks = []
    mlp_ranks = []
    for i in range(num_layers):
        attn_ranks.append(np.mean([per_layer[i][n]["effective_rank_ratio"]
                                    for n in ["q_proj", "k_proj", "v_proj", "o_proj"]]))
        mlp_ranks.append(np.mean([per_layer[i][n]["effective_rank_ratio"]
                                    for n in ["gate_proj", "up_proj", "down_proj"]]))

    agg["attn_vs_mlp_comparison"] = {
        "attn_mean_eff_rank_ratio": round(float(np.mean(attn_ranks)), 4),
        "mlp_mean_eff_rank_ratio": round(float(np.mean(mlp_ranks)), 4),
    }

    # Plateau (layers 0-16) vs late (layers 17-35) comparison
    # From shuffle experiment: layers 0-16 have flat weight norms, 17+ diverge
    plateau_layers = list(range(0, 17))
    late_layers = list(range(17, num_layers))

    plateau_vs_late = {}
    for name in WEIGHT_NAMES:
        p_eff = [per_layer[i][name]["effective_rank_ratio"] for i in plateau_layers]
        l_eff = [per_layer[i][name]["effective_rank_ratio"] for i in late_layers]
        p_plaw = [per_layer[i][name]["power_law"]["exponent"] for i in plateau_layers]
        l_plaw = [per_layer[i][name]["power_law"]["exponent"] for i in late_layers]
        p_stable = [per_layer[i][name]["stable_rank_ratio"] for i in plateau_layers]
        l_stable = [per_layer[i][name]["stable_rank_ratio"] for i in late_layers]

        # Welch's t-test for significance
        t_stat, p_val = ttest_ind(p_eff, l_eff, equal_var=False)

        plateau_vs_late[name] = {
            "plateau_eff_rank_ratio": round(float(np.mean(p_eff)), 4),
            "late_eff_rank_ratio": round(float(np.mean(l_eff)), 4),
            "diff": round(float(np.mean(l_eff) - np.mean(p_eff)), 4),
            "t_statistic": round(float(t_stat), 3),
            "p_value": round(float(p_val), 6),
            "plateau_plaw_exp": round(float(np.mean(p_plaw)), 4),
            "late_plaw_exp": round(float(np.mean(l_plaw)), 4),
            "plateau_stable_rank_ratio": round(float(np.mean(p_stable)), 4),
            "late_stable_rank_ratio": round(float(np.mean(l_stable)), 4),
        }

    agg["plateau_vs_late"] = plateau_vs_late

    # Stable rank summary
    for name in WEIGHT_NAMES:
        stable_ranks = [per_layer[i][name]["stable_rank_ratio"] for i in range(num_layers)]
        agg[name]["stable_rank_ratio_mean"] = round(float(np.mean(stable_ranks)), 4)
        agg[name]["stable_rank_ratio_std"] = round(float(np.std(stable_ranks)), 4)

    return agg


def create_plots(per_layer, num_layers, t7_data=None):
    """Generate visualization plots."""
    layers = list(range(num_layers))

    # ── Plot 1: Effective Rank Ratio by Layer for All Matrices ────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = cm.tab10(np.linspace(0, 1, len(WEIGHT_NAMES)))

    for (name, _), color in zip(WEIGHT_NAMES.items(), colors):
        vals = [per_layer[i][name]["effective_rank_ratio"] for i in layers]
        ax.plot(layers, vals, marker="o", markersize=3, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank Ratio (eff_rank / max_rank)")
    ax.set_title("Effective Rank Ratio Across Layers — All Weight Matrices")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, num_layers - 0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "effective_rank_ratio_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Power-Law Exponent by Layer ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    for (name, _), color in zip(WEIGHT_NAMES.items(), colors):
        vals = [per_layer[i][name]["power_law"]["exponent"] for i in layers]
        ax.plot(layers, vals, marker="s", markersize=3, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Power-Law Exponent (−slope of log-log SV decay)")
    ax.set_title("Power-Law Exponent of Singular Value Decay Across Layers")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, num_layers - 0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "power_law_exponent_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Q/K vs V/O Effective Rank Comparison ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    qk_ranks = [np.mean([per_layer[i]["q_proj"]["effective_rank_ratio"],
                          per_layer[i]["k_proj"]["effective_rank_ratio"]]) for i in layers]
    vo_ranks = [np.mean([per_layer[i]["v_proj"]["effective_rank_ratio"],
                          per_layer[i]["o_proj"]["effective_rank_ratio"]]) for i in layers]

    axes[0].plot(layers, qk_ranks, "b-o", markersize=4, label="Q/K (routing)", linewidth=2)
    axes[0].plot(layers, vo_ranks, "r-s", markersize=4, label="V/O (value)", linewidth=2)
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Mean Effective Rank Ratio")
    axes[0].set_title("Attention Routing (Q/K) vs Value Processing (V/O)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Difference plot
    diff = [v - q for q, v in zip(qk_ranks, vo_ranks)]
    axes[1].bar(layers, diff, color=["green" if d > 0 else "red" for d in diff], alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("V/O − Q/K Effective Rank Ratio")
    axes[1].set_title("V/O vs Q/K Gap (positive = V/O higher-rank)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "qk_vs_vo_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Cumulative Energy (Rank for 50/90/99% Energy) ────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    thresholds = [("rank_for_50pct", "50%"), ("rank_for_90pct", "90%"), ("rank_for_99pct", "99%")]

    for ax, (key, label) in zip(axes, thresholds):
        for (name, _), color in zip(WEIGHT_NAMES.items(), colors):
            vals = [per_layer[i][name][key] for i in layers]
            ax.plot(layers, vals, marker=".", markersize=3, label=name, color=color, linewidth=1.2)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(f"Rank for {label} Energy")
        ax.set_title(f"Dimensions Needed for {label} Energy")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "cumulative_energy_ranks.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 5: Spectral Entropy Across Layers ───────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    for (name, _), color in zip(WEIGHT_NAMES.items(), colors):
        vals = [per_layer[i][name]["spectral_entropy"] for i in layers]
        ax.plot(layers, vals, marker="^", markersize=3, label=name, color=color, linewidth=1.5)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Spectral Entropy (normalized)")
    ax.set_title("Spectral Entropy of Weight Matrices Across Layers")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, num_layers - 0.5)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "spectral_entropy_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 6: Heatmap of Effective Rank Ratio (layer × matrix) ────────
    fig, ax = plt.subplots(figsize=(10, 14))
    matrix_names = list(WEIGHT_NAMES.keys())
    heatmap_data = np.array([[per_layer[i][name]["effective_rank_ratio"]
                               for name in matrix_names] for i in layers])
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(matrix_names)))
    ax.set_xticklabels(matrix_names, rotation=45, ha="right")
    ax.set_yticks(range(0, num_layers, 2))
    ax.set_yticklabels(range(0, num_layers, 2))
    ax.set_xlabel("Weight Matrix")
    ax.set_ylabel("Layer Index")
    ax.set_title("Effective Rank Ratio Heatmap (layer × matrix)")
    # Add horizontal line at plateau boundary
    ax.axhline(16.5, color="cyan", linewidth=2, linestyle="--", label="Plateau boundary (layer 16/17)")
    ax.legend(loc="lower right", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Effective Rank Ratio")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "heatmap_effective_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 7: SV Spectrum for Representative Layers ────────────────────
    representative_layers = [0, 1, 9, 17, 25, 35]  # early, anomalous, mid-plateau, late-start, late-mid, final
    representative_layers = [l for l in representative_layers if l < num_layers]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax_idx, layer_idx in enumerate(representative_layers):
        ax = axes[ax_idx // 3][ax_idx % 3]
        for (name, _), color in zip(WEIGHT_NAMES.items(), colors):
            svs = per_layer[layer_idx][name]["sv_spectrum_sampled"]
            ax.semilogy(range(len(svs)), svs, label=name, color=color, linewidth=1.2)
        ax.set_xlabel("Singular Value Index (sampled)")
        ax.set_ylabel("Singular Value (log)")
        ax.set_title(f"Layer {layer_idx}")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Singular Value Spectra — Representative Layers", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sv_spectra_representative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 8: Stable Rank vs Effective Rank comparison ─────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    for (name, _), color in zip(WEIGHT_NAMES.items(), colors):
        eff = [per_layer[i][name]["effective_rank_ratio"] for i in layers]
        stb = [per_layer[i][name]["stable_rank_ratio"] for i in layers]
        ax.scatter(eff, stb, label=name, color=color, s=20, alpha=0.7)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Effective Rank Ratio (participation ratio)")
    ax.set_ylabel("Stable Rank Ratio (||W||_F² / ||W||_2²)")
    ax.set_title("Effective Rank vs Stable Rank — All Matrices, All Layers")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "effective_vs_stable_rank.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 9: Plateau vs Late Layer Comparison ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    matrix_names = list(WEIGHT_NAMES.keys())
    x = np.arange(len(matrix_names))
    width = 0.35

    plateau_effs = [np.mean([per_layer[i][name]["effective_rank_ratio"] for i in range(17)]) for name in matrix_names]
    late_effs = [np.mean([per_layer[i][name]["effective_rank_ratio"] for i in range(17, num_layers)]) for name in matrix_names]

    axes[0].bar(x - width/2, plateau_effs, width, label="Plateau (0-16)", color="steelblue")
    axes[0].bar(x + width/2, late_effs, width, label="Late (17-35)", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(matrix_names, rotation=45, ha="right")
    axes[0].set_ylabel("Mean Effective Rank Ratio")
    axes[0].set_title("Effective Rank: Plateau vs Late Layers")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    plateau_plaws = [np.mean([per_layer[i][name]["power_law"]["exponent"] for i in range(17)]) for name in matrix_names]
    late_plaws = [np.mean([per_layer[i][name]["power_law"]["exponent"] for i in range(17, num_layers)]) for name in matrix_names]

    axes[1].bar(x - width/2, plateau_plaws, width, label="Plateau (0-16)", color="steelblue")
    axes[1].bar(x + width/2, late_plaws, width, label="Late (17-35)", color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(matrix_names, rotation=45, ha="right")
    axes[1].set_ylabel("Mean Power-Law Exponent")
    axes[1].set_title("SV Decay Rate: Plateau vs Late Layers")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "plateau_vs_late_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 10: Cross-reference with T-7 if available ───────────────────
    if t7_data is not None:
        _plot_t7_crossref(per_layer, num_layers, t7_data)


def _plot_t7_crossref(per_layer, num_layers, t7_data):
    """Cross-reference effective rank with T-7 linearization gap."""
    layers = list(range(num_layers))

    # Extract linearization gap per layer from T-7
    lin_gaps = None
    if "per_layer" in t7_data:
        pl = t7_data["per_layer"]
        # Try to get a scalar gap per layer
        if isinstance(pl, dict):
            try:
                lin_gaps = {}
                for k, v in pl.items():
                    if not isinstance(v, dict):
                        continue
                    try:
                        idx = int(k.replace("layer_", ""))
                    except ValueError:
                        continue
                    gap = v.get("perturb_gap_mean", v.get("jvp_gap_mean", v.get("linearization_gap", v.get("gap"))))
                    if gap is not None:
                        lin_gaps[idx] = gap
            except (AttributeError, TypeError):
                pass

    if not lin_gaps or len(lin_gaps) < num_layers:
        print("  Could not extract per-layer linearization gap from T-7 results, skipping cross-ref plot")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Mean effective rank ratio per layer (across all matrices)
    mean_eff_rank = []
    gap_vals = []
    for i in layers:
        if i not in lin_gaps or lin_gaps[i] is None:
            continue
        er = np.mean([per_layer[i][n]["effective_rank_ratio"] for n in WEIGHT_NAMES])
        mean_eff_rank.append(er)
        gap_vals.append(lin_gaps[i])

    if len(mean_eff_rank) < 3:
        print("  Not enough matching layers for cross-reference plot")
        plt.close(fig)
        return

    ax.scatter(mean_eff_rank, gap_vals, c=layers[:len(mean_eff_rank)], cmap="viridis", s=50, edgecolors="k", linewidth=0.5)

    # Fit line
    r, p = stats.pearsonr(mean_eff_rank, gap_vals)
    ax.set_xlabel("Mean Effective Rank Ratio (all matrices)")
    ax.set_ylabel("Linearization Gap (T-7)")
    ax.set_title(f"Effective Rank vs Linearization Gap (r={r:.3f}, p={p:.4f})")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Layer Index")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "crossref_t7_linearization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  T-7 cross-reference: Pearson r={r:.3f}, p={p:.4f}")


def _plot_t2_crossref(per_layer, num_layers, t2_criticality):
    """Cross-reference effective rank with T-2 layer knockout criticality."""
    layers = list(range(num_layers))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Per-layer mean effective rank (all matrices)
    mean_eff_rank = [np.mean([per_layer[i][n]["effective_rank_ratio"] for n in WEIGHT_NAMES])
                     for i in layers]
    criticality = [t2_criticality.get(i, 0) for i in layers]

    # Scatter plot
    sc = axes[0].scatter(mean_eff_rank, criticality, c=layers, cmap="viridis",
                          s=60, edgecolors="k", linewidth=0.5)
    r, p = stats.pearsonr(mean_eff_rank, criticality)
    axes[0].set_xlabel("Mean Effective Rank Ratio")
    axes[0].set_ylabel("Layer Criticality (loss delta from T-2)")
    axes[0].set_title(f"Effective Rank vs Layer Criticality (r={r:.3f}, p={p:.4f})")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(sc, ax=axes[0], label="Layer Index")

    # Dual-axis line plot
    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(layers, mean_eff_rank, "b-o", markersize=3, linewidth=1.5, label="Eff Rank Ratio")
    ax2.plot(layers, criticality, "r-s", markersize=3, linewidth=1.5, label="Criticality")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Eff Rank Ratio", color="blue")
    ax2.set_ylabel("Criticality (loss delta)", color="red")
    ax1.set_title("Effective Rank and Criticality Across Layers")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "crossref_t2_criticality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  T-2 cross-reference: Pearson r={r:.3f}, p={p:.4f}")

    # Also check per-matrix correlations
    print("  Per-matrix correlation with criticality:")
    for name in WEIGHT_NAMES:
        eff = [per_layer[i][name]["effective_rank_ratio"] for i in layers]
        r_m, p_m = stats.pearsonr(eff, criticality)
        print(f"    {name:12s}: r={r_m:.3f}, p={p_m:.4f}")

    return r, p


def _plot_t4_crossref(per_layer, num_layers, t4_geometry):
    """Cross-reference weight spectral structure with T-4 representation geometry."""
    layers = list(range(num_layers))

    # Get matching layers
    matched = [(i, t4_geometry[i]) for i in layers if i in t4_geometry]
    if len(matched) < 3:
        print("  Not enough matching layers for T-4 cross-reference")
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    idxs = [m[0] for m in matched]
    iso_scores = [m[1]["isoscore"] for m in matched if m[1]["isoscore"] is not None]
    repr_pr = [m[1]["participation_ratio"] for m in matched if m[1]["participation_ratio"] is not None]

    if len(iso_scores) == len(idxs):
        mean_eff_rank = [np.mean([per_layer[i][n]["effective_rank_ratio"] for n in WEIGHT_NAMES])
                         for i in idxs]
        sc = axes[0].scatter(mean_eff_rank, iso_scores, c=idxs, cmap="viridis",
                              s=60, edgecolors="k", linewidth=0.5)
        r, p = stats.pearsonr(mean_eff_rank, iso_scores)
        axes[0].set_xlabel("Mean Weight Effective Rank Ratio")
        axes[0].set_ylabel("Representation Isotropy (T-4)")
        axes[0].set_title(f"Weight Rank vs Repr Isotropy (r={r:.3f}, p={p:.4f})")
        axes[0].grid(True, alpha=0.3)
        fig.colorbar(sc, ax=axes[0], label="Layer Index")
        print(f"  T-4 isotropy cross-ref: r={r:.3f}, p={p:.4f}")

    if len(repr_pr) == len(idxs):
        mean_eff_rank = [np.mean([per_layer[i][n]["effective_rank_ratio"] for n in WEIGHT_NAMES])
                         for i in idxs]
        sc = axes[1].scatter(mean_eff_rank, repr_pr, c=idxs, cmap="viridis",
                              s=60, edgecolors="k", linewidth=0.5)
        r, p = stats.pearsonr(mean_eff_rank, repr_pr)
        axes[1].set_xlabel("Mean Weight Effective Rank Ratio")
        axes[1].set_ylabel("Representation Participation Ratio (T-4)")
        axes[1].set_title(f"Weight Rank vs Repr PR (r={r:.3f}, p={p:.4f})")
        axes[1].grid(True, alpha=0.3)
        fig.colorbar(sc, ax=axes[1], label="Layer Index")
        print(f"  T-4 participation ratio cross-ref: r={r:.3f}, p={p:.4f}")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "crossref_t4_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    num_layers = len(model.model.layers)
    print(f"  Loaded in {time.time() - t0:.1f}s — {num_layers} layers")

    # ── Print matrix shapes ────────────────────────────────────────────────
    layer0 = model.model.layers[0]
    print("\nWeight matrix shapes (layer 0):")
    for name, attr_path in WEIGHT_NAMES.items():
        obj = layer0
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        print(f"  {name:12s}: {list(obj.shape)}")

    # ── Compute spectral stats ─────────────────────────────────────────────
    print(f"\nComputing SVD for {len(WEIGHT_NAMES)} matrices × {num_layers} layers...")
    t0 = time.time()
    per_layer = analyze_model(model, num_layers)
    analysis_time = time.time() - t0
    print(f"  Analysis completed in {analysis_time:.1f}s")

    # ── Free GPU memory ────────────────────────────────────────────────────
    del model, tokenizer
    torch.cuda.empty_cache()

    # ── Aggregates ─────────────────────────────────────────────────────────
    print("\nComputing aggregates...")
    aggregates = compute_aggregates(per_layer, num_layers)

    # Print key findings
    print("\n=== Key Findings ===")
    comp = aggregates["qk_vs_vo_comparison"]
    print(f"  Q/K mean eff rank ratio: {comp['qk_mean_eff_rank_ratio']:.4f}")
    print(f"  V/O mean eff rank ratio: {comp['vo_mean_eff_rank_ratio']:.4f}")
    print(f"  Q/K higher in {comp['qk_higher_than_vo_layers']}/{num_layers} layers")

    comp2 = aggregates["attn_vs_mlp_comparison"]
    print(f"  Attn mean eff rank ratio: {comp2['attn_mean_eff_rank_ratio']:.4f}")
    print(f"  MLP mean eff rank ratio:  {comp2['mlp_mean_eff_rank_ratio']:.4f}")

    for name in WEIGHT_NAMES:
        a = aggregates[name]
        print(f"  {name:12s}: eff_rank_ratio={a['effective_rank_ratio_mean']:.4f} "
              f"plaw_exp={a['power_law_exponent_mean']:.3f}±{a['power_law_exponent_std']:.3f} "
              f"(R²={a['power_law_r2_mean']:.3f})")

    # ── Print plateau vs late comparison ──────────────────────────────────
    print("\n=== Plateau (0-16) vs Late (17-35) ===")
    pvl = aggregates["plateau_vs_late"]
    for name in WEIGHT_NAMES:
        d = pvl[name]
        sig = "*" if d["p_value"] < 0.05 else ""
        print(f"  {name:12s}: plateau={d['plateau_eff_rank_ratio']:.4f} "
              f"late={d['late_eff_rank_ratio']:.4f} "
              f"diff={d['diff']:+.4f} (p={d['p_value']:.4f}){sig}")

    # ── Load cross-references ──────────────────────────────────────────────
    print("\nLoading cross-reference data...")
    t7_data = load_t7_results()
    t2_criticality = load_t2_criticality()
    t4_geometry = load_t4_geometry()

    for name, data in [("T-7 linearization gap", t7_data),
                       ("T-2 criticality", t2_criticality),
                       ("T-4 geometry", t4_geometry)]:
        print(f"  {name}: {'found' if data else 'not found'}")

    # ── Generate plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    create_plots(per_layer, num_layers, t7_data)

    # Cross-reference plots
    if t2_criticality:
        print("\n--- T-2 Cross-Reference ---")
        t2_corr = _plot_t2_crossref(per_layer, num_layers, t2_criticality)
        aggregates["crossref_t2"] = {
            "pearson_r": round(t2_corr[0], 4),
            "p_value": round(t2_corr[1], 6),
        }

    if t4_geometry:
        print("\n--- T-4 Cross-Reference ---")
        _plot_t4_crossref(per_layer, num_layers, t4_geometry)

    # ── Save results ───────────────────────────────────────────────────────
    # Convert keys to strings for JSON
    per_layer_json = {str(k): v for k, v in per_layer.items()}

    output = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": DEVICE,
            "num_layers": num_layers,
            "weight_matrices": list(WEIGHT_NAMES.keys()),
        },
        "per_layer": per_layer_json,
        "aggregates": aggregates,
        "timing": {
            "analysis_s": round(analysis_time, 1),
            "total_s": round(time.time() - t_total, 1),
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print(f"Total time: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
