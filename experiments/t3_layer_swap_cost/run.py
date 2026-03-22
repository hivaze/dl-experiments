"""
Layer Swap Cost Matrix (T-3)
============================
For every pair of layers (i, j), swap them and measure loss degradation.
Produces an N×N "swap cost" matrix — a functional distance between layers.

Uses pre-generated calibration completions for loss computation: the model is
scored on its ability to predict completion tokens given the prompt context,
providing a more stable and representative evaluation signal than prompt-only
next-token loss.

Key questions:
  1. Which pairs are nearly interchangeable (low swap cost)?
  2. Is the swap cost matrix symmetric?
  3. Cluster layers by swap cost — do clusters align with findings from T-1/T-2?
  4. Which layers can be reordered for pipeline parallelism without quality loss?

Model: Qwen/Qwen3-4B-Instruct-2507
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42
DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).parent / "results"
CALIBRATION_PATH = Path(__file__).parents[2] / "data" / "text_completions" / "qwen3-4b-instruct-2507" / "completions.json"


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_calibration_data():
    """Load pre-generated calibration completions from disk."""
    with open(CALIBRATION_PATH, "r") as f:
        data = json.load(f)
    completions = data["completions"]
    print(f"Loaded {len(completions)} calibration entries from {CALIBRATION_PATH}")
    return completions


def compute_loss(model, tokenizer, calibration_data, device):
    """Compute mean cross-entropy loss on completion tokens across calibration entries."""
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
        # Loss only on completion positions: predict token at pos t from pos t-1
        # Completion tokens start at position prompt_token_count
        # In shifted logits (pos t-1 predicts pos t), we need logits from
        # position (prompt_token_count - 1) onward to predict tokens at
        # position prompt_token_count onward
        logits = outputs.logits[:, prompt_token_count - 1:-1, :]
        targets = input_ids[:, prompt_token_count:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += targets.numel()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


# ============================================================================
# Part 1: Full Swap Cost Matrix
# ============================================================================

def run_swap_cost_matrix(model, tokenizer, num_layers, calibration_data):
    """Swap each pair of layers (i, j) and measure loss degradation."""
    print("\n" + "=" * 70)
    print("PART 1: Full Swap Cost Matrix")
    print("=" * 70)

    original_layers = list(model.model.layers)

    print("Computing baseline loss (original order)...")
    baseline_loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
    print(f"Baseline loss: {baseline_loss:.4f}")

    n_pairs = num_layers * (num_layers - 1) // 2
    print(f"\nSwapping all {n_pairs} unique pairs of {num_layers} layers...")

    swap_loss = np.full((num_layers, num_layers), np.nan)
    swap_delta = np.full((num_layers, num_layers), 0.0)

    for i in range(num_layers):
        swap_loss[i, i] = baseline_loss
        swap_delta[i, i] = 0.0

    t0 = time.time()

    for i in tqdm(range(num_layers), desc="Swap rows"):
        for j in range(i + 1, num_layers):
            swapped = list(original_layers)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            model.model.layers = nn.ModuleList(swapped)

            loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
            delta = loss - baseline_loss

            swap_loss[i, j] = loss
            swap_loss[j, i] = loss
            swap_delta[i, j] = delta
            swap_delta[j, i] = delta

    elapsed = time.time() - t0
    print(f"Swap cost matrix computed in {elapsed:.1f}s ({elapsed / n_pairs:.2f}s per pair)")

    model.model.layers = nn.ModuleList(original_layers)

    results = {
        "baseline_loss": baseline_loss,
        "swap_loss_matrix": swap_loss.tolist(),
        "swap_delta_matrix": swap_delta.tolist(),
        "num_layers": num_layers,
        "elapsed_seconds": elapsed,
        "calibration_source": str(CALIBRATION_PATH),
        "num_calibration_entries": len(calibration_data),
    }
    return results


# ============================================================================
# Part 2: Analysis
# ============================================================================

def analyze_swap_matrix(swap_results):
    """Analyze the swap cost matrix for structure."""
    print("\n" + "=" * 70)
    print("PART 2: Swap Cost Matrix Analysis")
    print("=" * 70)

    delta = np.array(swap_results["swap_delta_matrix"])
    n = swap_results["num_layers"]
    baseline = swap_results["baseline_loss"]

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j, delta[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 most costly swaps:")
    for i, j, d in pairs[:10]:
        print(f"  Swap({i}, {j}): Δloss = {d:+.4f} (loss = {baseline + d:.4f})")

    print("\nTop 10 cheapest swaps:")
    for i, j, d in pairs[-10:]:
        print(f"  Swap({i}, {j}): Δloss = {d:+.4f} (loss = {baseline + d:.4f})")

    threshold_pct = 0.01
    interchangeable = [(i, j, d) for i, j, d in pairs if d / baseline < threshold_pct]
    print(f"\nNear-interchangeable pairs (<{threshold_pct*100:.0f}% loss increase): "
          f"{len(interchangeable)} of {len(pairs)}")
    for i, j, d in interchangeable[:20]:
        print(f"  Swap({i}, {j}): Δloss = {d:+.4f} ({d/baseline*100:.2f}%)")

    avg_cost = np.zeros(n)
    for i in range(n):
        costs = [delta[i, j] for j in range(n) if j != i]
        avg_cost[i] = np.mean(costs)

    print("\nPer-layer average swap cost:")
    order = np.argsort(avg_cost)
    for rank, idx in enumerate(order):
        if rank < 5 or rank >= n - 5 or rank == n // 2:
            marker = " ***" if rank < 5 else (" ..." if rank >= n - 5 else "")
            print(f"  Layer {idx}: avg Δloss = {avg_cost[idx]:+.4f}{marker}")

    distance_costs = {}
    for i, j, d in pairs:
        dist = abs(i - j)
        distance_costs.setdefault(dist, []).append(d)
    avg_by_distance = {d: np.mean(v) for d, v in sorted(distance_costs.items())}
    print("\nSwap cost vs layer distance:")
    for dist, avg in avg_by_distance.items():
        if dist <= 5 or dist % 5 == 0 or dist == n - 1:
            print(f"  |i-j| = {dist:2d}: avg Δloss = {avg:+.4f}")

    dist_matrix = np.maximum(delta, 0.0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method="ward")

    cluster_results = {}
    for n_clusters in [3, 4, 5, 6]:
        labels = fcluster(Z, n_clusters, criterion="maxclust")
        clusters = {}
        for layer_idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(layer_idx)
        cluster_results[n_clusters] = clusters
        print(f"\n{n_clusters} clusters (Ward linkage):")
        for cid in sorted(clusters):
            members = clusters[cid]
            print(f"  Cluster {cid}: layers {members}")

    analysis = {
        "top_costly_swaps": [(int(i), int(j), float(d)) for i, j, d in pairs[:20]],
        "top_cheap_swaps": [(int(i), int(j), float(d)) for i, j, d in pairs[-20:]],
        "interchangeable_pairs_1pct": [(int(i), int(j), float(d)) for i, j, d in interchangeable],
        "per_layer_avg_cost": avg_cost.tolist(),
        "avg_cost_by_distance": {str(k): float(v) for k, v in avg_by_distance.items()},
        "clusters": {str(k): v for k, v in cluster_results.items()},
        "linkage_matrix": Z.tolist(),
    }
    return analysis


# ============================================================================
# Part 3: Visualization (Enhanced)
# ============================================================================

def plot_results(swap_results, analysis):
    """Generate enhanced plots."""
    print("\n" + "=" * 70)
    print("PART 3: Generating Plots")
    print("=" * 70)

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    delta = np.array(swap_results["swap_delta_matrix"])
    n = swap_results["num_layers"]
    model_short = MODEL_NAME.split("/")[-1]

    # 1. Swap cost heatmap (enhanced)
    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(delta, ax=ax, cmap="RdYlBu_r", center=0, mask=mask,
                xticklabels=range(n), yticklabels=range(n),
                square=True, cbar_kws={"label": "Δ Loss", "shrink": 0.8},
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title(f"Layer Swap Cost Matrix — {model_short}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "swap_cost_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved swap_cost_heatmap.png")

    # 2. Swap cost vs distance (enhanced with violin overlay)
    distances = []
    costs = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(abs(i - j))
            costs.append(delta[i, j])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(distances, costs, alpha=0.25, s=15, c="#1976D2", edgecolors="none")
    avg_by_dist = analysis["avg_cost_by_distance"]
    dists_sorted = sorted(avg_by_dist.keys(), key=int)
    ax.plot([int(d) for d in dists_sorted],
            [avg_by_dist[d] for d in dists_sorted],
            "-o", color="#D32F2F", linewidth=2.5, markersize=5, label="Mean", zorder=5)
    # Add std bands
    dist_to_costs = {}
    for d_val, c_val in zip(distances, costs):
        dist_to_costs.setdefault(d_val, []).append(c_val)
    std_vals = [np.std(dist_to_costs[int(d)]) for d in dists_sorted]
    mean_vals = [avg_by_dist[d] for d in dists_sorted]
    d_ints = [int(d) for d in dists_sorted]
    ax.fill_between(d_ints, [m - s for m, s in zip(mean_vals, std_vals)],
                     [m + s for m, s in zip(mean_vals, std_vals)],
                     alpha=0.15, color="#D32F2F", label="±1 std")
    ax.set_xlabel("Layer Distance |i − j|")
    ax.set_ylabel("Δ Loss (swap cost)")
    ax.set_title(f"Swap Cost vs Layer Distance — {model_short}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "swap_cost_vs_distance.png", dpi=150)
    plt.close(fig)
    print("  Saved swap_cost_vs_distance.png")

    # 3. Per-layer average swap cost (enhanced with region annotations)
    avg_cost = np.array(analysis["per_layer_avg_cost"])
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.RdYlBu_r(avg_cost / avg_cost.max())
    bars = ax.bar(range(n), avg_cost, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Average Swap Cost (Δ Loss)")
    ax.set_title(f"Per-Layer Position Sensitivity — {model_short}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(0, n, 2))
    # Annotate top 3 and bottom 3
    top3 = sorted(range(n), key=lambda i: avg_cost[i], reverse=True)[:3]
    bot3 = sorted(range(n), key=lambda i: avg_cost[i])[:3]
    for idx in top3:
        ax.annotate(f"L{idx}\n{avg_cost[idx]:.2f}", (idx, avg_cost[idx]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#B71C1C")
    for idx in bot3:
        ax.annotate(f"L{idx}\n{avg_cost[idx]:.2f}", (idx, avg_cost[idx]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1B5E20")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "per_layer_avg_swap_cost.png", dpi=150)
    plt.close(fig)
    print("  Saved per_layer_avg_swap_cost.png")

    # 4. Dendrogram (enhanced with colored clusters)
    from scipy.cluster.hierarchy import dendrogram
    Z = np.array(analysis["linkage_matrix"])
    fig, ax = plt.subplots(figsize=(16, 7))
    dn = dendrogram(Z, ax=ax, labels=[str(i) for i in range(n)],
                    leaf_rotation=0, leaf_font_size=10,
                    color_threshold=Z[-3, 2],  # Color by 4-cluster level
                    above_threshold_color="#9E9E9E")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Ward Distance")
    ax.set_title(f"Hierarchical Clustering by Swap Cost — {model_short}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "swap_cost_dendrogram.png", dpi=150)
    plt.close(fig)
    print("  Saved swap_cost_dendrogram.png")

    # 5. Log-scale heatmap
    fig, ax = plt.subplots(figsize=(13, 11))
    log_delta = np.log10(np.maximum(delta, 1e-6))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(log_delta, ax=ax, cmap="RdYlBu_r", mask=mask,
                xticklabels=range(n), yticklabels=range(n),
                square=True, cbar_kws={"label": "log₁₀(Δ Loss)", "shrink": 0.8},
                linewidths=0.3, linecolor="white")
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title(f"Swap Cost Matrix (log scale) — {model_short}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "swap_cost_heatmap_log.png", dpi=150)
    plt.close(fig)
    print("  Saved swap_cost_heatmap_log.png")

    # 6. NEW: Interchangeability map — highlight near-interchangeable pairs
    fig, ax = plt.subplots(figsize=(13, 11))
    baseline = swap_results["baseline_loss"]
    pct_change = delta / baseline * 100
    # Clip for visualization
    pct_clipped = np.clip(pct_change, -5, 50)
    mask = np.eye(n, dtype=bool)
    sns.heatmap(pct_clipped, ax=ax, cmap="RdYlGn_r", center=0, mask=mask,
                xticklabels=range(n), yticklabels=range(n),
                square=True, cbar_kws={"label": "% Loss Change", "shrink": 0.8},
                linewidths=0.3, linecolor="white",
                vmin=-5, vmax=20)
    ax.set_xlabel("Layer j")
    ax.set_ylabel("Layer i")
    ax.set_title(f"Swap Cost (% Loss Change) — {model_short}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "swap_cost_pct_change.png", dpi=150)
    plt.close(fig)
    print("  Saved swap_cost_pct_change.png")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    calibration_data = load_calibration_data()

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Calibration entries: {len(calibration_data)}")

    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.model.layers)
    print(f"Model loaded in {time.time() - t0:.1f}s: {num_layers} decoder layers")

    # Part 1: Swap cost matrix
    swap_results = run_swap_cost_matrix(model, tokenizer, num_layers, calibration_data)

    with open(RESULTS_DIR / "swap_cost_raw.json", "w") as f:
        json.dump(swap_results, f, indent=2)
    print(f"\nSaved raw swap cost matrix")

    # Part 2: Analysis
    analysis = analyze_swap_matrix(swap_results)

    with open(RESULTS_DIR / "swap_cost_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis")

    # Part 3: Plots
    plot_results(swap_results, analysis)

    total_time = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total experiment time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
