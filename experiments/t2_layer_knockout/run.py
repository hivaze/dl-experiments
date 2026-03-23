"""
Layer Knockout / Criticality Mapping (T-2)
==========================================
Skip each layer one at a time (and pairs) and measure loss degradation.
Loss is computed on pre-generated calibration completions (completion tokens
only), loaded from data/text_completions/.

Key questions:
  1. Are there "critical" layers whose removal is catastrophic vs "redundant"
     ones with minimal impact?
  2. Do critical layers cluster or distribute uniformly?
  3. Can you remove N layers while keeping loss below some threshold?
"""

import json
import time
import random
import itertools
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42
DEVICE = "cuda:1"
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
    """Load pre-generated calibration completions from JSON."""
    with open(CALIBRATION_PATH, "r") as f:
        data = json.load(f)
    completions = data["completions"]
    print(f"Loaded {len(completions)} calibration entries from {CALIBRATION_PATH}")
    return completions


def compute_loss(model, tokenizer, calibration_data, device):
    """Compute mean cross-entropy loss on completion tokens only.

    Each calibration entry contains 'full_text' (prompt + completion) and
    'prompt_token_count' indicating where the completion begins. Loss is
    computed only on completion token positions (>= prompt_token_count).
    """
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


# ============================================================================
# Part 1: Single Layer Knockout
# ============================================================================

def run_single_knockouts(model, tokenizer, num_layers, calibration_data):
    """Skip each layer one at a time and measure loss degradation."""
    print("\n" + "=" * 70)
    print("PART 1: Single Layer Knockout")
    print("=" * 70)

    original_layers = list(model.model.layers)

    print("Computing baseline loss (all layers)...")
    baseline_loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
    print(f"Baseline loss: {baseline_loss:.4f}")

    results = {"baseline_loss": baseline_loss, "knockouts": {}}

    for skip_idx in tqdm(range(num_layers), desc="Single knockouts"):
        remaining = [l for i, l in enumerate(original_layers) if i != skip_idx]
        model.model.layers = torch.nn.ModuleList(remaining)

        loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
        delta = loss - baseline_loss
        ratio = loss / baseline_loss

        results["knockouts"][skip_idx] = {
            "loss": loss,
            "loss_delta": delta,
            "loss_ratio": ratio,
        }

    model.model.layers = torch.nn.ModuleList(original_layers)

    print(f"\n--- Single Knockout Results ---")
    sorted_by_impact = sorted(results["knockouts"].items(),
                              key=lambda x: x[1]["loss_delta"], reverse=True)
    print("\nMost critical (highest loss increase):")
    for idx, info in sorted_by_impact[:5]:
        print(f"  Layer {idx}: loss={info['loss']:.4f} (Δ={info['loss_delta']:+.4f}, "
              f"ratio={info['loss_ratio']:.3f}x)")
    print("\nLeast critical (lowest loss increase):")
    for idx, info in sorted_by_impact[-5:]:
        print(f"  Layer {idx}: loss={info['loss']:.4f} (Δ={info['loss_delta']:+.4f}, "
              f"ratio={info['loss_ratio']:.3f}x)")

    return results


# ============================================================================
# Part 2: Pair Layer Knockout
# ============================================================================

def run_pair_knockouts(model, tokenizer, num_layers, single_results, calibration_data):
    """Skip pairs of layers and measure loss."""
    print("\n" + "=" * 70)
    print("PART 2: Pair Layer Knockout")
    print("=" * 70)

    original_layers = list(model.model.layers)
    baseline_loss = single_results["baseline_loss"]

    sorted_by_impact = sorted(single_results["knockouts"].items(),
                              key=lambda x: x[1]["loss_delta"], reverse=True)
    top5_critical = [int(idx) for idx, _ in sorted_by_impact[:5]]
    top5_redundant = [int(idx) for idx, _ in sorted_by_impact[-5:]]

    pairs_to_test = set()

    # Adjacent pairs
    for i in range(num_layers - 1):
        pairs_to_test.add((i, i + 1))
    # Critical x critical
    for a, b in itertools.combinations(top5_critical, 2):
        pairs_to_test.add((min(a, b), max(a, b)))
    # Redundant x redundant
    for a, b in itertools.combinations(top5_redundant, 2):
        pairs_to_test.add((min(a, b), max(a, b)))
    # Critical x redundant
    for a in top5_critical[:3]:
        for b in top5_redundant[:3]:
            pairs_to_test.add((min(a, b), max(a, b)))
    # Evenly spaced
    for i in range(0, num_layers, 4):
        for j in range(i + 4, num_layers, 4):
            pairs_to_test.add((i, j))

    pairs_to_test = sorted(pairs_to_test)
    print(f"Testing {len(pairs_to_test)} layer pairs...")

    results = {"baseline_loss": baseline_loss, "pair_knockouts": {}}

    for skip_a, skip_b in tqdm(pairs_to_test, desc="Pair knockouts"):
        remaining = [l for i, l in enumerate(original_layers)
                     if i != skip_a and i != skip_b]
        model.model.layers = torch.nn.ModuleList(remaining)

        loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
        delta = loss - baseline_loss

        single_a = single_results["knockouts"][skip_a]["loss_delta"]
        single_b = single_results["knockouts"][skip_b]["loss_delta"]
        expected_additive = single_a + single_b
        synergy = delta - expected_additive

        key = f"{skip_a},{skip_b}"
        results["pair_knockouts"][key] = {
            "loss": loss,
            "loss_delta": delta,
            "expected_additive_delta": expected_additive,
            "synergy": synergy,
        }

    model.model.layers = torch.nn.ModuleList(original_layers)

    print(f"\n--- Pair Knockout Results ---")
    sorted_pairs = sorted(results["pair_knockouts"].items(),
                          key=lambda x: x[1]["loss_delta"], reverse=True)
    print("\nMost damaging pairs:")
    for key, info in sorted_pairs[:5]:
        print(f"  Layers ({key}): loss={info['loss']:.4f} (Δ={info['loss_delta']:+.4f}, "
              f"synergy={info['synergy']:+.4f})")
    print("\nLeast damaging pairs:")
    for key, info in sorted_pairs[-5:]:
        print(f"  Layers ({key}): loss={info['loss']:.4f} (Δ={info['loss_delta']:+.4f}, "
              f"synergy={info['synergy']:+.4f})")

    sorted_synergy = sorted(results["pair_knockouts"].items(),
                            key=lambda x: x[1]["synergy"], reverse=True)
    print("\nMost synergistic (super-additive) pairs:")
    for key, info in sorted_synergy[:5]:
        print(f"  Layers ({key}): synergy={info['synergy']:+.4f}")

    return results


# ============================================================================
# Part 3: Greedy N-Layer Pruning
# ============================================================================

def run_greedy_pruning(model, tokenizer, num_layers, single_results, calibration_data):
    """Greedily remove layers one at a time, always removing the least
    critical remaining layer."""
    print("\n" + "=" * 70)
    print("PART 3: Greedy N-Layer Pruning")
    print("=" * 70)

    original_layers = list(model.model.layers)
    baseline_loss = single_results["baseline_loss"]

    remaining_indices = list(range(num_layers))
    removal_order = []
    losses = [baseline_loss]

    max_removals = num_layers // 2

    for step in tqdm(range(max_removals), desc="Greedy pruning"):
        best_idx = None
        best_loss = float("inf")

        for candidate in remaining_indices:
            test_indices = [i for i in remaining_indices if i != candidate]
            test_layers = [original_layers[i] for i in test_indices]
            model.model.layers = torch.nn.ModuleList(test_layers)
            loss = compute_loss(model, tokenizer, calibration_data, DEVICE)
            if loss < best_loss:
                best_loss = loss
                best_idx = candidate

        remaining_indices.remove(best_idx)
        removal_order.append(best_idx)
        losses.append(best_loss)

        delta = best_loss - baseline_loss
        pct = (best_loss / baseline_loss - 1) * 100
        print(f"  Step {step+1}: removed layer {best_idx}, "
              f"loss={best_loss:.4f} (Δ={delta:+.4f}, {pct:+.1f}%), "
              f"{len(remaining_indices)} layers remain")

        if best_loss > baseline_loss * 3:
            print("  Loss > 3x baseline, stopping early.")
            break

    model.model.layers = torch.nn.ModuleList(original_layers)

    results = {
        "baseline_loss": baseline_loss,
        "removal_order": removal_order,
        "losses_after_removal": losses,
        "remaining_layers_at_end": remaining_indices,
    }

    for threshold in [1.05, 1.10, 1.25, 1.50, 2.0]:
        max_removable = 0
        for i, loss in enumerate(losses):
            if loss <= baseline_loss * threshold:
                max_removable = i
        results[f"max_removable_{int(threshold*100)}pct"] = max_removable

    return results


# ============================================================================
# Visualization
# ============================================================================

def create_plots(single_results, pair_results, pruning_results, num_layers):
    """Generate visualization plots for Parts 1-3."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

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

    layers = list(range(num_layers))
    model_short = MODEL_NAME.split("/")[-1]
    deltas = [single_results["knockouts"][i]["loss_delta"] for i in layers]

    # =========================================================================
    # Plot 1: Single knockout criticality profile (2-panel)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Loss delta with gradient coloring
    ax = axes[0]
    norm_deltas = np.array(deltas)
    colors = plt.cm.RdYlBu_r((norm_deltas - norm_deltas.min()) /
                               (norm_deltas.max() - norm_deltas.min() + 1e-10))
    ax.bar(layers, deltas, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=np.mean(deltas), color="red", linestyle="--", alpha=0.5,
               label=f"mean={np.mean(deltas):.3f}")
    top3 = sorted(range(num_layers), key=lambda i: deltas[i], reverse=True)[:3]
    bot3 = sorted(range(num_layers), key=lambda i: deltas[i])[:3]
    for idx in top3:
        ax.annotate(f"L{idx}\n{deltas[idx]:+.2f}", (idx, deltas[idx]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#B71C1C")
    for idx in bot3:
        ax.annotate(f"L{idx}\n{deltas[idx]:+.2f}", (idx, deltas[idx]),
                    ha="center", va="top", fontsize=8, fontweight="bold", color="#1B5E20")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss Delta")
    ax.set_title("Single Layer Knockout: Loss Impact")
    ax.legend(fontsize=9)

    # Panel 2: Ranked criticality
    ax = axes[1]
    sorted_idx = sorted(layers, key=lambda i: deltas[i], reverse=True)
    sorted_deltas = [deltas[i] for i in sorted_idx]
    bar_colors = ["#D32F2F" if i < 5 else "#1976D2" if i >= num_layers - 5
                  else "#9E9E9E" for i in range(len(sorted_idx))]
    ax.bar(range(num_layers), sorted_deltas, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([str(i) for i in sorted_idx], fontsize=6, rotation=45)
    ax.set_xlabel("Layer (ranked by criticality)")
    ax.set_ylabel("Loss Delta")
    ax.set_title("Layers Ranked by Criticality")

    fig.suptitle(f"Layer Knockout Criticality — {model_short}",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "single_knockout_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 2: Pair knockout heatmaps
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    loss_matrix = np.full((num_layers, num_layers), np.nan)
    synergy_matrix = np.full((num_layers, num_layers), np.nan)
    for key, info in pair_results["pair_knockouts"].items():
        a, b = map(int, key.split(","))
        loss_matrix[a, b] = info["loss_delta"]
        loss_matrix[b, a] = info["loss_delta"]
        synergy_matrix[a, b] = info["synergy"]
        synergy_matrix[b, a] = info["synergy"]

    ax = axes[0]
    mask = np.isnan(loss_matrix)
    sns.heatmap(loss_matrix, ax=ax, cmap="YlOrRd", mask=mask,
                square=True, cbar_kws={"label": "Loss delta", "shrink": 0.8},
                xticklabels=2, yticklabels=2, linewidths=0.1, linecolor="white")
    ax.set_xlabel("Layer B")
    ax.set_ylabel("Layer A")
    ax.set_title("Pair Knockout: Loss Delta")

    ax = axes[1]
    vmax = np.nanpercentile(np.abs(synergy_matrix), 95)
    mask = np.isnan(synergy_matrix)
    sns.heatmap(synergy_matrix, ax=ax, cmap="RdBu_r", mask=mask,
                vmin=-vmax, vmax=vmax, square=True,
                cbar_kws={"label": "Synergy", "shrink": 0.8},
                xticklabels=2, yticklabels=2, linewidths=0.1, linecolor="white")
    ax.set_xlabel("Layer B")
    ax.set_ylabel("Layer A")
    ax.set_title("Pair Knockout: Synergy (>0 = super-additive)")

    fig.suptitle(f"Pair Knockout Analysis — {model_short}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pair_knockout_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 3: Greedy pruning curve
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    n_removed = list(range(len(pruning_results["losses_after_removal"])))
    losses = pruning_results["losses_after_removal"]
    baseline = pruning_results["baseline_loss"]

    ax = axes[0]
    ax.fill_between(n_removed, baseline, losses, where=[l > baseline for l in losses],
                     alpha=0.15, color="red")
    ax.fill_between(n_removed, baseline, losses, where=[l <= baseline for l in losses],
                     alpha=0.15, color="green")
    ax.plot(n_removed, losses, "-o", color="#1976D2", markersize=5, linewidth=2)
    ax.axhline(y=baseline, color="green", linestyle="--", alpha=0.7, label="Baseline")
    ax.axhline(y=baseline * 1.1, color="orange", linestyle="--", alpha=0.5, label="+10%")
    ax.axhline(y=baseline * 1.5, color="red", linestyle="--", alpha=0.5, label="+50%")

    for i, layer_idx in enumerate(pruning_results["removal_order"]):
        if i < 10 or i % 3 == 0:
            ax.annotate(f"L{layer_idx}", (i + 1, losses[i + 1]),
                        fontsize=7, ha="center", va="bottom", rotation=45)
    ax.set_xlabel("Layers Removed")
    ax.set_ylabel("Loss")
    ax.set_title("Greedy Pruning: Loss Trajectory")
    ax.legend(fontsize=9)

    ax = axes[1]
    pct_changes = [(l / baseline - 1) * 100 for l in losses]
    colors = ["#4CAF50" if p <= 0 else "#FF9800" if p < 10 else "#E53935" for p in pct_changes]
    ax.bar(n_removed, pct_changes, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=10, color="orange", linestyle="--", alpha=0.5, label="+10%")
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="+50%")
    ax.set_xlabel("Layers Removed")
    ax.set_ylabel("Loss Change (%)")
    ax.set_title("Loss Change vs Layers Removed")
    ax.legend(fontsize=9)

    fig.suptitle(f"Greedy Layer Pruning — {model_short}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "greedy_pruning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {RESULTS_DIR}/")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} decoder layers on {DEVICE}")
    t_load = time.time()
    print(f"Load time: {t_load - t_start:.1f}s")

    print("\nLoading calibration data...")
    calibration_data = load_calibration_data()

    # Part 1: Single knockouts
    t1 = time.time()
    single_results = run_single_knockouts(model, tokenizer, num_layers, calibration_data)
    t2 = time.time()
    print(f"Single knockout time: {t2 - t1:.1f}s")

    # Part 2: Pair knockouts
    pair_results = run_pair_knockouts(model, tokenizer, num_layers, single_results, calibration_data)
    t3 = time.time()
    print(f"Pair knockout time: {t3 - t2:.1f}s")

    # Part 3: Greedy pruning
    pruning_results = run_greedy_pruning(model, tokenizer, num_layers, single_results, calibration_data)
    t4 = time.time()
    print(f"Greedy pruning time: {t4 - t3:.1f}s")

    # Generate plots
    print("\nGenerating plots...")
    create_plots(single_results, pair_results, pruning_results, num_layers)

    # Save results
    all_results = {
        "config": {
            "model": MODEL_NAME,
            "num_layers": num_layers,
            "num_calibration_entries": len(calibration_data),
            "calibration_source": str(CALIBRATION_PATH),
            "seed": SEED,
        },
        "single_knockouts": single_results,
        "pair_knockouts": pair_results,
        "pruning": pruning_results,
        "timing": {
            "load_s": t_load - t_start,
            "single_knockout_s": t2 - t1,
            "pair_knockout_s": t3 - t2,
            "pruning_s": t4 - t3,
            "total_s": time.time() - t_start,
        },
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_time = time.time() - t_start
    print(f"\nTotal experiment time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
