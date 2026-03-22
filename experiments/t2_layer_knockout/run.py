"""
Layer Knockout / Criticality Mapping (T-2)
==========================================
Skip each layer one at a time (and pairs) and measure loss degradation.
Loss is computed on pre-generated calibration completions (completion tokens
only), loaded from data/calibration/completions.json.

Key questions:
  1. Are there "critical" layers whose removal is catastrophic vs "redundant"
     ones with minimal impact?
  2. Is criticality correlated with weight norm?
  3. Do critical layers cluster or distribute uniformly?
  4. Can you remove N layers while keeping loss below some threshold?
  5. Is the "causal bottleneck" (activation patching) the same as the "most
     critical" layer from knockout?
"""

import json
import time
import random
import itertools
from pathlib import Path
from collections import defaultdict

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

# Activation patching prompts — factual/linguistic/arithmetic with known answers
PATCHING_PROMPTS = [
    {"prompt": "The capital of France is", "subject": "France", "target": " Paris", "type": "factual"},
    {"prompt": "The largest ocean on Earth is the", "subject": "ocean on Earth", "target": " Pacific", "type": "factual"},
    {"prompt": "The element with atomic number 79 is", "subject": "atomic number 79", "target": " gold", "type": "factual"},
    {"prompt": "The square root of 144 is", "subject": "square root of 144", "target": " 12", "type": "arithmetic"},
    {"prompt": "If a train travels 120 km in 2 hours, its speed is", "subject": "120 km in 2 hours", "target": " 60", "type": "arithmetic"},
    {"prompt": "The quick brown fox jumps over the lazy", "subject": "fox jumps over", "target": " dog", "type": "linguistic"},
    {"prompt": "Time flies like an arrow, fruit flies like a", "subject": "fruit flies like", "target": " banana", "type": "linguistic"},
    {"prompt": "The horse raced past the barn", "subject": "horse raced past", "target": " fell", "type": "linguistic"},
]


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
        # Shift logits and labels for next-token prediction
        logits = outputs.logits[:, :-1, :]   # (1, seq_len-1, vocab)
        targets = input_ids[:, 1:]            # (1, seq_len-1)
        # Only compute loss on completion positions:
        # target at position i corresponds to predicting token i+1 from token i.
        # Completion tokens start at index prompt_token_count, so we want
        # targets from index (prompt_token_count - 1) onward (predicting
        # the first completion token) through the end.
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


def compute_weight_norms(model, num_layers):
    """Compute Frobenius norm of all weights per layer."""
    norms = {}
    for i in range(num_layers):
        layer = model.model.layers[i]
        total_norm = 0.0
        for p in layer.parameters():
            total_norm += p.data.float().norm().item() ** 2
        norms[i] = total_norm ** 0.5
    return norms


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
# Part 4: Weight Norm vs Criticality Correlation
# ============================================================================

def analyze_weight_norm_correlation(single_results, weight_norms, num_layers):
    """Compute correlation between layer criticality and weight norm."""
    print("\n" + "=" * 70)
    print("PART 4: Weight Norm vs Criticality Correlation")
    print("=" * 70)

    deltas = [single_results["knockouts"][i]["loss_delta"] for i in range(num_layers)]
    norms = [weight_norms[i] for i in range(num_layers)]

    from scipy.stats import spearmanr, pearsonr

    pearson_r, pearson_p = pearsonr(deltas, norms)
    spearman_r, spearman_p = spearmanr(deltas, norms)

    print(f"  Pearson r={pearson_r:.4f} (p={pearson_p:.4e})")
    print(f"  Spearman ρ={spearman_r:.4f} (p={spearman_p:.4e})")

    results = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "per_layer": {
            i: {"loss_delta": deltas[i], "weight_norm": norms[i]}
            for i in range(num_layers)
        },
    }

    return results


# ============================================================================
# Part 5: Activation Patching (Causal Bottleneck)
# ============================================================================

def run_activation_patching(model, tokenizer, num_layers):
    """Causal tracing: corrupt subject tokens' embeddings, then restore
    clean hidden states at individual (layer, last-subject-position) to
    find where factual associations are stored."""
    print("\n" + "=" * 70)
    print("PART 5: Activation Patching / Causal Tracing")
    print("=" * 70)

    results = {"per_prompt": [], "per_type": defaultdict(list)}
    noise_level = 3.0

    for pinfo in tqdm(PATCHING_PROMPTS, desc="Activation patching"):
        prompt = pinfo["prompt"]
        target_str = pinfo["target"]
        subject_str = pinfo["subject"]
        prompt_type = pinfo["type"]

        # Use raw tokenization for patching
        tokens = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        input_ids = tokens["input_ids"]
        target_ids = tokenizer(target_str, add_special_tokens=False,
                               return_tensors="pt")["input_ids"][0]
        if len(target_ids) == 0:
            continue
        target_id = target_ids[0].item()

        # Find subject token positions
        subject_tokens = tokenizer(subject_str, add_special_tokens=False,
                                   return_tensors="pt")["input_ids"][0]
        subject_positions = []
        for start in range(len(input_ids[0]) - len(subject_tokens) + 1):
            if all(input_ids[0, start + j] == subject_tokens[j]
                   for j in range(len(subject_tokens))):
                subject_positions = list(range(start, start + len(subject_tokens)))
                break
        if not subject_positions:
            n = input_ids.shape[1]
            subject_positions = list(range(n // 3, 2 * n // 3))
        last_subject_pos = subject_positions[-1]

        # Step 1: Clean run
        clean_states = {}

        def make_capture_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    clean_states[layer_idx] = output[0].detach().clone()
                else:
                    clean_states[layer_idx] = output.detach().clone()
            return hook_fn

        hooks = []
        for i in range(num_layers):
            h = model.model.layers[i].register_forward_hook(make_capture_hook(i))
            hooks.append(h)

        with torch.no_grad():
            clean_out = model(**tokens, use_cache=False)
        clean_logits = clean_out.logits[0, -1, :]
        clean_prob = F.softmax(clean_logits.float(), dim=-1)[target_id].item()
        clean_rank = (clean_logits > clean_logits[target_id]).sum().item()

        for h in hooks:
            h.remove()

        # Step 2: Corrupted embeddings
        with torch.no_grad():
            clean_embed = model.model.embed_tokens(input_ids).detach().clone()
            embed_std = clean_embed.std().item()
            corrupted_embed = clean_embed.clone()
            noise = torch.randn_like(corrupted_embed[:, subject_positions, :])
            corrupted_embed[:, subject_positions, :] += noise_level * embed_std * noise

        # Step 3: Fully corrupted run
        with torch.no_grad():
            orig_embed_forward = model.model.embed_tokens.forward
            model.model.embed_tokens.forward = lambda x: corrupted_embed
            corrupt_out = model(**tokens, use_cache=False)
            model.model.embed_tokens.forward = orig_embed_forward
        corrupt_logits = corrupt_out.logits[0, -1, :]
        corrupt_prob = F.softmax(corrupt_logits.float(), dim=-1)[target_id].item()
        corrupt_rank = (corrupt_logits > corrupt_logits[target_id]).sum().item()

        # Step 4: Restore clean state at each layer
        layer_recoveries = []

        for restore_layer in range(num_layers):
            def make_patch_hook(layer_idx, clean_activation, pos):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                        patched[0, pos, :] = clean_activation[0, pos, :]
                        return (patched,) + output[1:]
                    else:
                        patched = output.clone()
                        patched[0, pos, :] = clean_activation[0, pos, :]
                        return patched
                return hook_fn

            h = model.model.layers[restore_layer].register_forward_hook(
                make_patch_hook(restore_layer, clean_states[restore_layer],
                                last_subject_pos)
            )

            with torch.no_grad():
                orig_embed_forward = model.model.embed_tokens.forward
                model.model.embed_tokens.forward = lambda x: corrupted_embed
                patched_out = model(**tokens, use_cache=False)
                model.model.embed_tokens.forward = orig_embed_forward

            h.remove()

            patched_logits = patched_out.logits[0, -1, :]
            patched_prob = F.softmax(patched_logits.float(), dim=-1)[target_id].item()
            patched_rank = (patched_logits > patched_logits[target_id]).sum().item()

            denom = clean_prob - corrupt_prob
            if abs(denom) > 1e-10:
                recovery = (patched_prob - corrupt_prob) / denom
            else:
                recovery = 0.0

            layer_recoveries.append({
                "layer": restore_layer,
                "patched_prob": patched_prob,
                "patched_rank": patched_rank,
                "recovery_ratio": recovery,
            })

        prompt_result = {
            "prompt": prompt,
            "target": target_str,
            "subject": subject_str,
            "type": prompt_type,
            "subject_positions": subject_positions,
            "last_subject_pos": last_subject_pos,
            "clean_prob": clean_prob,
            "clean_rank": clean_rank,
            "corrupt_prob": corrupt_prob,
            "corrupt_rank": corrupt_rank,
            "layer_recoveries": layer_recoveries,
        }
        results["per_prompt"].append(prompt_result)

        best_layer = max(layer_recoveries, key=lambda x: x["recovery_ratio"])
        print(f"  '{prompt}' → '{target_str}': "
              f"bottleneck=L{best_layer['layer']} "
              f"(recovery={best_layer['recovery_ratio']:.3f})")

    # Aggregate by type
    type_bottlenecks = defaultdict(list)
    for pr in results["per_prompt"]:
        best = max(pr["layer_recoveries"], key=lambda x: x["recovery_ratio"])
        type_bottlenecks[pr["type"]].append(best["layer"])

    results["type_bottlenecks"] = {}
    print("\n--- Causal Bottleneck by Prompt Type ---")
    for ptype, layers in sorted(type_bottlenecks.items()):
        mean_layer = np.mean(layers)
        results["type_bottlenecks"][ptype] = {
            "mean_bottleneck_layer": float(mean_layer),
            "bottleneck_layers": layers,
        }
        print(f"  {ptype}: mean bottleneck layer = {mean_layer:.1f} (layers: {layers})")

    return results


# ============================================================================
# Visualization (Enhanced)
# ============================================================================

def create_plots(single_results, pair_results, pruning_results,
                 norm_correlation, patching_results, weight_norms, num_layers):
    """Generate enhanced visualization plots."""
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

    # =========================================================================
    # Plot 1: Single knockout criticality profile (4-panel, enhanced)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    deltas = [single_results["knockouts"][i]["loss_delta"] for i in layers]
    norms_list = [weight_norms[i] for i in layers]

    # Panel 1: Loss delta with gradient coloring
    ax = axes[0, 0]
    norm_deltas = np.array(deltas)
    colors = plt.cm.RdYlBu_r((norm_deltas - norm_deltas.min()) /
                               (norm_deltas.max() - norm_deltas.min() + 1e-10))
    bars = ax.bar(layers, deltas, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=np.mean(deltas), color="red", linestyle="--", alpha=0.5,
               label=f"mean={np.mean(deltas):.3f}")
    # Annotate extremes
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

    # Panel 2: Loss ratio with threshold lines
    ratios = [single_results["knockouts"][i]["loss_ratio"] for i in layers]
    ax = axes[0, 1]
    ax.plot(layers, ratios, "-o", color="#1976D2", markersize=4, linewidth=1.5)
    ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="baseline (1.0x)")
    ax.axhline(y=1.1, color="orange", linestyle="--", alpha=0.5, label="1.1x threshold")
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="2.0x threshold")
    ax.fill_between(layers, 1.0, ratios, where=[r < 1.0 for r in ratios],
                     alpha=0.2, color="green", label="Removal improves loss")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss Ratio (knockout / baseline)")
    ax.set_title("Loss Ratio by Layer")
    ax.legend(fontsize=8, loc="upper left")

    # Panel 3: Weight norm vs criticality with regression line
    ax = axes[1, 0]
    scatter = ax.scatter(norms_list, deltas, c=layers, cmap="viridis", s=60, alpha=0.8,
                         edgecolors="white", linewidth=0.5)
    # Add regression line
    z = np.polyfit(norms_list, deltas, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(norms_list), max(norms_list), 100)
    ax.plot(x_line, p(x_line), "--", color="red", alpha=0.5, linewidth=1.5)
    for i in layers:
        if deltas[i] > np.mean(deltas) + 1.5 * np.std(deltas) or \
           deltas[i] < np.mean(deltas) - np.std(deltas):
            ax.annotate(str(i), (norms_list[i], deltas[i]), fontsize=8,
                        ha="center", va="bottom")
    r = norm_correlation["pearson_r"]
    ax.set_xlabel("Weight Norm (Frobenius)")
    ax.set_ylabel("Loss Delta (criticality)")
    ax.set_title(f"Weight Norm vs Criticality (r={r:.3f})")
    plt.colorbar(scatter, ax=ax, label="Layer index", shrink=0.8)

    # Panel 4: Ranked criticality with regions
    ax = axes[1, 1]
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
    # Plot 2: Pair knockout heatmaps (enhanced with seaborn)
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
    # Plot 3: Greedy pruning curve (enhanced)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    n_removed = list(range(len(pruning_results["losses_after_removal"])))
    losses = pruning_results["losses_after_removal"]
    baseline = pruning_results["baseline_loss"]

    # Left: loss curve
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

    # Right: percentage change
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

    # =========================================================================
    # Plot 4: Activation patching (enhanced)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    type_colors = {"factual": "#1976D2", "arithmetic": "#D32F2F", "linguistic": "#388E3C"}

    # Left: recovery curves
    ax = axes[0]
    type_avg_recovery = defaultdict(lambda: np.zeros(num_layers))
    type_counts = defaultdict(int)
    for pr in patching_results["per_prompt"]:
        recoveries = [lr["recovery_ratio"] for lr in pr["layer_recoveries"]]
        color = type_colors.get(pr["type"], "gray")
        ax.plot(layers, recoveries, "-", alpha=0.3, color=color, linewidth=1)
        type_avg_recovery[pr["type"]] += np.array(recoveries)
        type_counts[pr["type"]] += 1
    for ptype in type_avg_recovery:
        avg = type_avg_recovery[ptype] / type_counts[ptype]
        color = type_colors.get(ptype, "gray")
        ax.plot(layers, avg, "-o", color=color, linewidth=2.5, markersize=4,
                label=f"{ptype} avg (n={type_counts[ptype]})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery Ratio")
    ax.set_title("Activation Patching: Recovery by Layer")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.3, label="Full recovery")
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    # Right: knockout vs patching comparison
    ax = axes[1]
    knockout_deltas = [single_results["knockouts"][i]["loss_delta"] for i in layers]
    avg_recovery = np.zeros(num_layers)
    for pr in patching_results["per_prompt"]:
        for lr in pr["layer_recoveries"]:
            avg_recovery[lr["layer"]] += lr["recovery_ratio"]
    avg_recovery /= len(patching_results["per_prompt"])

    ax2 = ax.twinx()
    ax.bar(layers, knockout_deltas, alpha=0.4, color="#FF7043", label="Knockout Δloss",
           edgecolor="white", linewidth=0.5)
    ax2.plot(layers, avg_recovery, "-o", color="#1976D2", markersize=4, linewidth=2,
             label="Patching recovery")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Knockout Loss Delta", color="#FF7043")
    ax2.set_ylabel("Mean Patching Recovery", color="#1976D2")
    ax.set_title("Knockout vs Patching")
    ax.legend(loc="upper left", fontsize=9)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"Causal Tracing — {model_short}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "activation_patching.png", dpi=150, bbox_inches="tight")
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

    print("\nComputing weight norms...")
    weight_norms = compute_weight_norms(model, num_layers)
    for i in range(num_layers):
        print(f"  Layer {i:2d}: norm={weight_norms[i]:.2f}")

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

    # Part 4: Weight norm correlation
    norm_results = analyze_weight_norm_correlation(single_results, weight_norms,
                                                  num_layers)
    t5 = time.time()

    # Part 5: Activation patching
    patching_results = run_activation_patching(model, tokenizer, num_layers)
    t6 = time.time()
    print(f"Activation patching time: {t6 - t5:.1f}s")

    # Generate plots
    print("\nGenerating plots...")
    create_plots(single_results, pair_results, pruning_results,
                 norm_results, patching_results, weight_norms, num_layers)

    # Save all results
    all_results = {
        "config": {
            "model": MODEL_NAME,
            "num_layers": num_layers,
            "num_calibration_entries": len(calibration_data),
            "calibration_source": str(CALIBRATION_PATH),
            "num_patching_prompts": len(PATCHING_PROMPTS),
            "seed": SEED,
        },
        "single_knockouts": single_results,
        "pair_knockouts": pair_results,
        "pruning": pruning_results,
        "weight_norm_correlation": norm_results,
        "activation_patching": {
            "per_prompt": patching_results["per_prompt"],
            "type_bottlenecks": patching_results.get("type_bottlenecks", {}),
        },
        "weight_norms": weight_norms,
        "timing": {
            "load_s": t_load - t_start,
            "single_knockout_s": t2 - t1,
            "pair_knockout_s": t3 - t2,
            "pruning_s": t4 - t3,
            "patching_s": t6 - t5,
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
