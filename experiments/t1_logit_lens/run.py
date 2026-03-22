"""
Logit Lens / Residual Stream Decoding (T-1)
============================================
Project the residual stream after each layer through the final LM head
to see how token predictions evolve across depth.

Uses pre-generated calibration completions (vLLM, temp=0) so we analyze
the model's own output rather than template tokens. Loss and crystallization
metrics are computed only on completion tokens.

Key questions:
  1. At which layer does the correct next-token first appear in top-k?
  2. Are there layers that *hurt* predictions (correct token drops in rank)?
  3. Do different token types (function words vs content words) crystallize
     at different depths?
"""

import json
import time
import random
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
DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).parent / "results"
CALIBRATION_PATH = Path(__file__).parents[2] / "data" / "text_completions" / "qwen3-4b-instruct-2507" / "completions.json"
TOP_K = 10  # Track top-k predictions at each layer

# Token type classification keywords
FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own", "same",
    "than", "too", "very", "just", "because", "if", "when", "while",
    "that", "which", "who", "whom", "this", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "my", "your",
}

# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classify_token(token_str):
    """Classify a token as function_word, content_word, punctuation, or number."""
    cleaned = token_str.strip().lower()
    if not cleaned:
        return "other"
    if cleaned in FUNCTION_WORDS:
        return "function_word"
    if cleaned.isdigit() or cleaned.replace(".", "", 1).isdigit():
        return "number"
    if all(c in ".,;:!?()[]{}\"'-/\\@#$%^&*+=<>~`|" for c in cleaned):
        return "punctuation"
    return "content_word"


def load_calibration_data():
    """Load pre-generated completions from vLLM."""
    with open(CALIBRATION_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data['completions'])} completions from {CALIBRATION_PATH}")
    print(f"  Total completion tokens: {data['config']['total_completion_tokens']}")
    return data["completions"]


# ============================================================================
# Core: Logit Lens
# ============================================================================

def run_logit_lens(model, tokenizer, num_layers, calibration_data):
    """Run logit lens on all prompts+completions, analyzing completion tokens."""
    lm_head = model.lm_head
    norm = model.model.norm  # Final RMSNorm before LM head

    all_results = []

    for entry in tqdm(calibration_data, desc="Processing prompts"):
        prompt_idx = entry["idx"]
        full_text = entry["full_text"]
        prompt_len = entry["prompt_token_count"]

        tokens = tokenizer(full_text, return_tensors="pt").to(DEVICE)
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]

        # Get hidden states at every layer via hooks
        hidden_states = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states[layer_idx] = output[0].detach()
                else:
                    hidden_states[layer_idx] = output.detach()
            return hook_fn

        hooks = []
        for i in range(num_layers):
            h = model.model.layers[i].register_forward_hook(make_hook(i))
            hooks.append(h)

        with torch.no_grad():
            outputs = model(**tokens, use_cache=False)

        for h in hooks:
            h.remove()

        with torch.no_grad():
            embed_output = model.model.embed_tokens(input_ids)

        # Only analyze completion token positions (skip prompt tokens)
        # Position pos predicts token at pos+1, so we analyze positions
        # where pos+1 >= prompt_len (i.e., the target is a completion token)
        token_results = []
        for pos in range(max(0, prompt_len - 1), seq_len - 1):
            target_id = input_ids[0, pos + 1].item()
            target_str = tokenizer.decode([target_id])
            input_str = tokenizer.decode([input_ids[0, pos].item()])
            token_type = classify_token(target_str)
            is_completion = pos + 1 >= prompt_len

            layer_ranks = []
            layer_probs = []
            layer_top1 = []
            layer_entropies = []

            # Embedding layer
            with torch.no_grad():
                h = embed_output[0, pos].unsqueeze(0).unsqueeze(0)
                h_normed = norm(h)
                logits_at_pos = (lm_head(h_normed))[0, 0].float()
                probs = F.softmax(logits_at_pos, dim=-1)
                rank = (logits_at_pos > logits_at_pos[target_id]).sum().item()
                entropy = -(probs * (probs + 1e-10).log()).sum().item()

            layer_ranks.append(int(rank))
            layer_probs.append(float(probs[target_id].item()))
            layer_top1.append(int(logits_at_pos.argmax().item()))
            layer_entropies.append(float(entropy))

            # Each transformer layer
            for layer_idx in range(num_layers):
                with torch.no_grad():
                    h = hidden_states[layer_idx][0, pos].unsqueeze(0).unsqueeze(0)
                    h_normed = norm(h)
                    logits_at_pos = (lm_head(h_normed))[0, 0].float()
                    probs = F.softmax(logits_at_pos, dim=-1)
                    rank = (logits_at_pos > logits_at_pos[target_id]).sum().item()
                    entropy = -(probs * (probs + 1e-10).log()).sum().item()

                layer_ranks.append(int(rank))
                layer_probs.append(float(probs[target_id].item()))
                layer_top1.append(int(logits_at_pos.argmax().item()))
                layer_entropies.append(float(entropy))

            # Find crystallization layer
            crystal_layer = None
            for l in range(len(layer_ranks)):
                if all(r < TOP_K for r in layer_ranks[l:]):
                    crystal_layer = l - 1
                    break

            first_top1 = None
            for l in range(len(layer_ranks)):
                if layer_ranks[l] == 0:
                    first_top1 = l - 1
                    break

            hurt_layers = []
            for l in range(1, len(layer_ranks)):
                if layer_ranks[l] > layer_ranks[l - 1] and layer_ranks[l - 1] < 100:
                    hurt_layers.append(l - 1)

            token_results.append({
                "position": pos,
                "input_token": input_str,
                "target_token": target_str,
                "target_id": target_id,
                "token_type": token_type,
                "is_completion": is_completion,
                "ranks_by_layer": layer_ranks,
                "probs_by_layer": layer_probs,
                "entropies_by_layer": layer_entropies,
                "crystallization_layer": crystal_layer,
                "first_top1_layer": first_top1,
                "hurt_layers": hurt_layers,
            })

        all_results.append({
            "prompt_idx": prompt_idx,
            "prompt": entry["prompt"],
            "completion": entry["completion"],
            "num_tokens": seq_len,
            "prompt_token_count": prompt_len,
            "tokens": token_results,
        })

        del hidden_states

    return all_results


# ============================================================================
# Analysis
# ============================================================================

def analyze_results(results, num_layers):
    """Compute aggregate statistics — only on completion tokens."""
    num_layer_slots = num_layers + 1

    all_ranks = defaultdict(list)
    all_probs = defaultdict(list)
    all_entropies = defaultdict(list)
    crystal_layers = []
    first_top1_layers = []
    hurt_counts = np.zeros(num_layers)

    type_crystal = defaultdict(list)
    type_ranks = defaultdict(lambda: defaultdict(list))

    total_completion_tokens = 0

    for prompt_result in results:
        for tok in prompt_result["tokens"]:
            if not tok["is_completion"]:
                continue

            total_completion_tokens += 1

            for l in range(num_layer_slots):
                all_ranks[l].append(tok["ranks_by_layer"][l])
                all_probs[l].append(tok["probs_by_layer"][l])
                all_entropies[l].append(tok["entropies_by_layer"][l])

            if tok["crystallization_layer"] is not None:
                crystal_layers.append(tok["crystallization_layer"])
            if tok["first_top1_layer"] is not None:
                first_top1_layers.append(tok["first_top1_layer"])

            for hl in tok["hurt_layers"]:
                if 0 <= hl < num_layers:
                    hurt_counts[hl] += 1

            tt = tok["token_type"]
            if tok["crystallization_layer"] is not None:
                type_crystal[tt].append(tok["crystallization_layer"])
            for l in range(num_layer_slots):
                type_ranks[tt][l].append(tok["ranks_by_layer"][l])

    summary = {
        "total_completion_tokens": total_completion_tokens,
        "per_layer": {},
        "crystallization": {},
        "hurt_layers": {},
        "token_types": {},
    }

    for l in range(num_layer_slots):
        layer_name = "embed" if l == 0 else f"layer_{l-1}"
        ranks = np.array(all_ranks[l])
        probs = np.array(all_probs[l])
        entropies = np.array(all_entropies[l])
        summary["per_layer"][layer_name] = {
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "top1_accuracy": float((ranks == 0).mean()),
            "top5_accuracy": float((ranks < 5).mean()),
            "top10_accuracy": float((ranks < 10).mean()),
            "mean_prob": float(probs.mean()),
            "mean_entropy": float(entropies.mean()),
        }

    if crystal_layers:
        crystal_arr = np.array(crystal_layers)
        summary["crystallization"] = {
            "mean_layer": float(crystal_arr.mean()),
            "median_layer": float(np.median(crystal_arr)),
            "std_layer": float(crystal_arr.std()),
            "histogram": np.histogram(crystal_arr, bins=range(-1, num_layers + 1))[0].tolist(),
        }

    if first_top1_layers:
        top1_arr = np.array(first_top1_layers)
        summary["crystallization"]["first_top1_mean"] = float(top1_arr.mean())
        summary["crystallization"]["first_top1_median"] = float(np.median(top1_arr))

    summary["hurt_layers"] = {
        f"layer_{i}": {
            "count": int(hurt_counts[i]),
            "fraction": float(hurt_counts[i] / total_completion_tokens) if total_completion_tokens > 0 else 0,
        }
        for i in range(num_layers)
    }

    for tt in sorted(type_crystal.keys()):
        arr = np.array(type_crystal[tt])
        per_layer_mean_rank = {}
        for l in range(num_layer_slots):
            layer_name = "embed" if l == 0 else f"layer_{l-1}"
            ranks = np.array(type_ranks[tt][l])
            per_layer_mean_rank[layer_name] = float(ranks.mean())

        summary["token_types"][tt] = {
            "count": len(arr),
            "mean_crystal_layer": float(arr.mean()),
            "median_crystal_layer": float(np.median(arr)),
            "mean_rank_by_layer": per_layer_mean_rank,
        }

    return summary


# ============================================================================
# Visualization (Enhanced)
# ============================================================================

def create_plots(results, summary, num_layers):
    """Generate enhanced visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
        "figure.facecolor": "white", "axes.facecolor": "#fafafa",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    })

    layers = list(range(-1, num_layers))
    model_short = MODEL_NAME.split("/")[-1]
    n_comp = summary["total_completion_tokens"]

    mean_ranks, top1_accs, top5_accs, mean_entropies, mean_probs = [], [], [], [], []
    for l in range(num_layers + 1):
        layer_name = "embed" if l == 0 else f"layer_{l-1}"
        stats = summary["per_layer"][layer_name]
        mean_ranks.append(stats["mean_rank"])
        top1_accs.append(stats["top1_accuracy"])
        top5_accs.append(stats["top5_accuracy"])
        mean_entropies.append(stats["mean_entropy"])
        mean_probs.append(stats["mean_prob"])

    mid = num_layers // 2

    # ---- Plot 1: 4-panel overview ----
    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(2, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    c1, c2 = "#2196F3", "#FF9800"
    ax1.fill_between(layers, mean_ranks, alpha=0.15, color=c1)
    ax1.plot(layers, mean_ranks, "-o", color=c1, markersize=3, linewidth=1.5, label="Mean rank")
    ax1.set_ylabel("Mean Rank (log)", color=c1)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1t = ax1.twinx()
    ax1t.plot(layers, mean_probs, "-s", color=c2, markersize=3, linewidth=1.5, alpha=0.8, label="Mean prob")
    ax1t.set_ylabel("Mean Probability", color=c2)
    ax1t.tick_params(axis="y", labelcolor=c2)
    ax1.axvline(x=mid, color="red", linestyle="--", alpha=0.4)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")
    ax1.set_title("Prediction Rank & Probability")
    ax1.set_xlabel("Layer")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(layers, top5_accs, alpha=0.2, color="#4CAF50")
    ax2.plot(layers, top5_accs, "-^", color="#4CAF50", markersize=3, linewidth=1.5, label="Top-5")
    ax2.fill_between(layers, top1_accs, alpha=0.2, color="#1B5E20")
    ax2.plot(layers, top1_accs, "-o", color="#1B5E20", markersize=3, linewidth=1.5, label="Top-1")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Prediction Accuracy")
    ax2.axvline(x=mid, color="red", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.02, 1.02)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(layers, mean_entropies, alpha=0.2, color="#E53935")
    ax3.plot(layers, mean_entropies, "-o", color="#E53935", markersize=3, linewidth=1.5)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Mean Entropy (nats)")
    ax3.set_title("Prediction Entropy")
    ax3.axvline(x=mid, color="red", linestyle="--", alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 1])
    hurt_fracs = [summary["hurt_layers"][f"layer_{i}"]["fraction"] for i in range(num_layers)]
    bars = ax4.bar(range(num_layers), hurt_fracs, color="#FF7043", alpha=0.8, edgecolor="white", linewidth=0.5)
    top5_hurt = sorted(range(num_layers), key=lambda i: hurt_fracs[i], reverse=True)[:5]
    for idx in top5_hurt:
        bars[idx].set_color("#D32F2F")
    for idx in top5_hurt[:3]:
        ax4.annotate(f"L{idx}\n{hurt_fracs[idx]:.1%}", (idx, hurt_fracs[idx]),
                     ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Fraction Hurt")
    ax4.set_title("Layers That Hurt Predictions")

    fig.suptitle(f"Logit Lens (completion tokens, n={n_comp}) — {model_short}",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(RESULTS_DIR / "logit_lens_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: Token type crystallization ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    type_colors = {
        "function_word": "#1976D2", "content_word": "#388E3C",
        "number": "#D32F2F", "punctuation": "#757575",
    }
    type_markers = {"function_word": "o", "content_word": "s", "number": "D", "punctuation": "^"}

    ax = axes[0]
    for tt in sorted(summary["token_types"].keys()):
        if tt == "other":
            continue
        ranks_by_layer = summary["token_types"][tt]["mean_rank_by_layer"]
        y = [ranks_by_layer.get("embed" if l == 0 else f"layer_{l-1}", 0) for l in range(num_layers + 1)]
        ax.plot(layers, y, f"-{type_markers.get(tt, 'o')}", markersize=4, linewidth=1.5,
                color=type_colors.get(tt, "purple"),
                label=f"{tt} (n={summary['token_types'][tt]['count']})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Rank")
    ax.set_title("Mean Rank by Token Type")
    ax.set_yscale("log")
    ax.legend(fontsize=9)

    ax = axes[1]
    type_data, type_labels_list, type_cols = [], [], []
    for tt in ["punctuation", "function_word", "content_word", "number"]:
        if tt in summary["token_types"]:
            info = summary["token_types"][tt]
            type_labels_list.append(f"{tt}\n(n={info['count']})")
            type_cols.append(type_colors.get(tt, "gray"))
            type_data.append(info["mean_crystal_layer"])
    ax.barh(range(len(type_data)), type_data, color=type_cols, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(type_labels_list)))
    ax.set_yticklabels(type_labels_list, fontsize=10)
    ax.set_xlabel("Mean Crystallization Layer")
    ax.set_title("Crystallization Depth by Token Type")
    for i, v in enumerate(type_data):
        ax.text(v + 0.3, i, f"L{v:.1f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, num_layers)

    fig.suptitle(f"Token Type Analysis (completion tokens) — {model_short}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "token_type_crystallization.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 3: Per-prompt heatmap ----
    prompt_labels = []
    rank_matrix = []
    for pr in results:
        short_label = pr["prompt"][:45] + ("..." if len(pr["prompt"]) > 45 else "")
        prompt_labels.append(short_label)
        completion_toks = [t for t in pr["tokens"] if t["is_completion"]]
        if not completion_toks:
            avg_ranks = [0] * (num_layers + 1)
        else:
            avg_ranks = [np.mean([t["ranks_by_layer"][l] for t in completion_toks]) for l in range(num_layers + 1)]
        rank_matrix.append(avg_ranks)
    rank_matrix = np.array(rank_matrix)

    fig, ax = plt.subplots(figsize=(18, 12))
    im = ax.imshow(np.log1p(rank_matrix), aspect="auto", cmap="magma_r", interpolation="nearest")
    ax.set_xticks(range(0, num_layers + 1, 2))
    ax.set_xticklabels(["emb"] + [str(i) for i in range(1, num_layers, 2)])
    ax.set_yticks(range(len(prompt_labels)))
    ax.set_yticklabels(prompt_labels, fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_title(f"log(1 + Mean Rank) per Prompt (completion tokens) — {model_short}",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="log(1 + rank)", shrink=0.8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "prompt_rank_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Plot 4: Accuracy gains ----
    fig, ax = plt.subplots(figsize=(14, 5))
    accs = [summary["per_layer"]["embed"]["top1_accuracy"]]
    accs += [summary["per_layer"][f"layer_{i}"]["top1_accuracy"] for i in range(num_layers)]
    gains = [accs[i] - accs[i - 1] for i in range(1, len(accs))]
    colors = ["#4CAF50" if g > 0 else "#E53935" for g in gains]
    ax.bar(range(num_layers), gains, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 Accuracy Change")
    ax.set_title(f"Layer-over-Layer Accuracy Gains (completion tokens) — {model_short}",
                 fontsize=13, fontweight="bold")
    gain_order = sorted(range(num_layers), key=lambda i: gains[i], reverse=True)
    for idx in gain_order[:3]:
        ax.annotate(f"L{idx}\n+{gains[idx]:.3f}", (idx, gains[idx]),
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1B5E20")
    for idx in gain_order[-3:]:
        ax.annotate(f"L{idx}\n{gains[idx]:.3f}", (idx, gains[idx]),
                    ha="center", va="top", fontsize=8, fontweight="bold", color="#B71C1C")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_gains.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {RESULTS_DIR}/")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    calibration_data = load_calibration_data()

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} decoder layers on {DEVICE}")
    t_load = time.time()
    print(f"Load time: {t_load - t_start:.1f}s")

    print(f"\nRunning logit lens on {len(calibration_data)} prompts+completions...")
    results = run_logit_lens(model, tokenizer, num_layers, calibration_data)
    t_lens = time.time()
    print(f"Logit lens time: {t_lens - t_load:.1f}s")

    print("\nAnalyzing results (completion tokens only)...")
    summary = analyze_results(results, num_layers)
    t_analyze = time.time()

    # Print key findings
    print("\n" + "=" * 70)
    print(f"KEY FINDINGS (n={summary['total_completion_tokens']} completion tokens)")
    print("=" * 70)

    print("\n--- Per-Layer Top-1 Accuracy ---")
    for l in range(num_layers + 1):
        layer_name = "embed" if l == 0 else f"layer_{l-1}"
        stats = summary["per_layer"][layer_name]
        bar = "█" * int(stats["top1_accuracy"] * 50)
        print(f"  {layer_name:>10s}: {stats['top1_accuracy']:.3f} {bar}")

    if summary["crystallization"]:
        print(f"\n--- Crystallization (top-{TOP_K} stable) ---")
        print(f"  Mean layer: {summary['crystallization']['mean_layer']:.1f}")
        print(f"  Median layer: {summary['crystallization']['median_layer']:.1f}")
        if "first_top1_mean" in summary["crystallization"]:
            print(f"  First top-1 mean: {summary['crystallization']['first_top1_mean']:.1f}")

    print(f"\n--- Token Type Crystallization ---")
    for tt in sorted(summary["token_types"].keys()):
        if tt == "other":
            continue
        info = summary["token_types"][tt]
        print(f"  {tt:>15s}: mean crystal layer = {info['mean_crystal_layer']:.1f} "
              f"(n={info['count']})")

    print(f"\n--- Top 5 Hurt Layers ---")
    hurt_sorted = sorted(summary["hurt_layers"].items(), key=lambda x: x[1]["fraction"], reverse=True)[:5]
    for name, info in hurt_sorted:
        print(f"  {name}: {info['fraction']:.3f} ({info['count']} tokens)")

    mid = num_layers // 2
    print(f"\n--- Plateau Analysis (layers 0-{mid-1} vs {mid}-{num_layers-1}) ---")
    early_top1 = [summary["per_layer"][f"layer_{i}"]["top1_accuracy"] for i in range(mid)]
    late_top1 = [summary["per_layer"][f"layer_{i}"]["top1_accuracy"] for i in range(mid, num_layers)]
    early_entropy = [summary["per_layer"][f"layer_{i}"]["mean_entropy"] for i in range(mid)]
    late_entropy = [summary["per_layer"][f"layer_{i}"]["mean_entropy"] for i in range(mid, num_layers)]
    print(f"  Early avg top-1: {np.mean(early_top1):.3f} (std={np.std(early_top1):.3f})")
    print(f"  Late avg top-1: {np.mean(late_top1):.3f} (std={np.std(late_top1):.3f})")
    print(f"  Early avg entropy: {np.mean(early_entropy):.1f} (std={np.std(early_entropy):.1f})")
    print(f"  Late avg entropy: {np.mean(late_entropy):.1f} (std={np.std(late_entropy):.1f})")

    print("\nGenerating plots...")
    create_plots(results, summary, num_layers)

    save_data = {
        "config": {
            "model": MODEL_NAME,
            "num_layers": num_layers,
            "num_prompts": len(calibration_data),
            "total_completion_tokens": summary["total_completion_tokens"],
            "top_k": TOP_K,
            "seed": SEED,
            "calibration_source": str(CALIBRATION_PATH),
        },
        "summary": summary,
        "timing": {
            "load_s": t_load - t_start,
            "lens_s": t_lens - t_load,
            "analyze_s": t_analyze - t_lens,
            "total_s": time.time() - t_start,
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    with open(RESULTS_DIR / "full_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
