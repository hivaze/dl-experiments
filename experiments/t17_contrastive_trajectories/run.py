"""
Contrastive Completion Trajectories (T-17)
==========================================
For each prompt, force-decode semantically related completions (synonyms,
antonyms, style variants, unrelated) and track how their hidden-state
representations diverge or converge layer-by-layer.

Extends T-1 (logit lens) and T-4 (residual stream geometry) to contrastive
settings: instead of analyzing a single completion, we compare *pairs* of
completions and measure when the model's representations distinguish them.

Key questions:
  1. At which layer do antonym completions diverge in hidden-state space?
  2. Do synonym completions maintain similar trajectories despite different surface forms?
  3. Does meaning crystallize before form?
  4. How does the divergence profile differ across relationship types?
"""

import argparse
import json
import time
import random
from pathlib import Path
from itertools import combinations
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
DEFAULT_DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).parent / "results"
DATA_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "text_completions"
    / "contrastive_pairs.json"
)

# ============================================================================
# Utilities
# ============================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_contrastive_data():
    """Load hand-crafted contrastive pairs."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    groups = data["groups"]
    system_message = data["system_message"]
    counts = defaultdict(int)
    for g in groups:
        counts[g["relationship"]] += 1
    print(f"Loaded {len(groups)} contrastive groups from {DATA_PATH}")
    for rel, cnt in sorted(counts.items()):
        print(f"  {rel}: {cnt}")
    return groups, system_message


def build_templated_input(tokenizer, system_message, prompt, completion):
    """Build chat-templated input with forced completion using apply_chat_template."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    # Get the prompt portion (up to assistant turn)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    # Full text = prompt + forced completion
    full_text = prompt_text + completion
    return prompt_text, full_text


def cosine_similarity(a, b):
    """Cosine similarity between two vectors (float32)."""
    a = a.float()
    b = b.float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def linear_cka(X, Y):
    """Linear CKA between two matrices X (n×d1) and Y (n×d2).

    Measures representational similarity independent of dimensionality.
    """
    X = X.float()
    Y = Y.float()
    # Center
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    hsic_xy = (X @ X.T * (Y @ Y.T)).sum()
    hsic_xx = (X @ X.T * (X @ X.T)).sum()
    hsic_yy = (Y @ Y.T * (Y @ Y.T)).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0
    return (hsic_xy / denom).item()


# ============================================================================
# Core: Hidden State Extraction
# ============================================================================


def extract_hidden_states(model, tokenizer, full_text, num_layers, device):
    """Run forward pass and extract hidden states at every layer.

    Returns:
        hidden_states: dict[int, Tensor] — layer_idx -> (seq_len, hidden_dim)
        embed_output: Tensor — (seq_len, hidden_dim)
        input_ids: Tensor — (seq_len,)
    """
    tokens = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]

    hidden_states = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_states[layer_idx] = output[0][0].detach()  # (seq_len, hidden_dim)
            else:
                hidden_states[layer_idx] = output[0].detach()
        return hook_fn

    hooks = []
    for i in range(num_layers):
        h = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(**tokens, use_cache=False)

    for h in hooks:
        h.remove()

    with torch.no_grad():
        embed_output = model.model.embed_tokens(input_ids)[0].detach()

    return hidden_states, embed_output, input_ids[0]


# ============================================================================
# Core: Contrastive Analysis
# ============================================================================


def analyze_pair(
    hidden_states_a, embed_a, ids_a,
    hidden_states_b, embed_b, ids_b,
    model, num_layers,
    prompt_len_a, prompt_len_b,
):
    """Compare hidden states between two completions at every layer.

    Returns per-layer metrics dict.
    """
    lm_head = model.lm_head
    norm = model.model.norm

    # Find shared prefix length in token IDs (first position where they differ)
    min_len = min(len(ids_a), len(ids_b))
    shared_prefix = min_len  # assume all shared unless we find a mismatch
    for i in range(min_len):
        if ids_a[i].item() != ids_b[i].item():
            shared_prefix = i
            break

    # Use the last shared token position and the first diverging position
    last_shared_pos = max(0, shared_prefix - 1)
    first_div_pos_a = min(shared_prefix, len(ids_a) - 1)
    first_div_pos_b = min(shared_prefix, len(ids_b) - 1)

    # Prompt token count (use the larger of the two, they should be equal
    # since same prompt template is used)
    prompt_len = max(prompt_len_a, prompt_len_b)

    layer_metrics = []

    # Process embedding layer + transformer layers
    for layer_idx in range(-1, num_layers):  # -1 = embedding
        if layer_idx == -1:
            h_a = embed_a
            h_b = embed_b
        else:
            h_a = hidden_states_a[layer_idx]
            h_b = hidden_states_b[layer_idx]

        # --- Cosine similarity at last shared position ---
        cos_shared = cosine_similarity(
            h_a[last_shared_pos], h_b[last_shared_pos]
        )

        # --- Cosine similarity at first diverging position ---
        cos_div = cosine_similarity(
            h_a[first_div_pos_a], h_b[first_div_pos_b]
        )

        # --- Mean cosine similarity over shared prefix ---
        if shared_prefix > 0:
            cos_prefix_vals = []
            for pos in range(shared_prefix):
                cos_prefix_vals.append(
                    cosine_similarity(h_a[pos], h_b[pos])
                )
            cos_prefix_mean = float(np.mean(cos_prefix_vals))
        else:
            cos_prefix_mean = cos_div

        # --- L2 distance at diverging position (normalized) ---
        h_a_div = h_a[first_div_pos_a].float()
        h_b_div = h_b[first_div_pos_b].float()
        l2_dist = torch.norm(h_a_div - h_b_div).item()
        mean_norm = (torch.norm(h_a_div).item() + torch.norm(h_b_div).item()) / 2
        l2_normalized = l2_dist / (mean_norm + 1e-10)

        # --- CKA over completion tokens ---
        # Use completion portion of each sequence
        comp_start_a = max(prompt_len_a, 0)
        comp_start_b = max(prompt_len_b, 0)
        comp_a = h_a[comp_start_a:]
        comp_b = h_b[comp_start_b:]
        # Truncate to same length for CKA
        comp_len = min(len(comp_a), len(comp_b))
        if comp_len >= 2:
            cka_val = linear_cka(comp_a[:comp_len], comp_b[:comp_len])
        else:
            cka_val = cos_div  # Fall back to pointwise cosine

        # --- Logit-level analysis at diverging position ---
        kl_div = None
        if layer_idx >= 0:
            with torch.no_grad():
                ha = h_a[first_div_pos_a].unsqueeze(0).unsqueeze(0)
                hb = h_b[first_div_pos_b].unsqueeze(0).unsqueeze(0)
                logits_a = lm_head(norm(ha))[0, 0].float()
                logits_b = lm_head(norm(hb))[0, 0].float()
                log_probs_a = F.log_softmax(logits_a, dim=-1)
                probs_b = F.softmax(logits_b, dim=-1)
                kl_div = F.kl_div(log_probs_a, probs_b, reduction="sum").item()

        layer_metrics.append({
            "layer": layer_idx,
            "cosine_shared": cos_shared,
            "cosine_diverging": cos_div,
            "cosine_prefix_mean": cos_prefix_mean,
            "l2_normalized": l2_normalized,
            "cka": cka_val,
            "kl_divergence": kl_div,
        })

    return {
        "shared_prefix_len": shared_prefix,
        "first_div_pos_a": first_div_pos_a,
        "first_div_pos_b": first_div_pos_b,
        "layer_metrics": layer_metrics,
    }


def analyze_pivot_token(
    hidden_states_a, hidden_states_b,
    embed_a, embed_b,
    ids_a, ids_b,
    num_layers, shared_prefix,
):
    """Track the pivot token's representation across layers.

    The pivot token is the first token where the two completions differ.
    Returns cosine similarity of the pivot token at each layer.
    """
    if shared_prefix >= min(len(ids_a), len(ids_b)):
        return None

    pivot_pos_a = shared_prefix
    pivot_pos_b = shared_prefix

    # Also compute similarity for a non-pivot (shared prefix) token as baseline
    baseline_pos = max(0, shared_prefix - 1)

    pivot_sims = []
    baseline_sims = []

    for layer_idx in range(-1, num_layers):
        if layer_idx == -1:
            h_a, h_b = embed_a, embed_b
        else:
            h_a = hidden_states_a[layer_idx]
            h_b = hidden_states_b[layer_idx]

        pivot_sim = cosine_similarity(h_a[pivot_pos_a], h_b[pivot_pos_b])
        base_sim = cosine_similarity(h_a[baseline_pos], h_b[baseline_pos])
        pivot_sims.append({"layer": layer_idx, "cosine": pivot_sim})
        baseline_sims.append({"layer": layer_idx, "cosine": base_sim})

    return {
        "pivot_pos_a": pivot_pos_a,
        "pivot_pos_b": pivot_pos_b,
        "pivot_token_a": ids_a[pivot_pos_a].item(),
        "pivot_token_b": ids_b[pivot_pos_b].item(),
        "pivot_similarities": pivot_sims,
        "baseline_similarities": baseline_sims,
    }


# ============================================================================
# Visualization
# ============================================================================


def plot_divergence_curves(summary, results_dir):
    """Plot cosine similarity vs. layer depth, one curve per relationship type."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Colors for relationship types
    colors = {
        "synonym": "#2ecc71",
        "antonym": "#e74c3c",
        "style": "#3498db",
        "unrelated": "#95a5a6",
    }

    # --- Panel 1: Cosine similarity at diverging position ---
    ax = axes[0, 0]
    for rel, stats in summary.items():
        layers = [s["layer"] for s in stats]
        means = [s["cosine_diverging_mean"] for s in stats]
        stds = [s["cosine_diverging_std"] for s in stats]
        means = np.array(means)
        stds = np.array(stds)
        ax.plot(layers, means, color=colors.get(rel, "gray"), label=rel, linewidth=2)
        ax.fill_between(layers, means - stds, means + stds,
                        color=colors.get(rel, "gray"), alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Hidden-State Similarity at Diverging Token Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: KL divergence ---
    ax = axes[0, 1]
    for rel, stats in summary.items():
        layers = [s["layer"] for s in stats if s["kl_divergence_mean"] is not None]
        kls = [s["kl_divergence_mean"] for s in stats if s["kl_divergence_mean"] is not None]
        if layers:
            ax.plot(layers, kls, color=colors.get(rel, "gray"), label=rel, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Logit Distribution KL Divergence Between Completions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # --- Panel 3: L2 distance (normalized) ---
    ax = axes[1, 0]
    for rel, stats in summary.items():
        layers = [s["layer"] for s in stats]
        means = [s["l2_normalized_mean"] for s in stats]
        ax.plot(layers, means, color=colors.get(rel, "gray"), label=rel, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized L2 Distance")
    ax.set_title("Normalized L2 Distance at Diverging Position")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 4: CKA ---
    ax = axes[1, 1]
    for rel, stats in summary.items():
        layers = [s["layer"] for s in stats]
        means = [s["cka_mean"] for s in stats]
        ax.plot(layers, means, color=colors.get(rel, "gray"), label=rel, linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear CKA")
    ax.set_title("Completion-Level CKA Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "T-17: Contrastive Completion Trajectories\n"
        "How do synonym/antonym/style/unrelated completions diverge across depth?",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "divergence_curves.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved divergence_curves.png")


def plot_pivot_trajectories(pivot_data, results_dir):
    """Plot pivot token cosine similarity across layers."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        "synonym": "#2ecc71",
        "antonym": "#e74c3c",
        "style": "#3498db",
        "unrelated": "#95a5a6",
    }

    # Left panel: pivot token similarity
    ax = axes[0]
    for rel, entries in pivot_data.items():
        if not entries:
            continue
        all_layers = None
        all_sims = []
        for entry in entries:
            sims = [s["cosine"] for s in entry["pivot_similarities"]]
            layers = [s["layer"] for s in entry["pivot_similarities"]]
            all_sims.append(sims)
            all_layers = layers
        if all_layers is None:
            continue
        all_sims = np.array(all_sims)
        mean = all_sims.mean(axis=0)
        std = all_sims.std(axis=0)
        ax.plot(all_layers, mean, color=colors.get(rel, "gray"),
                label=rel, linewidth=2)
        ax.fill_between(all_layers, mean - std, mean + std,
                        color=colors.get(rel, "gray"), alpha=0.15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Pivot Token Similarity Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right panel: per-pair pivot trajectories for antonyms (individual traces)
    ax = axes[1]
    if "antonym" in pivot_data:
        entries = pivot_data["antonym"]
        for i, entry in enumerate(entries):
            layers = [s["layer"] for s in entry["pivot_similarities"]]
            sims = [s["cosine"] for s in entry["pivot_similarities"]]
            alpha = 0.4
            ax.plot(layers, sims, color="#e74c3c", alpha=alpha, linewidth=0.8)
        # Add mean on top
        all_sims = np.array([[s["cosine"] for s in e["pivot_similarities"]] for e in entries])
        mean = all_sims.mean(axis=0)
        ax.plot(layers, mean, color="#e74c3c", linewidth=2.5,
                label=f"antonym mean (n={len(entries)})")
    if "synonym" in pivot_data:
        entries = pivot_data["synonym"]
        for i, entry in enumerate(entries):
            layers = [s["layer"] for s in entry["pivot_similarities"]]
            sims = [s["cosine"] for s in entry["pivot_similarities"]]
            ax.plot(layers, sims, color="#2ecc71", alpha=0.3, linewidth=0.8)
        all_sims = np.array([[s["cosine"] for s in e["pivot_similarities"]] for e in entries])
        mean = all_sims.mean(axis=0)
        ax.plot(layers, mean, color="#2ecc71", linewidth=2.5,
                label=f"synonym mean (n={len(entries)})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Individual Pivot Token Traces (Antonym + Synonym)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "T-17: Pivot Token Analysis\n"
        "How does the representation of the first differing token evolve?",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "pivot_token_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved pivot_token_trajectories.png")


def plot_relationship_heatmap(all_results, results_dir):
    """Plot groups × layers heatmap of cosine similarity, grouped by relationship."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    # Sort groups by relationship type
    rel_order = ["synonym", "antonym", "style", "unrelated"]
    sorted_results = sorted(
        all_results,
        key=lambda r: (
            rel_order.index(r["relationship"])
            if r["relationship"] in rel_order else 99,
            r["group_id"],
        ),
    )

    # Build matrix: rows = group-pair combos, cols = layers
    rows = []
    labels = []
    rel_boundaries = []
    current_rel = None
    for result in sorted_results:
        for pair in result["pairs"]:
            layer_cos = [m["cosine_diverging"] for m in pair["layer_metrics"]]
            rows.append(layer_cos)
            labels.append(f"{result['relationship']}_{result['group_id']}")
            if result["relationship"] != current_rel:
                rel_boundaries.append(len(rows) - 1)
                current_rel = result["relationship"]

    if not rows:
        return

    matrix = np.array(rows)
    layers = [m["layer"] for m in sorted_results[0]["pairs"][0]["layer_metrics"]]

    fig, ax = plt.subplots(figsize=(16, max(8, len(rows) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=1.0)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Group-Pair", fontsize=12)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)

    # Draw relationship boundaries and labels
    rel_label_positions = []
    prev_b = 0
    for idx, b in enumerate(rel_boundaries[1:] + [len(rows)]):
        mid = (prev_b + b) / 2
        rel_label_positions.append((mid, rel_order[idx] if idx < len(rel_order) else ""))
        prev_b = b
    # First section
    if rel_boundaries:
        mid = (rel_boundaries[0] + (rel_boundaries[1] if len(rel_boundaries) > 1 else len(rows))) / 2

    for b in rel_boundaries[1:]:
        ax.axhline(y=b - 0.5, color="white", linewidth=3)
        ax.axhline(y=b - 0.5, color="black", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title(
        "T-17: Cosine Similarity Heatmap (Groups × Layers)\n"
        "Grouped by relationship type (synonym | antonym | style | unrelated)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "relationship_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved relationship_heatmap.png")


def plot_kl_crossover(summary, results_dir):
    """Plot KL divergence showing the synonym/antonym crossover."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    colors = {
        "synonym": "#2ecc71",
        "antonym": "#e74c3c",
        "style": "#3498db",
        "unrelated": "#95a5a6",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: KL divergence (linear scale for detail)
    ax = axes[0]
    for rel in ["synonym", "antonym", "style", "unrelated"]:
        if rel not in summary:
            continue
        stats = summary[rel]
        layers = [s["layer"] for s in stats if s["kl_divergence_mean"] is not None]
        kl_means = [s["kl_divergence_mean"] for s in stats if s["kl_divergence_mean"] is not None]
        kl_stds = [s["kl_divergence_std"] for s in stats if s["kl_divergence_std"] is not None]
        kl_means = np.array(kl_means)
        kl_stds = np.array(kl_stds)
        ax.plot(layers, kl_means, color=colors[rel], label=rel, linewidth=2)
        ax.fill_between(layers, np.maximum(kl_means - kl_stds, 0), kl_means + kl_stds,
                        color=colors[rel], alpha=0.12)

    # Mark crossover points
    if "synonym" in summary and "antonym" in summary:
        syn_kl = {s["layer"]: s["kl_divergence_mean"] for s in summary["synonym"]
                  if s["kl_divergence_mean"] is not None}
        ant_kl = {s["layer"]: s["kl_divergence_mean"] for s in summary["antonym"]
                  if s["kl_divergence_mean"] is not None}
        common_layers = sorted(set(syn_kl) & set(ant_kl))
        prev_diff = None
        for layer in common_layers:
            diff = syn_kl[layer] - ant_kl[layer]
            if prev_diff is not None and prev_diff * diff < 0:
                ax.axvline(x=layer, color="purple", linestyle="--", alpha=0.6,
                           linewidth=1.5, label=f"crossover ~layer {layer}")
            prev_diff = diff

    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence: Synonym-Antonym Crossover")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right panel: synonym minus antonym KL (delta)
    ax = axes[1]
    if "synonym" in summary and "antonym" in summary:
        syn_kl = {s["layer"]: s["kl_divergence_mean"] for s in summary["synonym"]
                  if s["kl_divergence_mean"] is not None}
        ant_kl = {s["layer"]: s["kl_divergence_mean"] for s in summary["antonym"]
                  if s["kl_divergence_mean"] is not None}
        common_layers = sorted(set(syn_kl) & set(ant_kl))
        deltas = [syn_kl[l] - ant_kl[l] for l in common_layers]
        ax.bar(common_layers, deltas,
               color=["#e74c3c" if d < 0 else "#2ecc71" for d in deltas],
               alpha=0.7, width=0.8)
        ax.axhline(y=0, color="black", linewidth=1)
        ax.set_xlabel("Layer")
        ax.set_ylabel("KL(synonym) − KL(antonym)")
        ax.set_title("Synonym vs Antonym KL Delta\n(green = synonyms MORE divergent)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "T-17: KL Divergence — Synonym vs Antonym\n"
        "All pairs (prefix structure not controlled)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "kl_crossover.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved kl_crossover.png")


def plot_prefix_controlled(prefix_summary, results_dir):
    """Plot prefix-controlled KL comparison: immediate-antonym vs immediate-synonym
    and shared-antonym vs shared-synonym."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    colors = {
        "synonym_immediate": "#2ecc71",
        "antonym_immediate": "#e74c3c",
        "synonym_shared": "#27ae60",
        "antonym_shared": "#c0392b",
    }
    linestyles = {
        "synonym_immediate": "-",
        "antonym_immediate": "-",
        "synonym_shared": "--",
        "antonym_shared": "--",
    }
    labels = {
        "synonym_immediate": "synonym (immediate divergence)",
        "antonym_immediate": "antonym (immediate divergence)",
        "synonym_shared": "synonym (shared prefix)",
        "antonym_shared": "antonym (shared prefix)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- Panel 1: All four conditions KL ---
    ax = axes[0]
    for key in ["antonym_immediate", "synonym_immediate", "antonym_shared", "synonym_shared"]:
        if key not in prefix_summary:
            continue
        stats = prefix_summary[key]
        layers = [s["layer"] for s in stats if s["kl_divergence_mean"] is not None]
        kls = [s["kl_divergence_mean"] for s in stats if s["kl_divergence_mean"] is not None]
        if layers:
            ax.plot(layers, kls, color=colors[key], linestyle=linestyles[key],
                    label=labels[key], linewidth=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence by Relationship × Prefix Group")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Immediate divergence only (prefix-controlled comparison) ---
    ax = axes[1]
    for key in ["antonym_immediate", "synonym_immediate"]:
        if key not in prefix_summary:
            continue
        stats = prefix_summary[key]
        layers = [s["layer"] for s in stats if s["kl_divergence_mean"] is not None]
        kls = [s["kl_divergence_mean"] for s in stats if s["kl_divergence_mean"] is not None]
        kl_stds = [s["kl_divergence_std"] for s in stats if s["kl_divergence_std"] is not None]
        if layers:
            kls_arr = np.array(kls)
            stds_arr = np.array(kl_stds)
            ax.plot(layers, kls_arr, color=colors[key], label=labels[key], linewidth=2)
            ax.fill_between(layers, np.maximum(kls_arr - stds_arr, 0), kls_arr + stds_arr,
                            color=colors[key], alpha=0.12)

    # Mark crossover points
    if "synonym_immediate" in prefix_summary and "antonym_immediate" in prefix_summary:
        syn_kl = {s["layer"]: s["kl_divergence_mean"] for s in prefix_summary["synonym_immediate"]
                  if s["kl_divergence_mean"] is not None}
        ant_kl = {s["layer"]: s["kl_divergence_mean"] for s in prefix_summary["antonym_immediate"]
                  if s["kl_divergence_mean"] is not None}
        common_layers = sorted(set(syn_kl) & set(ant_kl))
        prev_diff = None
        for layer in common_layers:
            diff = syn_kl[layer] - ant_kl[layer]
            if prev_diff is not None and prev_diff * diff < 0:
                ax.axvline(x=layer, color="purple", linestyle="--", alpha=0.6,
                           linewidth=1.5, label=f"crossover ~L{layer}")
            prev_diff = diff

    ax.set_xlabel("Layer")
    ax.set_ylabel("KL Divergence")
    ax.set_title("Prefix-Controlled: Immediate Divergence Only\n(fair comparison — both diverge at token 1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Delta bar chart for immediate-divergence pairs ---
    ax = axes[2]
    if "synonym_immediate" in prefix_summary and "antonym_immediate" in prefix_summary:
        syn_kl = {s["layer"]: s["kl_divergence_mean"] for s in prefix_summary["synonym_immediate"]
                  if s["kl_divergence_mean"] is not None}
        ant_kl = {s["layer"]: s["kl_divergence_mean"] for s in prefix_summary["antonym_immediate"]
                  if s["kl_divergence_mean"] is not None}
        common_layers = sorted(set(syn_kl) & set(ant_kl))
        deltas = [syn_kl[l] - ant_kl[l] for l in common_layers]
        ax.bar(common_layers, deltas,
               color=["#e74c3c" if d < 0 else "#2ecc71" for d in deltas],
               alpha=0.7, width=0.8)
        ax.axhline(y=0, color="black", linewidth=1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL(synonym) − KL(antonym)")
    ax.set_title("Prefix-Controlled Delta\n(green = synonyms more divergent, red = antonyms)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "T-17: Prefix-Controlled Meaning-vs-Form Test\n"
        "Comparing synonym vs antonym pairs with matched prefix structure",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(results_dir / "prefix_controlled.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved prefix_controlled.png")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="T-17: Contrastive Completion Trajectories")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="CUDA device")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    args = parser.parse_args()

    device = args.device
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    t_start = time.time()

    # ---- Load data ----
    print("=" * 70)
    print("T-17: Contrastive Completion Trajectories")
    print("=" * 70)
    groups, system_message = load_contrastive_data()

    # ---- Load model ----
    print(f"\nLoading model: {MODEL_NAME}")
    t_model = time.time()
    torch.set_default_dtype(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"  Model loaded in {time.time() - t_model:.1f}s")
    print(f"  Layers: {num_layers}")

    # ---- Process each group ----
    print(f"\nProcessing {len(groups)} contrastive groups...")
    t_process = time.time()

    all_results = []
    pivot_data = defaultdict(list)  # relationship -> list of pivot analyses

    for group in tqdm(groups, desc="Groups"):
        group_id = group["id"]
        prompt = group["prompt"]
        relationship = group["relationship"]
        completions = group["completions"]

        # Extract hidden states for each completion
        completion_data = []
        for comp in completions:
            prompt_text, full_text = build_templated_input(
                tokenizer, system_message, prompt, comp["text"]
            )
            prompt_tokens = tokenizer(prompt_text, return_tensors="pt")
            prompt_len = prompt_tokens["input_ids"].shape[1]

            hidden_states, embed_output, input_ids = extract_hidden_states(
                model, tokenizer, full_text, num_layers, device
            )
            completion_data.append({
                "label": comp["label"],
                "text": comp["text"],
                "hidden_states": hidden_states,
                "embed_output": embed_output,
                "input_ids": input_ids,
                "prompt_len": prompt_len,
            })

        # Compare all pairs
        pair_results = []
        for (i, ca), (j, cb) in combinations(enumerate(completion_data), 2):
            pair_metrics = analyze_pair(
                ca["hidden_states"], ca["embed_output"], ca["input_ids"],
                cb["hidden_states"], cb["embed_output"], cb["input_ids"],
                model, num_layers,
                ca["prompt_len"], cb["prompt_len"],
            )
            pair_metrics["label_a"] = ca["label"]
            pair_metrics["label_b"] = cb["label"]
            pair_results.append(pair_metrics)

            # Pivot token analysis
            pivot = analyze_pivot_token(
                ca["hidden_states"], cb["hidden_states"],
                ca["embed_output"], cb["embed_output"],
                ca["input_ids"], cb["input_ids"],
                num_layers, pair_metrics["shared_prefix_len"],
            )
            if pivot is not None:
                pivot_data[relationship].append(pivot)

        prefix_group = group.get("prefix_group", "unknown")
        all_results.append({
            "group_id": group_id,
            "prompt": prompt,
            "relationship": relationship,
            "prefix_group": prefix_group,
            "completions": [c["text"] for c in completions],
            "pairs": pair_results,
        })

        # Free hidden state tensors
        del completion_data
        torch.cuda.empty_cache()

    print(f"  Processing completed in {time.time() - t_process:.1f}s")

    # ---- Aggregate summary per relationship type per layer ----
    print("\nAggregating results by relationship type...")
    summary = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        rel = result["relationship"]
        for pair in result["pairs"]:
            for lm in pair["layer_metrics"]:
                layer = lm["layer"]
                summary[rel][layer].append(lm)

    # Compute mean/std per layer per relationship
    summary_stats = {}
    for rel, layers in summary.items():
        layer_stats = []
        for layer in sorted(layers.keys()):
            metrics = layers[layer]
            cos_div = [m["cosine_diverging"] for m in metrics]
            cos_shared = [m["cosine_shared"] for m in metrics]
            l2_norm = [m["l2_normalized"] for m in metrics]
            cka_vals = [m["cka"] for m in metrics]
            kl_vals = [m["kl_divergence"] for m in metrics if m["kl_divergence"] is not None]

            layer_stats.append({
                "layer": layer,
                "n_pairs": len(metrics),
                "cosine_diverging_mean": float(np.mean(cos_div)),
                "cosine_diverging_std": float(np.std(cos_div)),
                "cosine_shared_mean": float(np.mean(cos_shared)),
                "cosine_shared_std": float(np.std(cos_shared)),
                "l2_normalized_mean": float(np.mean(l2_norm)),
                "l2_normalized_std": float(np.std(l2_norm)),
                "cka_mean": float(np.mean(cka_vals)),
                "cka_std": float(np.std(cka_vals)),
                "kl_divergence_mean": float(np.mean(kl_vals)) if kl_vals else None,
                "kl_divergence_std": float(np.std(kl_vals)) if kl_vals else None,
            })
        summary_stats[rel] = layer_stats

    # ---- Aggregate by relationship × prefix_group ----
    print("\nAggregating by relationship × prefix_group...")
    prefix_summary_raw = defaultdict(lambda: defaultdict(list))

    for result in all_results:
        key = f"{result['relationship']}_{result['prefix_group']}"
        for pair in result["pairs"]:
            for lm in pair["layer_metrics"]:
                prefix_summary_raw[key][lm["layer"]].append(lm)

    prefix_summary_stats = {}
    for key, layers in prefix_summary_raw.items():
        layer_stats = []
        for layer in sorted(layers.keys()):
            metrics = layers[layer]
            cos_div = [m["cosine_diverging"] for m in metrics]
            kl_vals = [m["kl_divergence"] for m in metrics if m["kl_divergence"] is not None]
            layer_stats.append({
                "layer": layer,
                "n_pairs": len(metrics),
                "cosine_diverging_mean": float(np.mean(cos_div)),
                "cosine_diverging_std": float(np.std(cos_div)),
                "kl_divergence_mean": float(np.mean(kl_vals)) if kl_vals else None,
                "kl_divergence_std": float(np.std(kl_vals)) if kl_vals else None,
            })
        prefix_summary_stats[key] = layer_stats

    for key, stats in prefix_summary_stats.items():
        n = stats[0]["n_pairs"] if stats else 0
        print(f"  {key}: {n} pairs")

    # ---- Save results ----
    print("\nSaving results...")

    # Strip tensors from full results before saving
    saveable_results = []
    for result in all_results:
        saveable_results.append({
            "group_id": result["group_id"],
            "prompt": result["prompt"],
            "relationship": result["relationship"],
            "prefix_group": result["prefix_group"],
            "completions": result["completions"],
            "pairs": [{
                "label_a": p["label_a"],
                "label_b": p["label_b"],
                "shared_prefix_len": p["shared_prefix_len"],
                "layer_metrics": p["layer_metrics"],
            } for p in result["pairs"]],
        })

    with open(results_dir / "full_results.json", "w") as f:
        json.dump(saveable_results, f, indent=2)
    print(f"  Saved full_results.json ({len(saveable_results)} groups)")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)
    print(f"  Saved summary.json")

    with open(results_dir / "prefix_summary.json", "w") as f:
        json.dump(prefix_summary_stats, f, indent=2)
    print(f"  Saved prefix_summary.json")

    # ---- Visualizations ----
    print("\nGenerating visualizations...")
    plot_divergence_curves(summary_stats, results_dir)
    plot_pivot_trajectories(pivot_data, results_dir)
    plot_relationship_heatmap(all_results, results_dir)
    plot_kl_crossover(summary_stats, results_dir)
    plot_prefix_controlled(prefix_summary_stats, results_dir)

    # ---- Print key findings ----
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    for rel in ["synonym", "antonym", "style", "unrelated"]:
        if rel not in summary_stats:
            continue
        stats = summary_stats[rel]
        # Find layer of minimum cosine similarity (maximum divergence)
        min_cos_layer = min(stats, key=lambda s: s["cosine_diverging_mean"])
        max_cos_layer = max(stats, key=lambda s: s["cosine_diverging_mean"])
        print(f"\n  {rel.upper()}:")
        print(f"    Max similarity: {max_cos_layer['cosine_diverging_mean']:.4f} (layer {max_cos_layer['layer']})")
        print(f"    Min similarity: {min_cos_layer['cosine_diverging_mean']:.4f} (layer {min_cos_layer['layer']})")

        # Divergence onset: first layer where cos drops below synonym mean - 2σ
        if rel == "antonym" and "synonym" in summary_stats:
            syn_stats = {s["layer"]: s for s in summary_stats["synonym"]}
            onset = None
            for s in stats:
                layer = s["layer"]
                if layer in syn_stats:
                    threshold = syn_stats[layer]["cosine_diverging_mean"] - 2 * syn_stats[layer]["cosine_diverging_std"]
                    if s["cosine_diverging_mean"] < threshold:
                        onset = layer
                        break
            if onset is not None:
                print(f"    Divergence onset (< synonym - 2σ): layer {onset}")

    # ---- Prefix-controlled findings ----
    print("\n" + "=" * 70)
    print("PREFIX-CONTROLLED ANALYSIS")
    print("=" * 70)
    if "synonym_immediate" in prefix_summary_stats and "antonym_immediate" in prefix_summary_stats:
        syn_kl = {s["layer"]: s["kl_divergence_mean"] for s in prefix_summary_stats["synonym_immediate"]
                  if s["kl_divergence_mean"] is not None}
        ant_kl = {s["layer"]: s["kl_divergence_mean"] for s in prefix_summary_stats["antonym_immediate"]
                  if s["kl_divergence_mean"] is not None}
        common_layers = sorted(set(syn_kl) & set(ant_kl))
        n_syn_higher = sum(1 for l in common_layers if syn_kl[l] > ant_kl[l])
        n_ant_higher = sum(1 for l in common_layers if ant_kl[l] > syn_kl[l])
        print(f"\n  Immediate-divergence pairs (both diverge at token 1):")
        print(f"    Layers where synonym KL > antonym KL: {n_syn_higher}/{len(common_layers)}")
        print(f"    Layers where antonym KL > synonym KL: {n_ant_higher}/{len(common_layers)}")

        # Check for crossovers
        prev_diff = None
        crossovers = []
        for layer in common_layers:
            diff = syn_kl[layer] - ant_kl[layer]
            if prev_diff is not None and prev_diff * diff < 0:
                crossovers.append(layer)
            prev_diff = diff
        if crossovers:
            print(f"    Crossover points: layers {crossovers}")
        else:
            print(f"    No crossover points found")

        # KL at key layers
        for l in [0, 16, 34]:
            if l in syn_kl and l in ant_kl:
                print(f"    L{l}: synonym={syn_kl[l]:.2f}, antonym={ant_kl[l]:.2f}, ratio={syn_kl[l]/ant_kl[l]:.2f}x")

    t_total = time.time() - t_start
    print(f"\nTotal runtime: {t_total:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
