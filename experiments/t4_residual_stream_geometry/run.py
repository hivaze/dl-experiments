"""
Residual Stream Geometry Across Depth (T-4)
============================================
Track how the geometry of the hidden-state manifold changes layer by layer
in Qwen3-4B-Instruct-2507.

Key metrics:
  1. Effective dimensionality (participation ratio of singular values)
  2. Isotropy (cosine similarity distribution, spectral flatness)
  3. Token clustering by prompt category (intra vs inter similarity)
  4. Anisotropy collapse (norm growth, centering effect, SV dominance)

Uses pre-generated calibration completions (vLLM, temp=0) so we analyze
the model's own output. Metrics computed only on completion tokens.
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
NUM_RANDOM_PAIRS = 5000  # for cosine similarity sampling
NUM_CLUSTER_PAIRS = 2000  # per category group for clustering analysis
SV_SPECTRUM_K = 200  # number of singular values to store for plotting

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
    print(f"  Total completion tokens: {data['config']['total_completion_tokens']}")
    return completions


def cosine_similarity_pairs(matrix, num_pairs, rng):
    """Compute cosine similarity for random pairs of rows."""
    n = matrix.shape[0]
    if n < 2:
        return np.array([])
    idx_a = rng.integers(0, n, size=num_pairs)
    idx_b = rng.integers(0, n, size=num_pairs)
    # Avoid self-pairs
    mask = idx_a == idx_b
    idx_b[mask] = (idx_b[mask] + 1) % n
    a = matrix[idx_a]
    b = matrix[idx_b]
    norms_a = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
    norms_b = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    cos = np.sum((a / norms_a) * (b / norms_b), axis=1)
    return cos


# ============================================================================
# Hidden State Extraction
# ============================================================================

def extract_hidden_states(model, tokenizer, calibration_data):
    """Extract hidden states at every layer for all completion tokens.

    Returns:
        pooled: dict mapping layer_idx -> np.array [total_tokens, hidden_dim]
                layer -1 = embedding, 0..N-1 = transformer layers
        per_prompt: list of dicts, each mapping layer_idx -> np.array [n_tokens, hidden_dim]
        categories: list of category strings (one per prompt)
    """
    num_layers = len(model.model.layers)
    pooled = {i: [] for i in range(-1, num_layers)}
    per_prompt = []
    categories = []

    for entry in tqdm(calibration_data, desc="Extracting hidden states"):
        full_text = entry["full_text"]
        prompt_len = entry["prompt_token_count"]
        categories.append(entry["category"])

        tokens = tokenizer(full_text, return_tensors="pt").to(DEVICE)
        input_ids = tokens["input_ids"]

        # Hook-based extraction
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
            model(**tokens, use_cache=False)

        for h in hooks:
            h.remove()

        # Embedding layer
        with torch.no_grad():
            embed_out = model.model.embed_tokens(input_ids)

        # Extract completion tokens only (positions >= prompt_len)
        prompt_entry = {}
        embed_np = embed_out[0, prompt_len:, :].cpu().float().numpy()
        pooled[-1].append(embed_np)
        prompt_entry[-1] = embed_np

        for i in range(num_layers):
            hs = hidden_states[i][0, prompt_len:, :].cpu().float().numpy()
            pooled[i].append(hs)
            prompt_entry[i] = hs

        per_prompt.append(prompt_entry)

        del hidden_states, embed_out, tokens
        torch.cuda.empty_cache()

    # Concatenate pooled arrays
    for key in pooled:
        pooled[key] = np.concatenate(pooled[key], axis=0)

    total_tokens = pooled[-1].shape[0]
    hidden_dim = pooled[-1].shape[1]
    print(f"Extracted {total_tokens} completion tokens, hidden_dim={hidden_dim}, "
          f"{num_layers + 1} layers (embed + {num_layers} transformer)")
    return pooled, per_prompt, categories


# ============================================================================
# Geometry Metrics
# ============================================================================

def compute_svd_metrics(matrix):
    """Compute SVD-based geometry metrics for a token matrix.

    Args:
        matrix: np.array [N, D] of token representations

    Returns dict with participation_ratio, variance_explained, spectral_flatness, etc.
    """
    # Center
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    n = centered.shape[0]

    # SVD (economy)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)

    S2 = S ** 2
    total_var = S2.sum()

    # Participation ratio
    pr = (total_var ** 2) / (S2 ** 2).sum() if (S2 ** 2).sum() > 0 else 0.0

    # Variance explained by top-k
    cumvar = np.cumsum(S2) / (total_var + 1e-15)
    var_top1 = cumvar[0] if len(cumvar) > 0 else 0.0
    var_top5 = cumvar[4] if len(cumvar) > 4 else cumvar[-1]
    var_top10 = cumvar[9] if len(cumvar) > 9 else cumvar[-1]
    var_top50 = cumvar[49] if len(cumvar) > 49 else cumvar[-1]
    var_top100 = cumvar[99] if len(cumvar) > 99 else cumvar[-1]

    # Spectral flatness (geometric mean / arithmetic mean of eigenvalues)
    eigenvalues = S2 / max(n - 1, 1)
    ev_pos = eigenvalues[eigenvalues > 1e-15]
    if len(ev_pos) > 0:
        log_gm = np.mean(np.log(ev_pos))
        am = np.mean(ev_pos)
        spectral_flatness = float(np.exp(log_gm) / am) if am > 0 else 0.0
    else:
        spectral_flatness = 0.0

    # IsoScore = 1 - max_eigenvalue / sum_eigenvalues
    isoscore = 1.0 - float(eigenvalues[0] / (eigenvalues.sum() + 1e-15))

    # Store top-k singular values for plotting
    sv_spectrum = S[:SV_SPECTRUM_K].tolist()

    return {
        "participation_ratio": float(pr),
        "var_top1": float(var_top1),
        "var_top5": float(var_top5),
        "var_top10": float(var_top10),
        "var_top50": float(var_top50),
        "var_top100": float(var_top100),
        "spectral_flatness": spectral_flatness,
        "isoscore": isoscore,
        "sv_spectrum": sv_spectrum,
        "num_nonzero_sv": int(np.sum(S > 1e-10)),
    }


def compute_cosine_metrics(matrix, rng):
    """Compute cosine similarity metrics (raw and centered)."""
    # Raw cosine similarity
    cos_raw = cosine_similarity_pairs(matrix, NUM_RANDOM_PAIRS, rng)

    # Centered cosine similarity
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cos_centered = cosine_similarity_pairs(centered, NUM_RANDOM_PAIRS, rng)

    return {
        "mean_cosine_raw": float(np.mean(cos_raw)) if len(cos_raw) > 0 else 0.0,
        "std_cosine_raw": float(np.std(cos_raw)) if len(cos_raw) > 0 else 0.0,
        "mean_cosine_centered": float(np.mean(cos_centered)) if len(cos_centered) > 0 else 0.0,
        "std_cosine_centered": float(np.std(cos_centered)) if len(cos_centered) > 0 else 0.0,
    }


def compute_norm_metrics(matrix):
    """Compute L2 norm statistics."""
    norms = np.linalg.norm(matrix, axis=1)
    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
    }


# ============================================================================
# Layer Impact & Residual Persistence
# ============================================================================

def compute_layer_impact(pooled, num_layers):
    """How much does each layer change the residual stream?

    For each transformer layer l (0..N-1), computes:
      - delta_norm: mean ||h^(l) - h^(l-1)|| — magnitude of layer's update
      - cosine_input_output: mean cos(h^(l), h^(l-1)) — directional preservation
      - update_ratio: mean ||delta|| / ||h^(l)|| — relative update size
      - update_orthogonality: mean cos(delta, h^(l-1)) — is update aligned with or orthogonal to residual?
    """
    results = {}
    for l in range(num_layers):
        prev = pooled[l - 1]  # l-1 = -1 is embedding
        curr = pooled[l]
        delta = curr - prev

        delta_norms = np.linalg.norm(delta, axis=1)
        curr_norms = np.linalg.norm(curr, axis=1)
        prev_norms = np.linalg.norm(prev, axis=1)

        # Directional preservation: cos(output, input)
        cos_io = np.sum(curr * prev, axis=1) / (curr_norms * prev_norms + 1e-10)

        # Relative update size
        update_ratio = delta_norms / (curr_norms + 1e-10)

        # Update alignment with existing residual: cos(delta, prev)
        # ~0 means orthogonal update; ~1 means reinforcing; ~-1 means opposing
        cos_update_prev = np.sum(delta * prev, axis=1) / (delta_norms * prev_norms + 1e-10)

        results[l] = {
            "delta_norm_mean": float(np.mean(delta_norms)),
            "delta_norm_std": float(np.std(delta_norms)),
            "cosine_input_output": float(np.mean(cos_io)),
            "cosine_input_output_std": float(np.std(cos_io)),
            "update_ratio_mean": float(np.mean(update_ratio)),
            "update_ratio_std": float(np.std(update_ratio)),
            "update_orthogonality": float(np.mean(cos_update_prev)),
            "update_orthogonality_std": float(np.std(cos_update_prev)),
        }
    return results


def compute_persistence(pooled, num_layers):
    """How strongly do earlier layer signals persist through later processing?

    For each layer l:
      - cosine_to_embedding: mean cos(h^(l), h^(emb)) — embedding persistence
      - cosine_to_final: mean cos(h^(l), h^(final)) — alignment with output
      - update_alignment_to_final: mean cos(delta_l, h^(final)) — does this layer's update survive to output?
      - update_projection_fraction: mean |<delta_l, h_final_unit>| / ||delta_l|| — fraction of update in final direction

    Also computes cumulative drift: cos(h^(l), h^(l-k)) for gaps k=1,2,4,8,16.
    """
    embed = pooled[-1]
    final = pooled[num_layers - 1]

    embed_norms = np.linalg.norm(embed, axis=1) + 1e-10
    final_norms = np.linalg.norm(final, axis=1) + 1e-10
    final_unit = final / (final_norms[:, None])

    per_layer = {}
    for l in range(-1, num_layers):
        curr = pooled[l]
        curr_norms = np.linalg.norm(curr, axis=1) + 1e-10

        cos_to_embed = np.sum(curr * embed, axis=1) / (curr_norms * embed_norms)
        cos_to_final = np.sum(curr * final, axis=1) / (curr_norms * final_norms)

        entry = {
            "cosine_to_embedding": float(np.mean(cos_to_embed)),
            "cosine_to_final": float(np.mean(cos_to_final)),
        }

        if l >= 0:
            prev = pooled[l - 1]
            delta = curr - prev
            delta_norms = np.linalg.norm(delta, axis=1) + 1e-10

            cos_update_final = np.sum(delta * final, axis=1) / (delta_norms * final_norms)
            entry["update_alignment_to_final"] = float(np.mean(cos_update_final))

            # Fraction of update vector that projects onto final direction
            proj_scalar = np.sum(delta * final_unit, axis=1)
            entry["update_projection_on_final"] = float(np.mean(np.abs(proj_scalar)))
            entry["update_projection_fraction"] = float(np.mean(np.abs(proj_scalar) / delta_norms))

        per_layer[l] = entry

    # Cumulative drift: cos(h^l, h^(l-k)) for various gap sizes
    drift = {}
    for gap in [1, 2, 4, 8, 16]:
        gap_cosines = []
        for l in range(gap, num_layers):
            curr = pooled[l]
            prev = pooled[l - gap]
            cn = np.linalg.norm(curr, axis=1) + 1e-10
            pn = np.linalg.norm(prev, axis=1) + 1e-10
            cos = np.sum(curr * prev, axis=1) / (cn * pn)
            gap_cosines.append({"layer": l, "cosine": float(np.mean(cos))})
        drift[f"gap_{gap}"] = gap_cosines

    return per_layer, drift


def compute_update_correlations(pooled, num_layers):
    """Cosine similarity matrix between mean layer updates.

    Reveals which layers push the residual stream in similar directions.
    """
    hidden_dim = pooled[0].shape[1]
    mean_deltas = np.zeros((num_layers, hidden_dim))
    for l in range(num_layers):
        mean_deltas[l] = (pooled[l] - pooled[l - 1]).mean(axis=0)

    norms = np.linalg.norm(mean_deltas, axis=1, keepdims=True) + 1e-10
    normed = mean_deltas / norms
    corr = normed @ normed.T
    return corr


def compute_residual_decomposition(pooled, num_layers):
    """Decompose final representation into per-layer contributions.

    h^(final) = h^(emb) + sum_{l=0}^{N-1} delta_l

    For each layer l, compute the projection of delta_l onto h^(final),
    giving the fraction of the final representation "explained" by each layer.
    """
    final = pooled[num_layers - 1]
    final_norms = np.linalg.norm(final, axis=1, keepdims=True) + 1e-10
    final_unit = final / final_norms

    contributions = {}

    # Embedding contribution
    embed = pooled[-1]
    proj_embed = np.sum(embed * final_unit, axis=1)
    contributions["embedding"] = {
        "signed_projection_mean": float(np.mean(proj_embed)),
        "signed_projection_std": float(np.std(proj_embed)),
        "abs_projection_mean": float(np.mean(np.abs(proj_embed))),
        "fraction_of_final_norm": float(np.mean(proj_embed / final_norms.squeeze())),
    }

    for l in range(num_layers):
        delta = pooled[l] - pooled[l - 1]
        proj = np.sum(delta * final_unit, axis=1)
        contributions[f"layer_{l}"] = {
            "signed_projection_mean": float(np.mean(proj)),
            "signed_projection_std": float(np.std(proj)),
            "abs_projection_mean": float(np.mean(np.abs(proj))),
            "fraction_of_final_norm": float(np.mean(proj / final_norms.squeeze())),
        }

    return contributions


def compute_clustering_metrics(per_prompt, categories, layer_idx, rng):
    """Compute intra/inter-category cosine similarity for a layer."""
    # Group tokens by category
    cat_to_vectors = {}
    for i, cat in enumerate(categories):
        vecs = per_prompt[i][layer_idx]
        if cat not in cat_to_vectors:
            cat_to_vectors[cat] = []
        cat_to_vectors[cat].append(vecs)

    for cat in cat_to_vectors:
        cat_to_vectors[cat] = np.concatenate(cat_to_vectors[cat], axis=0)

    cats = sorted(cat_to_vectors.keys())

    # Intra-category similarity
    intra_sims = []
    for cat in cats:
        vecs = cat_to_vectors[cat]
        if vecs.shape[0] < 2:
            continue
        cos = cosine_similarity_pairs(vecs, min(NUM_CLUSTER_PAIRS, vecs.shape[0] * 10), rng)
        intra_sims.append(float(np.mean(cos)))
    mean_intra = float(np.mean(intra_sims)) if intra_sims else 0.0

    # Inter-category similarity
    inter_sims = []
    for i_c, cat_a in enumerate(cats):
        for j_c, cat_b in enumerate(cats):
            if j_c <= i_c:
                continue
            va = cat_to_vectors[cat_a]
            vb = cat_to_vectors[cat_b]
            n_a, n_b = va.shape[0], vb.shape[0]
            n_pairs = min(NUM_CLUSTER_PAIRS, n_a * n_b)
            idx_a = rng.integers(0, n_a, size=n_pairs)
            idx_b = rng.integers(0, n_b, size=n_pairs)
            a = va[idx_a]
            b = vb[idx_b]
            na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
            nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
            cos = np.sum((a / na) * (b / nb), axis=1)
            inter_sims.append(float(np.mean(cos)))
    mean_inter = float(np.mean(inter_sims)) if inter_sims else 0.0

    separation = (mean_intra - mean_inter) / (abs(mean_inter) + 1e-10)

    # Category centroids and pairwise distances
    centroids = {}
    for cat in cats:
        centroids[cat] = cat_to_vectors[cat].mean(axis=0)

    centroid_dists = {}
    for i_c, cat_a in enumerate(cats):
        for j_c, cat_b in enumerate(cats):
            if j_c <= i_c:
                continue
            ca, cb = centroids[cat_a], centroids[cat_b]
            cos = np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-10)
            centroid_dists[f"{cat_a}_vs_{cat_b}"] = float(cos)

    return {
        "intra_category_sim": mean_intra,
        "inter_category_sim": mean_inter,
        "cluster_separation": float(separation),
        "centroid_cosine": centroid_dists,
        "per_category_intra": dict(zip(cats, intra_sims)) if len(intra_sims) == len(cats) else {},
    }


# ============================================================================
# Main Analysis
# ============================================================================

def analyze(pooled, per_prompt, categories, num_layers):
    """Compute all geometry metrics for every layer."""
    rng = np.random.default_rng(SEED)
    layer_indices = list(range(-1, num_layers))
    results = {}

    for layer_idx in tqdm(layer_indices, desc="Computing metrics"):
        matrix = pooled[layer_idx]

        svd = compute_svd_metrics(matrix)
        cos = compute_cosine_metrics(matrix, rng)
        norms = compute_norm_metrics(matrix)
        clust = compute_clustering_metrics(per_prompt, categories, layer_idx, rng)

        results[layer_idx] = {**svd, **cos, **norms, **clust}

    return results


# ============================================================================
# Visualization
# ============================================================================

def create_plots(results, num_layers):
    """Generate all visualization plots."""
    layers = list(range(-1, num_layers))
    layer_labels = ["emb"] + [str(i) for i in range(num_layers)]
    x = np.arange(len(layers))

    def get_vals(key):
        return [results[l][key] for l in layers]

    # --- Plot 1: Geometry Overview (4-panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Residual Stream Geometry Across Depth — Qwen3-4B-Instruct-2507", fontsize=14)

    # A: Effective dimensionality
    ax = axes[0, 0]
    ax.plot(x, get_vals("participation_ratio"), "b-o", markersize=3)
    ax.set_ylabel("Participation Ratio")
    ax.set_title("A. Effective Dimensionality")
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.grid(True, alpha=0.3)

    # B: Isotropy
    ax = axes[0, 1]
    ax.plot(x, get_vals("mean_cosine_raw"), "r-o", markersize=3, label="Mean cosine (raw)")
    ax.set_ylabel("Mean Cosine Similarity", color="r")
    ax.tick_params(axis="y", labelcolor="r")
    ax2 = ax.twinx()
    ax2.plot(x, get_vals("spectral_flatness"), "g-s", markersize=3, label="Spectral flatness")
    ax2.set_ylabel("Spectral Flatness", color="g")
    ax2.tick_params(axis="y", labelcolor="g")
    ax.set_title("B. Isotropy Metrics")
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # C: Anisotropy decomposition
    ax = axes[1, 0]
    ax.plot(x, get_vals("mean_cosine_raw"), "r-o", markersize=3, label="Raw")
    ax.plot(x, get_vals("mean_cosine_centered"), "b-s", markersize=3, label="Centered")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("C. Anisotropy: Raw vs Centered")
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # D: Norm statistics
    ax = axes[1, 1]
    means = get_vals("mean_norm")
    stds = get_vals("std_norm")
    ax.plot(x, means, "k-o", markersize=3, label="Mean norm")
    ax.fill_between(x, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                     alpha=0.2, color="gray", label="±1 std")
    ax.set_ylabel("L2 Norm")
    ax.set_title("D. Representation Norms")
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "geometry_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved geometry_overview.png")

    # --- Plot 2: Token Clustering ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Token Clustering by Prompt Category", fontsize=14)

    # A: Intra vs inter similarity
    ax = axes[0]
    ax.plot(x, get_vals("intra_category_sim"), "b-o", markersize=3, label="Intra-category")
    ax.plot(x, get_vals("inter_category_sim"), "r-s", markersize=3, label="Inter-category")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("A. Intra vs Inter-Category Similarity")
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # B: Cluster separation
    ax = axes[1]
    ax.plot(x, get_vals("cluster_separation"), "g-o", markersize=3)
    ax.set_ylabel("Separation Ratio")
    ax.set_title("B. Cluster Separation (intra-inter)/|inter|")
    ax.set_xlabel("Layer")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # C: Centroid heatmaps at selected layers
    ax = axes[2]
    sample_layers = [0, num_layers // 3, 2 * num_layers // 3, num_layers - 1]
    centroid_data = results[sample_layers[0]].get("centroid_cosine", {})
    cats = sorted(set(c.split("_vs_")[0] for c in centroid_data.keys()) |
                  set(c.split("_vs_")[1] for c in centroid_data.keys()))
    if cats:
        n_cats = len(cats)
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        # Average centroid cosine across sample layers
        mat = np.zeros((n_cats, n_cats))
        for pair, val in results[sample_layers[-1]]["centroid_cosine"].items():
            a, b = pair.split("_vs_")
            mat[cat_to_idx[a], cat_to_idx[b]] = val
            mat[cat_to_idx[b], cat_to_idx[a]] = val
        np.fill_diagonal(mat, 1.0)
        im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(n_cats))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_cats))
        ax.set_yticklabels(cats, fontsize=7)
        ax.set_title(f"C. Centroid Cosine (Layer {sample_layers[-1]})")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No centroid data", ha="center", va="center")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "token_clustering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved token_clustering.png")

    # --- Plot 3: Singular Value Spectra ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Singular Value Spectra Across Depth", fontsize=14)

    # A: Heatmap of SV spectra
    ax = axes[0]
    max_k = min(SV_SPECTRUM_K, len(results[0]["sv_spectrum"]))
    sv_matrix = np.zeros((len(layers), max_k))
    for i, l in enumerate(layers):
        sv = results[l]["sv_spectrum"][:max_k]
        sv_matrix[i, :len(sv)] = sv
    # Log scale
    sv_log = np.log10(sv_matrix + 1e-10)
    im = ax.imshow(sv_log, aspect="auto", cmap="viridis",
                    extent=[0, max_k, len(layers) - 0.5, -0.5])
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Layer")
    ax.set_yticks(np.arange(0, len(layers), 4))
    ax.set_yticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.set_title("A. log₁₀(Singular Values)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # B: Cumulative variance for selected layers
    ax = axes[1]
    sample_layers_sv = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_layers_sv)))
    for sl, color in zip(sample_layers_sv, colors):
        sv = np.array(results[sl]["sv_spectrum"])
        sv2 = sv ** 2
        cumvar = np.cumsum(sv2) / (sv2.sum() + 1e-15)
        ax.plot(cumvar, color=color, label=f"Layer {sl}")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("B. Cumulative Variance Explained")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(100, max_k))
    ax.axhline(y=0.9, color="k", linestyle="--", alpha=0.3, label="90%")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "singular_value_spectra.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved singular_value_spectra.png")

    # --- Plot 4: Variance explained summary ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, label, color in [(1, "Top-1", "red"), (5, "Top-5", "orange"),
                             (10, "Top-10", "green"), (50, "Top-50", "blue"),
                             (100, "Top-100", "purple")]:
        key = f"var_top{k}"
        ax.plot(x, get_vals(key), "-o", markersize=3, color=color, label=label)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of Variance Explained")
    ax.set_title("Variance Explained by Top-k Singular Values")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([layer_labels[i] for i in range(0, len(layer_labels), 4)], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "variance_explained.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved variance_explained.png")


def create_impact_plots(impact, persistence, drift, update_corr, decomposition, num_layers):
    """Generate plots for layer impact, persistence, and update structure."""
    layers_t = list(range(num_layers))  # transformer layers only (0..35)
    layer_labels_t = [str(i) for i in range(num_layers)]
    x_t = np.arange(len(layers_t))

    layers_all = list(range(-1, num_layers))
    layer_labels_all = ["emb"] + [str(i) for i in range(num_layers)]
    x_all = np.arange(len(layers_all))

    # --- Plot 5: Layer Impact (4-panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Layer Impact on Residual Stream — Qwen3-4B-Instruct-2507", fontsize=14)

    # A: Delta norms (how much each layer changes the residual)
    ax = axes[0, 0]
    delta_norms = [impact[l]["delta_norm_mean"] for l in layers_t]
    delta_stds = [impact[l]["delta_norm_std"] for l in layers_t]
    ax.plot(x_t, delta_norms, "b-o", markersize=3)
    ax.fill_between(x_t, np.array(delta_norms) - np.array(delta_stds),
                     np.array(delta_norms) + np.array(delta_stds), alpha=0.2, color="blue")
    ax.set_ylabel("||h(l) - h(l-1)||")
    ax.set_title("A. Layer Update Magnitude")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_t[::4])
    ax.set_xticklabels([layer_labels_t[i] for i in range(0, len(layer_labels_t), 4)], fontsize=8)
    ax.grid(True, alpha=0.3)

    # B: Cosine(input, output) — directional preservation
    ax = axes[0, 1]
    cos_io = [impact[l]["cosine_input_output"] for l in layers_t]
    ax.plot(x_t, cos_io, "r-o", markersize=3)
    ax.set_ylabel("cos(h(l), h(l-1))")
    ax.set_title("B. Directional Preservation (output vs input)")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_t[::4])
    ax.set_xticklabels([layer_labels_t[i] for i in range(0, len(layer_labels_t), 4)], fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # C: Update ratio — relative contribution
    ax = axes[1, 0]
    ratios = [impact[l]["update_ratio_mean"] for l in layers_t]
    ax.plot(x_t, ratios, "g-o", markersize=3)
    ax.set_ylabel("||delta|| / ||h(l)||")
    ax.set_title("C. Relative Update Size")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_t[::4])
    ax.set_xticklabels([layer_labels_t[i] for i in range(0, len(layer_labels_t), 4)], fontsize=8)
    ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="delta = residual")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # D: Update orthogonality — cos(delta, prev)
    ax = axes[1, 1]
    orth = [impact[l]["update_orthogonality"] for l in layers_t]
    ax.plot(x_t, orth, "m-o", markersize=3)
    ax.set_ylabel("cos(delta, h(l-1))")
    ax.set_title("D. Update-Residual Alignment")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_t[::4])
    ax.set_xticklabels([layer_labels_t[i] for i in range(0, len(layer_labels_t), 4)], fontsize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3, label="orthogonal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "layer_impact.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved layer_impact.png")

    # --- Plot 6: Residual Persistence (4-panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Residual Stream Persistence — Qwen3-4B-Instruct-2507", fontsize=14)

    # A: Cosine to embedding
    ax = axes[0, 0]
    cos_embed = [persistence[l]["cosine_to_embedding"] for l in layers_all]
    ax.plot(x_all, cos_embed, "b-o", markersize=3)
    ax.set_ylabel("cos(h(l), h(emb))")
    ax.set_title("A. Embedding Persistence")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_all[::4])
    ax.set_xticklabels([layer_labels_all[i] for i in range(0, len(layer_labels_all), 4)], fontsize=8)
    ax.grid(True, alpha=0.3)

    # B: Cosine to final
    ax = axes[0, 1]
    cos_final = [persistence[l]["cosine_to_final"] for l in layers_all]
    ax.plot(x_all, cos_final, "r-o", markersize=3)
    ax.set_ylabel("cos(h(l), h(final))")
    ax.set_title("B. Alignment with Final Representation")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_all[::4])
    ax.set_xticklabels([layer_labels_all[i] for i in range(0, len(layer_labels_all), 4)], fontsize=8)
    ax.grid(True, alpha=0.3)

    # C: Update alignment to final — which layers' updates survive?
    ax = axes[1, 0]
    update_align = [persistence[l].get("update_alignment_to_final", 0) for l in layers_t]
    ax.bar(x_t, update_align, color=["green" if v > 0 else "red" for v in update_align], alpha=0.7)
    ax.set_ylabel("cos(delta_l, h(final))")
    ax.set_title("C. Layer Update Alignment to Final Output")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_t[::4])
    ax.set_xticklabels([layer_labels_t[i] for i in range(0, len(layer_labels_t), 4)], fontsize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # D: Cumulative drift for different gaps
    ax = axes[1, 1]
    gap_colors = {1: "blue", 2: "green", 4: "orange", 8: "red", 16: "purple"}
    for gap_key, color in gap_colors.items():
        drift_data = drift[f"gap_{gap_key}"]
        drift_x = [d["layer"] for d in drift_data]
        drift_y = [d["cosine"] for d in drift_data]
        ax.plot(drift_x, drift_y, "-o", markersize=2, color=color, label=f"gap={gap_key}")
    ax.set_ylabel("cos(h(l), h(l-gap))")
    ax.set_title("D. Cumulative Drift by Gap Size")
    ax.set_xlabel("Layer")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "residual_persistence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved residual_persistence.png")

    # --- Plot 7: Update Correlation Matrix ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Layer Update Correlation Matrix — cos(mean_delta_l, mean_delta_m)", fontsize=13)
    im = ax.imshow(update_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(0, num_layers, 4))
    ax.set_xticklabels([str(i) for i in range(0, num_layers, 4)], fontsize=8)
    ax.set_yticks(np.arange(0, num_layers, 4))
    ax.set_yticklabels([str(i) for i in range(0, num_layers, 4)], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "update_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved update_correlation.png")

    # --- Plot 8: Residual Decomposition ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Final Representation Decomposition — Projection of Each Layer's Update onto h(final)",
                 fontsize=13)

    # A: Signed projection (contribution to final norm along final direction)
    ax = axes[0]
    signed_proj = [decomposition["embedding"]["signed_projection_mean"]]
    signed_proj += [decomposition[f"layer_{l}"]["signed_projection_mean"] for l in range(num_layers)]
    decomp_labels = ["emb"] + [str(i) for i in range(num_layers)]
    x_decomp = np.arange(len(signed_proj))
    colors = ["green" if v > 0 else "red" for v in signed_proj]
    ax.bar(x_decomp, signed_proj, color=colors, alpha=0.7)
    ax.set_ylabel("<delta_l, h_final_unit>")
    ax.set_title("A. Signed Contribution to Final Direction")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_decomp[::4])
    ax.set_xticklabels([decomp_labels[i] for i in range(0, len(decomp_labels), 4)], fontsize=8)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)

    # B: Fraction of update that projects onto final direction
    ax = axes[1]
    proj_frac = [0.0]  # embedding doesn't have update_projection_fraction
    proj_frac += [decomposition[f"layer_{l}"]["fraction_of_final_norm"] for l in range(num_layers)]
    ax.bar(x_decomp[1:], proj_frac[1:], color="steelblue", alpha=0.7)
    ax.set_ylabel("|<delta_l, h_final_unit>| / ||delta_l||")
    ax.set_title("B. Fraction of Update Aligned with Final Direction")
    ax.set_xlabel("Layer")
    ax.set_xticks(x_decomp[1::4])
    ax.set_xticklabels([decomp_labels[i] for i in range(1, len(decomp_labels), 4)], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "residual_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved residual_decomposition.png")


# ============================================================================
# Main
# ============================================================================

def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Load calibration data
    calibration_data = load_calibration_data()

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

    # Extract hidden states
    print("\n--- Extracting Hidden States ---")
    t_extract = time.time()
    pooled, per_prompt, categories = extract_hidden_states(model, tokenizer, calibration_data)
    print(f"Extraction took {time.time() - t_extract:.1f}s")

    # Free GPU
    del model
    torch.cuda.empty_cache()
    print("Model freed from GPU")

    # Compute metrics
    print("\n--- Computing Geometry Metrics ---")
    t_analyze = time.time()
    results = analyze(pooled, per_prompt, categories, num_layers)
    print(f"Geometry analysis took {time.time() - t_analyze:.1f}s")

    # Compute layer impact & persistence metrics
    print("\n--- Computing Layer Impact & Persistence ---")
    t_impact = time.time()
    impact = compute_layer_impact(pooled, num_layers)
    persistence, drift = compute_persistence(pooled, num_layers)
    update_corr = compute_update_correlations(pooled, num_layers)
    decomposition = compute_residual_decomposition(pooled, num_layers)
    print(f"Impact & persistence analysis took {time.time() - t_impact:.1f}s")

    # Print geometry summary table
    print("\n" + "=" * 90)
    print(f"{'Layer':>6} | {'PR':>8} | {'Cos(raw)':>9} | {'Cos(ctr)':>9} | "
          f"{'SpFlat':>8} | {'Norm':>8} | {'Intra':>7} | {'Inter':>7} | {'Sep':>7}")
    print("-" * 90)
    for l in range(-1, num_layers):
        r = results[l]
        lbl = "emb" if l == -1 else str(l)
        print(f"{lbl:>6} | {r['participation_ratio']:>8.1f} | {r['mean_cosine_raw']:>9.4f} | "
              f"{r['mean_cosine_centered']:>9.4f} | {r['spectral_flatness']:>8.5f} | "
              f"{r['mean_norm']:>8.2f} | {r['intra_category_sim']:>7.4f} | "
              f"{r['inter_category_sim']:>7.4f} | {r['cluster_separation']:>7.3f}")
    print("=" * 90)

    # Print layer impact table
    print("\n" + "=" * 110)
    print(f"{'Layer':>6} | {'||delta||':>10} | {'cos(io)':>8} | {'upd_ratio':>10} | "
          f"{'upd_orth':>9} | {'cos→emb':>8} | {'cos→fin':>8} | {'upd→fin':>8} | {'proj_frac':>10}")
    print("-" * 110)
    for l in range(num_layers):
        im = impact[l]
        ps = persistence[l]
        print(f"{l:>6} | {im['delta_norm_mean']:>10.2f} | {im['cosine_input_output']:>8.4f} | "
              f"{im['update_ratio_mean']:>10.4f} | {im['update_orthogonality']:>9.4f} | "
              f"{ps['cosine_to_embedding']:>8.4f} | {ps['cosine_to_final']:>8.4f} | "
              f"{ps.get('update_alignment_to_final', 0):>8.4f} | "
              f"{ps.get('update_projection_fraction', 0):>10.4f}")
    print("=" * 110)

    # Generate plots
    print("\n--- Generating Plots ---")
    create_plots(results, num_layers)
    create_impact_plots(impact, persistence, drift, update_corr, decomposition, num_layers)

    # Save results
    # Convert int keys to strings for JSON
    json_results = {}
    for k, v in results.items():
        layer_key = f"layer_{k}" if k >= 0 else "embedding"
        v_copy = {kk: vv for kk, vv in v.items() if kk != "sv_spectrum"}
        json_results[layer_key] = v_copy

    sv_spectra = {}
    for k, v in results.items():
        layer_key = f"layer_{k}" if k >= 0 else "embedding"
        sv_spectra[layer_key] = v.get("sv_spectrum", [])

    # Convert impact/persistence dicts
    json_impact = {f"layer_{k}": v for k, v in impact.items()}
    json_persistence = {}
    for k, v in persistence.items():
        key = f"layer_{k}" if k >= 0 else "embedding"
        json_persistence[key] = v

    output = {
        "config": {
            "model": MODEL_NAME,
            "seed": SEED,
            "device": DEVICE,
            "num_layers": num_layers,
            "num_prompts": len(categories),
            "total_completion_tokens": int(pooled[-1].shape[0] if -1 in pooled else 0),
            "hidden_dim": int(pooled[-1].shape[1] if -1 in pooled else 0),
            "num_random_pairs": NUM_RANDOM_PAIRS,
            "num_cluster_pairs": NUM_CLUSTER_PAIRS,
        },
        "per_layer": json_results,
        "sv_spectra": sv_spectra,
        "layer_impact": json_impact,
        "persistence": json_persistence,
        "cumulative_drift": drift,
        "update_correlation_matrix": update_corr.tolist() if isinstance(update_corr, np.ndarray) else update_corr,
        "residual_decomposition": decomposition,
        "timing": {
            "total_s": time.time() - t_start,
            "extraction_s": t_analyze - t_extract,
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {RESULTS_DIR / 'summary.json'}")
    print(f"\nTotal time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
