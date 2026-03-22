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
    print(f"Analysis took {time.time() - t_analyze:.1f}s")

    # Print summary table
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

    # Generate plots
    print("\n--- Generating Plots ---")
    create_plots(results, num_layers)

    # Save results
    # Convert int keys to strings for JSON
    json_results = {}
    for k, v in results.items():
        layer_key = f"layer_{k}" if k >= 0 else "embedding"
        # Remove sv_spectrum from main summary (save separately)
        v_copy = {kk: vv for kk, vv in v.items() if kk != "sv_spectrum"}
        json_results[layer_key] = v_copy

    sv_spectra = {}
    for k, v in results.items():
        layer_key = f"layer_{k}" if k >= 0 else "embedding"
        sv_spectra[layer_key] = v.get("sv_spectrum", [])

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
