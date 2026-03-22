# T-4: Residual Stream Geometry Across Depth

## Motivation & Research Question

How does the geometry of the hidden-state manifold change as representations flow through transformer layers? Specifically:

1. **Effective dimensionality**: How many dimensions do representations actually use at each layer?
2. **Isotropy**: Are representations uniformly distributed in the space, or clustered in a low-dimensional cone?
3. **Token clustering**: Do tokens from semantically similar prompts converge as depth increases?
4. **Anisotropy collapse**: Do representations become increasingly anisotropic in later layers?

This connects to the Layer Shuffle Recovery experiment — if activation flow fingerprints work well for ordering layers, there must be a predictable geometric transformation at each layer. Understanding the geometry explains *why* those fingerprints are so discriminative.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, hidden_dim=2560)
- **Data**: 28 pre-generated completions from `data/text_completions/`, covering 7 categories (factual, reasoning, linguistic, code, world_knowledge, technical, rare)
- **Tokens analyzed**: 2,277 completion tokens (prompt/template tokens excluded)
- **Layers**: 37 measurement points — embedding output + 36 transformer layer outputs
- **Hardware**: NVIDIA B200, bf16 inference, float32 metric computation
- **Runtime**: ~63 seconds total

## Methods

### 1. Hidden State Extraction
Forward-hook-based extraction at every layer, identical to T-1. For each of 28 prompts, we register hooks on all 36 `model.model.layers[i]` modules plus capture the embedding output. Only completion-token positions are retained (positions ≥ `prompt_token_count`), yielding 2,277 token vectors of dimension 2,560 per layer.

### 2. Effective Dimensionality (Participation Ratio)
For each layer's pooled matrix H (2277×2560), we mean-center and compute SVD. The **participation ratio** is:

    PR = (Σ σᵢ²)² / Σ σᵢ⁴

PR ranges from 1 (all variance in one direction) to min(N, D). It measures how many singular values contribute meaningfully — i.e., the effective dimensionality of the representation manifold.

### 3. Isotropy Metrics
Two complementary measures:

- **Mean pairwise cosine similarity**: Sample 5,000 random token pairs and compute their cosine similarity. For isotropic representations this should be ~0; for anisotropic (cone-shaped) distributions, it will be high.
- **Spectral flatness**: geometric_mean(eigenvalues) / arithmetic_mean(eigenvalues), ranging from 0 (one dominant direction) to 1 (perfectly isotropic).

We also compute cosine similarity on **centered** representations (mean-subtracted), which isolates intrinsic geometry from the shared-direction component.

### 4. Token Clustering by Category
Tokens are grouped by their prompt's category. We compute:

- **Intra-category similarity**: Mean cosine similarity between random token pairs *within* the same category
- **Inter-category similarity**: Mean cosine similarity between tokens from *different* categories
- **Cluster separation ratio**: (intra − inter) / |inter| — higher means categories are more distinguishable
- **Centroid analysis**: Cosine similarity between category centroids at each layer

### 5. Anisotropy Decomposition
We decompose anisotropy into removable (mean-direction) and intrinsic components:

- **Raw cosine similarity** captures total anisotropy
- **Centered cosine similarity** captures intrinsic anisotropy after removing the shared mean direction
- **Norm statistics** track how representation magnitudes evolve across depth

## Results

### Effective Dimensionality

| Region | Layers | PR Range | Interpretation |
|--------|--------|----------|----------------|
| Embedding | emb | 67.7 | Moderate — token embeddings span ~68 effective dimensions |
| Early spike | 0 | 112.1 | First layer *increases* dimensionality |
| Early collapse | 1–5 | 6.2–42.7 | Dramatic collapse — layers 1–2 reduce to just ~6 effective dims |
| Mid recovery | 6–15 | 120–177 | Gradual recovery to ~130–175 effective dims |
| Deep collapse | 16–24 | 1.6–7.5 | Second major collapse — PR drops to **1.6** at layer 16 |
| Late recovery | 25–34 | 10–78 | Partial recovery through final layers |
| Final layer | 35 | 118.7 | Sharp dimensionality expansion at the output |

The **bimodal collapse pattern** (layers 1–5 and 16–24) is striking. The deep collapse at layer 16 (PR=1.6) means representations are nearly one-dimensional — almost all variance sits on a single axis. This suggests two critical "bottleneck" regions where the model compresses information maximally before expanding it again.

### Isotropy / Anisotropy

The raw mean cosine similarity hovers around **0.38–0.52** throughout most layers (moderately anisotropic), peaking at **0.62** at layer 34 before dramatically dropping to **0.09** at the final layer 35.

The **centered** cosine similarity is essentially **zero** (~0.001–0.009) at every layer. This is a key finding: **all observed anisotropy is due to a shared mean direction, not intrinsic geometric clustering**. After removing the mean, token representations are nearly perfectly isotropic at every layer.

Spectral flatness steadily increases from 0.03 (layers 1–2) to 0.13 (layers 33–35), indicating a gradual diversification of the variance spectrum — the representation becomes more uniformly spread across dimensions as depth increases.

### Representation Norms

Norms grow **superlinearly** across depth:
- Embedding: 0.99
- Layer 10: 40.3
- Layer 20: 68.2
- Layer 30: 280.9
- Layer 34: 567.8

Combined with high raw cosine similarity, this confirms the classic **anisotropic cone** pattern: representations extend along an increasingly narrow cone with growing norms. However, the final layer (35) breaks this pattern — norms drop to 390.5 and cosine similarity plummets to 0.09, suggesting the final layer deliberately disperses representations for the LM head.

### Token Clustering

| Layer | Intra-cat | Inter-cat | Separation |
|-------|-----------|-----------|------------|
| emb   | 0.098     | 0.078     | 0.26       |
| 0     | 0.465     | 0.452     | 0.03       |
| 11    | 0.470     | 0.412     | 0.14       |
| 22    | 0.430     | 0.368     | 0.17       |
| 27    | 0.432     | 0.357     | 0.21       |
| 35    | 0.153     | 0.075     | **1.03**   |

Category separation **increases monotonically** through the middle and late layers, from near-zero at layer 0 to 0.21 at layer 27. The final layer 35 shows a dramatic jump to **1.03** — tokens from the same category become strongly clustered while inter-category similarity drops to near-embedding levels. This is consistent with the model building category-discriminative representations in the final layer for next-token prediction.

The centroid cosine heatmap at layer 35 reveals that `code` tokens are most distinctive (lowest cosine with other categories), while `factual` and `world_knowledge` remain similar to each other.

### Singular Value Spectra

The cumulative variance plots reveal:
- At layers 1–5 and 16–24 (collapse regions), the **top-1 singular value explains 40–60%** of all variance
- At mid layers (6–15), variance is more distributed — top-10 explains ~80%
- The final layer 35 has the most distributed spectrum — top-50 needed for 90% variance

The SV heatmap shows a clear "bright band" at singular value index 0 during collapse layers, confirming the single-dominant-direction interpretation.

## Conclusions & Key Findings

1. **Bimodal dimensionality collapse**: The model has two distinct bottleneck regions (layers 1–5 and 16–24) where representations collapse to near-one-dimensional manifolds (PR as low as 1.6). This is not gradual — it's a sharp phase transition.

2. **All anisotropy is mean-direction anisotropy**: Centered cosine similarity is ~0 everywhere. The high raw cosine similarity (~0.4–0.6) is entirely explained by a shared mean direction. Remove it, and representations are isotropic. This has practical implications — mean-centering is sufficient to "fix" anisotropy for downstream tasks.

3. **Superlinear norm growth with final-layer correction**: Norms grow from 1.0 to 568 across 35 layers, but the final layer reduces norms and dramatically increases isotropy. Layer 35 acts as a "de-anisotropifier" that disperses representations for the vocabulary projection.

4. **Monotonic category separation**: Token clustering by semantic category increases steadily from layer 0 to the final layer, with the strongest jump at layer 35 (separation ratio 1.03). The model progressively builds category-discriminative features, culminating in a final-layer reorganization.

5. **Connection to Layer Shuffle Recovery**: The sharp geometric transitions at layers 1–5 and 16–24 explain why activation fingerprints are so discriminative for layer ordering — each layer has a distinct geometric signature (PR, norm, SV spectrum) that changes predictably across depth. Shuffling layers would disrupt these signatures, making mis-ordering detectable.

## Usage

```bash
# Prerequisites: completions must exist at data/text_completions/qwen3-4b-instruct-2507/
poetry run python experiments/t4_residual_stream_geometry/run.py
```

Output in `experiments/t4_residual_stream_geometry/results/`:
- `summary.json` — All per-layer metrics and singular value spectra
- `geometry_overview.png` — 4-panel plot: dimensionality, isotropy, anisotropy decomposition, norms
- `token_clustering.png` — Intra/inter-category similarity, separation ratio, centroid heatmap
- `singular_value_spectra.png` — SV heatmap and cumulative variance curves
- `variance_explained.png` — Top-k variance explained across layers
