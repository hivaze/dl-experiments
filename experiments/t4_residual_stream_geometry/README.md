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

1. **Bimodal dimensionality collapse**: The model has two distinct bottleneck regions (layers 1–5 and 16–24) where representations collapse to near-one-dimensional manifolds (PR as low as 1.6). This is not gradual — it's a sharp phase transition. The two collapse regions bookend the "mid recovery" plateau (layers 6–15, PR 120–178) where representations are high-dimensional and distributed.

2. **All anisotropy is mean-direction anisotropy**: Centered cosine similarity is ~0 everywhere (range −0.002 to +0.009). The high raw cosine similarity (~0.4–0.6) is entirely explained by a shared mean direction. Remove it, and representations are isotropic. This has practical implications — mean-centering is sufficient to "fix" anisotropy for downstream tasks like retrieval or clustering.

3. **Superlinear norm growth with final-layer correction**: Norms grow from 1.0 to 568 across 35 layers, but the final layer reduces norms to 391 and dramatically increases isotropy (raw cosine drops from 0.62 to 0.09). Layer 35 acts as a "de-anisotropifier" that disperses representations for the vocabulary projection. This is consistent with the unembedding matrix expecting more isotropic input.

4. **Monotonic category separation**: Token clustering by semantic category increases steadily from layer 0 to the final layer, with the strongest jump at layer 35 (separation ratio 0.03 → 1.03). The model progressively builds category-discriminative features, culminating in a final-layer reorganization. Code tokens are most distinctive at layer 35 (lowest cosine with other categories), while factual and world_knowledge remain similar.

5. **Connection to Layer Shuffle Recovery**: The sharp geometric transitions at layers 1–5 and 16–24 explain why activation fingerprints are so discriminative for layer ordering — each layer has a distinct geometric signature (PR, norm, SV spectrum) that changes predictably across depth. Shuffling layers would disrupt these signatures, making mis-ordering detectable.

## Cross-Experiment Connections

### T-2 (Layer Knockout): Geometry Explains Criticality Asymmetry

T-2 found that layer 0 is catastrophically critical (100.9× loss ratio) despite being linearizable (99% recovery with a full-rank linear map). T-4 explains why: layer 0 is the only layer that *increases* dimensionality (PR: 67.7 → 112.1) and dramatically increases norms (1.0 → 7.7). It performs a massive geometric expansion — projecting 68-dimensional token embeddings into a 112-dimensional working space with 7.7× norm amplification. This is a specific, learnable linear transformation, which is why linear replacement works but removal is catastrophic.

T-2's "bimodal linearity" finding — early and late layers are globally linearizable while middle layers (8–19) are not — maps onto T-4's geometry as follows:

| T-4 Geometry Region | Layers | PR | T-2 Linear Replacement |
|---|---|---|---|
| Early collapse (low PR) | 1–5 | 6–43 | 91–99% recovery |
| Mid recovery (high PR) | 6–15 | 120–178 | −2631% to 87% (fails) |
| Deep collapse (low PR) | 16–24 | 1.6–7.5 | −1795% to −54% (fails) |
| Late recovery | 25–34 | 10–78 | 85–92% recovery |

The correlation is striking: **low-dimensionality regions are linearizable, high-dimensionality regions are not**. When representations are squeezed into 1–6 effective dimensions, the layer's transformation is constrained enough to be captured by a fixed linear map. When representations span 120–178 dimensions, the transformation is input-dependent and globally nonlinear.

T-2 also found layer 6 is a computational hub (2nd most critical, appears in 4/5 top synergistic pairs). In T-4, layer 6 sits at the onset of the mid-recovery plateau (PR jumps from 42.7 at layer 5 to 170.3 at layer 6) — it's the layer where dimensionality first explodes, marking the transition from compressed to distributed representation.

### T-7 (Linearization Gap): Local Smoothness vs Global Geometry

T-7 found a U-shaped nonlinearity profile with middle layers (6–18) being most locally linear (gap ~0.13–0.15) but globally nonlinear (catastrophic replacement failure). T-4's Jacobian consistency data from T-7 resolves this paradox:

- **Layers 4–8**: Low Jacobian consistency (0.55–0.66) despite low linearization gap. The Jacobian is smooth at each operating point but *varies strongly across inputs*. T-4 shows these layers have high PR (120–178) — the many active dimensions create room for input-dependent behavior.
- **Layers 29–35**: High consistency (0.76–0.88), meaning the Jacobian is nearly the same regardless of input. T-4 shows norms are largest here (280–568), and the late-layer norm growth dominates over input-dependent variation, making the transformation effectively fixed.

The geometry also explains T-7's finding that MLP nonlinearity drives the late-layer spike (MLP gap 0.24 at layer 35): the final-layer "de-anisotropification" (PR: 78 → 119, cosine: 0.62 → 0.09) requires nonlinear dispersal that SwiGLU provides.

### T-9 (Weight Spectral Structure): Weight Rank vs Representation Dimensionality

T-9 found no significant correlation between weight effective rank and representation geometry (r = 0.157, p = 0.36). This is initially surprising — shouldn't low-rank weights produce low-dimensional representations? T-4 explains why not: the residual stream is an **accumulated sum** of all upstream layer contributions. A single layer's weight rank constrains only its *incremental update* to the residual stream, not the total dimensionality. The bimodal collapse at layers 1–5 and 16–24 is driven by dominant singular values in the *representations* (top-1 SV explains 40–79% of variance), which can emerge from accumulation dynamics regardless of individual weight ranks.

T-9's key finding — that Q/K routing matrices are lower-rank (0.31) than V/O value matrices (0.52) across all 36 layers — has a geometric interpretation from T-4: routing (deciding where to attend) operates in a lower-dimensional subspace, while value extraction must manipulate the full representation geometry. During the collapse regions (PR 1.6–7.5), even Q/K's low-rank routing is operating on a near-one-dimensional manifold, which may explain why attention patterns become less input-specific in these layers.

T-9 also found that gate_proj effective rank increases dramatically in late layers (+0.166, p < 0.0001). In T-4, late layers show partial PR recovery (10–78) with massive norm growth, suggesting the increased gating capacity is needed to manage the expanding, high-magnitude representations.

### T-17 (Contrastive Trajectories): Geometry of Divergence

T-17 found that antonym pairs maintain high cosine similarity (>0.63) across layers 2–34, with a dip at layer 21. T-4 explains this: during the deep collapse region (layers 16–24, PR 1.6–7.5), representations are squeezed onto a near-one-dimensional manifold. Two sequences processing different tokens ("star" vs "planet") are forced into the same narrow subspace — their cosine similarity remains high because there's essentially only one direction to point in. The dip at layer 21 (PR ~3.7) may reflect a brief re-expansion where the model has enough dimensions to begin separating the antonyms.

T-17's finding that layer 35 is a "universal discriminator" (all cosine similarities drop sharply) aligns directly with T-4's final-layer geometry: PR expands to 118.7, raw cosine drops to 0.09, and category separation jumps to 1.03. The model deliberately disperses representations at the final layer, maximally separating all token alternatives for the LM head.

### Synthesis: The Geometric Processing Pipeline

Combining T-4 with other experiments reveals a coherent geometric processing pipeline:

1. **Layer 0 — Geometric expansion**: Projects embeddings from ~68 to ~112 effective dimensions with 7.7× norm amplification. Linear and critical (T-2).
2. **Layers 1–5 — First compression**: Collapses to 6–43 dimensions. Linearizable (T-2), high local nonlinearity (T-7). The model squeezes out irrelevant embedding dimensions.
3. **Layers 6–15 — Distributed processing**: Expands to 120–178 dimensions. Layer 6 is the critical hub (T-2). Locally linear but globally nonlinear (T-7). Low Jacobian consistency — computation is highly input-dependent.
4. **Layers 16–24 — Second compression**: Collapses to 1.6–7.5 dimensions. The most extreme bottleneck in the network. Forces representations through an information bottleneck, discarding alternatives and committing to a specific interpretation. Globally nonlinear (T-2).
5. **Layers 25–34 — Output preparation**: Partial recovery to 10–78 dimensions with dramatic norm growth (139→568). High Jacobian consistency (T-7) — a more uniform, input-independent transformation. Linearizable (T-2). Late-layer gate_proj rank increases (T-9).
6. **Layer 35 — De-anisotropification**: Disperses representations (PR 119, cosine 0.09, norm ↓ to 391) and maximally separates semantic categories (separation 1.03). Universal discriminator (T-17). MLP-driven nonlinearity (T-7).

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
