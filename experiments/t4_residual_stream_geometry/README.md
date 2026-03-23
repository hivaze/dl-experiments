# T-4: Residual Stream Geometry Across Depth

## Motivation & Research Question

How does the geometry of the hidden-state manifold change as representations flow through transformer layers? Specifically:

1. **Effective dimensionality**: How many dimensions do representations actually use at each layer?
2. **Isotropy and anisotropy structure**: Are representations uniformly distributed in the space, or clustered in a low-dimensional cone? If anisotropic, is it intrinsic or removable?
3. **Token clustering**: Do tokens from semantically similar prompts converge as depth increases?

This connects to the Layer Shuffle Recovery experiment (on Qwen3-1.7B) — if activation flow fingerprints work well for ordering layers, there must be a predictable geometric transformation at each layer. Understanding the geometry explains *why* those fingerprints are so discriminative.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, hidden\_dim=2560)
- **Data**: 28 pre-generated completions from `data/text_completions/`, covering 7 categories (factual, reasoning, linguistic, code, world\_knowledge, technical, rare)
- **Tokens analyzed**: 2,277 completion tokens (prompt/template tokens excluded)
- **Layers**: 37 measurement points — embedding output + 36 transformer layer outputs
- **Hardware**: NVIDIA B200, bf16 inference, float32 metric computation
- **Runtime**: ~63 seconds total

## Mathematical Framework

### Participation Ratio as Effective Dimensionality

Given a set of $N$ token representations $\{\mathbf{h}_1, \ldots, \mathbf{h}_N\} \subset \mathbb{R}^d$, form the mean-centered data matrix $\mathbf{H} \in \mathbb{R}^{N \times d}$ with rows $\mathbf{h}_i - \bar{\mathbf{h}}$. The SVD gives singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r \ge 0$ where $r = \min(N, d)$. Define the normalized energy distribution:

$$p_i = \frac{\sigma_i^2}{\sum_{j=1}^{r} \sigma_j^2}$$

The **participation ratio** (inverse Simpson index of the energy distribution) is:

$$\text{PR} = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4} = \frac{1}{\sum_i p_i^2}$$

**Properties:**
- $\text{PR} = 1$ when all variance concentrates in a single direction ($p_1 = 1$)
- $\text{PR} = k$ when variance is uniformly distributed across exactly $k$ directions
- $\text{PR} = r$ when all singular values are equal (maximally isotropic)
- $\text{PR} = \exp(H_2)$ where $H_2 = -\ln\sum_i p_i^2$ is the Rényi entropy of order 2

The PR is more robust than rank (which is sensitive to numerical thresholds) and more interpretable than spectral entropy (which is unbounded). It directly answers: "how many orthogonal directions carry meaningful variance?"

### Isotropy via Cosine Similarity

For a set of vectors $\{\mathbf{h}_i\}$, the **mean pairwise cosine similarity** is:

$$\bar{c} = \mathbb{E}_{i \ne j}\left[\frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\| \cdot \|\mathbf{h}_j\|}\right]$$

For vectors drawn uniformly on $S^{d-1}$ (the unit sphere in $\mathbb{R}^d$), $\mathbb{E}[\bar{c}] = 0$ and $\text{Var}[\bar{c}] = \mathcal{O}(1/d)$. Any significant deviation from zero indicates anisotropy — the representations cluster in a sub-region of the sphere.

**Decomposition into mean-direction and intrinsic components.** Write $\mathbf{h}_i = \bar{\mathbf{h}} + \tilde{\mathbf{h}}_i$ where $\bar{\mathbf{h}} = \frac{1}{N}\sum_i \mathbf{h}_i$ is the centroid (so $\sum_i \tilde{\mathbf{h}}_i = \mathbf{0}$). The inner product decomposes as:

$$\langle \mathbf{h}_i, \mathbf{h}_j \rangle = \|\bar{\mathbf{h}}\|^2 + \langle \bar{\mathbf{h}},\, \tilde{\mathbf{h}}_i + \tilde{\mathbf{h}}_j \rangle + \langle \tilde{\mathbf{h}}_i, \tilde{\mathbf{h}}_j \rangle$$

When $\|\bar{\mathbf{h}}\| \gg \|\tilde{\mathbf{h}}_i\|$ for typical $i$, the first term dominates and:

$$\frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\|\|\mathbf{h}_j\|} \approx \frac{\|\bar{\mathbf{h}}\|^2}{\|\bar{\mathbf{h}}\|^2 + \mathbb{E}[\|\tilde{\mathbf{h}}\|^2]} \to 1$$

Computing $\bar{c}$ on the centered vectors $\tilde{\mathbf{h}}_i$ isolates the **intrinsic** anisotropy — any structure that remains after removing the shared mean direction.

### Spectral Flatness

The **spectral flatness** of the covariance eigenvalue spectrum $\{\lambda_i\}_{i=1}^{m}$ (where $m$ is the number of positive eigenvalues, with $\lambda_i = \sigma_i^2 / (N-1)$) is the ratio of geometric to arithmetic mean:

$$\text{SF} = \frac{\left(\prod_{i=1}^{m} \lambda_i\right)^{1/m}}{\frac{1}{m}\sum_{i=1}^{m} \lambda_i} = \frac{\exp\!\left(\frac{1}{m}\sum_{i=1}^{m} \ln \lambda_i\right)}{\frac{1}{m}\sum_{i=1}^{m} \lambda_i}$$

By the AM-GM inequality, $\text{SF} \in (0, 1]$, with equality to 1 iff all eigenvalues are equal (perfectly isotropic). A value near 0 indicates a "spiked" spectrum dominated by a few large eigenvalues.

**Relationship to PR.** Both PR and SF measure spectral concentration, but differently. PR counts how many directions carry meaningful variance (an effective count), while SF measures how uniformly the variance is spread (a shape statistic). Two distributions with the same PR can have different SF if their energy distributions have different shapes — e.g., a flat-then-zero spectrum and a smoothly decaying spectrum can yield the same PR but different SF.

### Cluster Separation

For tokens partitioned into $C$ categories, define:

$$\bar{c}_{\text{intra}} = \frac{1}{C} \sum_{c=1}^{C} \mathbb{E}_{i,j \in c,\, i \ne j}\!\left[\cos(\mathbf{h}_i, \mathbf{h}_j)\right], \qquad \bar{c}_{\text{inter}} = \mathbb{E}_{\substack{i \in c_1,\, j \in c_2 \\ c_1 \ne c_2}}\!\left[\cos(\mathbf{h}_i, \mathbf{h}_j)\right]$$

The **cluster separation ratio** is:

$$S = \frac{\bar{c}_{\text{intra}} - \bar{c}_{\text{inter}}}{|\bar{c}_{\text{inter}}| + \epsilon}$$

where $\epsilon = 10^{-10}$ prevents division by zero when inter-category similarity vanishes. $S > 0$ means tokens within a category are more similar to each other than to tokens in other categories. Note that $S$ is sensitive to the overall anisotropy level — in highly anisotropic layers where all cosines are large, $S$ can be small despite meaningful category structure (because both intra and inter are large).

## Methods

### 1. Hidden State Extraction
Forward-hook-based extraction at every layer, identical to T-1. For each of 28 prompts, we register hooks on all 36 `model.model.layers[i]` modules plus capture the embedding output. Only completion-token positions are retained (positions $\ge$ `prompt_token_count`), yielding 2,277 token vectors of dimension 2,560 per layer.

### 2. Effective Dimensionality (Participation Ratio)
For each layer's pooled matrix $\mathbf{H} \in \mathbb{R}^{2277 \times 2560}$, we mean-center and compute SVD. The participation ratio is computed from the squared singular values as defined above.

### 3. Isotropy and Anisotropy Decomposition
Three complementary measures, computed on both raw and mean-centered representations:

- **Mean pairwise cosine similarity**: Sample 5,000 random token pairs and compute their cosine similarity. For isotropic representations this should be ~0; for anisotropic (cone-shaped) distributions, it will be high. Computing on **centered** representations (mean-subtracted) isolates intrinsic geometry from the shared-direction component.
- **Spectral flatness**: Computed on the eigenvalues of the sample covariance matrix ($\sigma_i^2 / (N-1)$), ranging from 0 (one dominant direction) to 1 (perfectly isotropic).
- **Norm statistics**: Track how representation magnitudes (L2 norms) evolve across depth.

### 4. Token Clustering by Category
Tokens are grouped by their prompt's category. We compute:

- **Intra-category similarity**: Mean cosine similarity between random token pairs *within* the same category
- **Inter-category similarity**: Mean cosine similarity between tokens from *different* categories
- **Cluster separation ratio**: $(c_{\text{intra}} - c_{\text{inter}}) / (|c_{\text{inter}}| + \epsilon)$ — higher means categories are more distinguishable
- **Centroid analysis**: Cosine similarity between category centroids at each layer

## Results

### Effective Dimensionality

![Geometry Overview — dimensionality, isotropy, anisotropy decomposition, and norms across layers](results/geometry_overview.png)

| Region | Layers | PR Range | Interpretation |
|--------|--------|----------|----------------|
| Embedding | emb | 67.7 | Moderate — token embeddings span ~68 effective dimensions |
| Early spike | 0 | 112.1 | First layer *increases* dimensionality |
| Early collapse | 1–5 | 6.2–42.7 | Dramatic collapse — layers 1–2 reduce to just ~6 effective dims |
| Mid recovery | 6–15 | 120–178 | Gradual recovery to ~130–178 effective dims |
| Deep collapse | 16–24 | 1.6–7.5 | Second major collapse — PR drops to **1.6** at layer 16 |
| Late recovery | 25–34 | 10–78 | Partial recovery through final layers |
| Final layer | 35 | 118.7 | Sharp dimensionality expansion at the output |

The **bimodal collapse pattern** (layers 1–5 and 16–24) is striking. The deep collapse at layer 16 (PR=1.6) means representations are nearly one-dimensional — almost all variance sits on a single axis. This suggests two critical "bottleneck" regions where the model compresses information maximally before expanding it again.

### Isotropy / Anisotropy

The raw mean cosine similarity hovers around **0.38–0.52** throughout most layers (moderately anisotropic), rising to **0.57** at layer 33 and peaking at **0.62** at layer 34 before dramatically dropping to **0.09** at the final layer 35.

The **centered** cosine similarity is essentially **zero** (~−0.002 to +0.009) at every layer. This is a key finding: **all observed anisotropy is due to a shared mean direction, not intrinsic geometric clustering**. After removing the mean, token representations are nearly perfectly isotropic at every layer.

Spectral flatness follows a non-monotonic trajectory: the embedding starts high (0.43), drops sharply at layer 0 (0.03), then generally increases through the network — reaching 0.09–0.13 in layers 6–15, dipping back to 0.02 during the deep collapse (layers 16–17), and recovering to 0.12–0.13 by layers 30–35. The pattern tracks the PR trajectory: collapse regions have low SF (spiked spectra), recovery regions have higher SF (more distributed spectra).

### Representation Norms

Norms grow **superlinearly** across depth:
- Embedding: 0.99
- Layer 10: 40.3
- Layer 20: 68.2
- Layer 30: 280.9
- Layer 34: 567.8

Combined with high raw cosine similarity, this confirms the **anisotropic cone** pattern: representations extend along an increasingly narrow cone with growing norms. However, the final layer (35) breaks this pattern — norms drop to 390.5 and cosine similarity plummets to 0.09, indicating the final layer disperses representations for the LM head projection.

### Token Clustering

![Token Clustering — intra/inter-category similarity, separation ratio, and centroid heatmap](results/token_clustering.png)

| Layer | Intra-cat | Inter-cat | Separation |
|-------|-----------|-----------|------------|
| emb   | 0.098     | 0.078     | 0.26       |
| 0     | 0.465     | 0.452     | 0.03       |
| 11    | 0.470     | 0.412     | 0.14       |
| 22    | 0.430     | 0.368     | 0.17       |
| 27    | 0.432     | 0.357     | **0.21**   |
| 34    | 0.638     | 0.601     | 0.06       |
| 35    | 0.153     | 0.075     | **1.03**   |

Category separation generally increases through the middle layers, from near-zero at layer 0 to a peak of 0.21 at layer 27 (with minor fluctuations — e.g., a dip from 0.12 at L15 to 0.11 at L16). However, **separation reverses in layers 28–34**: as norms grow superlinearly and raw cosine similarity climbs to 0.62, both intra- and inter-category similarity rise together, compressing the gap and dropping separation back to 0.06 at layer 34. The final layer 35 then produces a dramatic jump to **1.03** — the dispersal of representations lets category structure become the dominant organizing principle, with intra-category similarity dropping to 0.15 while inter-category drops further to 0.08.

The centroid cosine heatmap at layer 35 reveals that `code` tokens are most distinctive (lowest cosine with other categories), while `factual` and `world_knowledge` remain similar to each other.

### Singular Value Spectra

![Singular Value Spectra — SV heatmap and cumulative variance curves](results/singular_value_spectra.png)

![Variance Explained — top-k variance explained across layers](results/variance_explained.png)

The cumulative variance plots reveal:
- At layers 1–2 (early collapse), the top-1 SV explains **~40%** of variance; at layers 3–5 the dominance decreases (31%→22%→14%) as PR recovers
- At layers 16–24 (deep collapse), the top-1 SV explains **36–79%** of variance, peaking at layer 16 (79%) and decreasing monotonically as PR slowly recovers through this region
- At mid layers (6–15), variance is more distributed — top-1 explains only 3–4%, and top-10 explains ~25–30%
- The final layer 35 has one of the most distributed spectra — top-1 explains only 5%, with top-50 needed for ~72% variance

The SV heatmap shows a clear "bright band" at singular value index 0 during collapse layers, confirming the single-dominant-direction interpretation.

## Conclusions & Key Findings

1. **Bimodal dimensionality collapse**: Two distinct bottleneck regions (layers 1–5, PR 6–43; layers 16–24, PR 1.6–7.5) where representations collapse to near-one-dimensional manifolds. These are sharp phase transitions, not gradual, and bookend a high-dimensional processing plateau (layers 6–15, PR 120–178).

2. **All anisotropy is mean-direction anisotropy**: Centered cosine similarity is ~0 everywhere. The high raw cosine (~0.4–0.6) is entirely explained by a shared mean direction — mean-centering produces near-isotropic representations at every layer.

3. **Superlinear norm growth with final-layer correction**: Norms grow from 1.0 to 568, but layer 35 reduces norms to 391 and increases isotropy (cosine 0.62→0.09). Layer 35 acts as a dispersal mechanism for the vocabulary projection.

4. **Non-monotonic category separation**: Separation increases through mid-layers (peak 0.21 at L27), reverses in layers 28–34 as the anisotropic cone dominates, then spikes to 1.03 at layer 35 when dispersal lets category structure emerge. Code tokens are most distinctive at layer 35.

5. **Connection to Layer Shuffle Recovery**: The sharp geometric transitions at each layer (distinct PR, norm, SV spectrum) explain why activation fingerprints are discriminative for layer ordering — shuffling disrupts these predictable signatures. (The shuffle experiment uses Qwen3-1.7B with 28 layers; the specific layer boundaries likely differ.)

## Cross-Experiment Connections

Throughout this section, T-4 geometric data (PR, norms, SV spectra) are as reported in the Results tables above. Only cross-experiment findings and their geometric interpretations are stated here.

### T-2 (Layer Knockout) + T-7 (Linearization Gap): Geometry Explains Criticality and Linearizability

T-2 found that layer 0 is catastrophically critical (99.6× loss ratio). T-7 showed it is also linearizable (99.1% recovery with a full-rank linear map). T-4 explains why both are true: layer 0 is the only layer that *increases* dimensionality (PR: 68→112) and dramatically amplifies norms (1.0→7.7). This geometric expansion is a specific, learnable linear transformation — which is why linear replacement works but removal is catastrophic.

T-7's "bimodal linearity" finding maps onto T-4's geometry:

| T-4 Geometry Region | Layers | PR | T-7 Linear Replacement |
|---|---|---|---|
| Early collapse (low PR) | 1–5 | 6–43 | 91–99% recovery |
| Mid recovery (high PR) | 6–15 | 120–178 | Mixed: L6 recovers (87%), L8–L12 fail (−2631% to −1067%) |
| Deep collapse (low PR) | 16–24 | 1.6–7.5 | Fails: −1796% to −101% |
| Late recovery | 25–34 | 10–78 | L32–L33 recover (87–88%), **L34 fails (−45%)** |

The general trend: **low-PR regions tend to be linearizable, high-PR regions tend to resist linear replacement**. When representations are squeezed into 1–6 effective dimensions, the layer's transformation is constrained enough to be captured by a fixed linear map. When representations span 120–178 dimensions, the transformation is input-dependent and globally nonlinear. However, the mapping is not strict — layer 6 (PR=170) achieves 87% recovery despite high dimensionality, and layer 34 (PR=53) fails at −45% despite moderate dimensionality.

T-2 also found layer 6 is a computational hub (2nd most critical at 21.7×, appears in 4/5 top synergistic pairs). In T-4, layer 6 sits at the onset of the mid-recovery plateau (PR jumps from 43 at layer 5 to 170 at layer 6) — it marks the transition from compressed to distributed representation.

### T-7 (Linearization Gap): Local Smoothness vs Global Geometry

T-7 found a U-shaped nonlinearity profile with middle layers (6–18) being most locally linear (gap ~0.13–0.15) but globally nonlinear (catastrophic replacement failure). T-7's Jacobian consistency data resolves this paradox when combined with T-4's geometry:

- **Layers 6–8**: Low Jacobian consistency (0.55–0.66) despite low linearization gap. The Jacobian is smooth at each operating point but *varies strongly across inputs*. These layers have high PR (170–178) — the many active dimensions create room for input-dependent behavior. (Layers 4–5 also have low consistency but lower PR of 20–43, so the PR-consistency connection is not purely about dimensionality.)
- **Layers 29–35**: High consistency (0.76–0.88), meaning the Jacobian is nearly the same regardless of input. Norms are largest here (235–568), and the late-layer norm growth dominates over input-dependent variation.

T-7 also found that MLP nonlinearity drives the late-layer spike (MLP gap 0.24 at layer 35). The final-layer dispersal (PR: 53→119, cosine: 0.62→0.09) requires nonlinear transformation that SwiGLU provides.

### T-9 (Weight Spectral Structure): Weight Rank vs Representation Dimensionality

T-9 found no significant correlation between weight effective rank and representation geometry (r = 0.157, p = 0.36). T-4 explains why: the residual stream is an **accumulated sum** of all upstream layer contributions:

$$\mathbf{h}^{(\ell)} = \mathbf{h}^{(0)} + \sum_{i=1}^{\ell} f_i\!\left(\mathbf{h}^{(i-1)}\right)$$

A single layer's weight rank constrains only its *incremental update* $f_i$, not the total dimensionality of $\mathbf{h}^{(\ell)}$. The bimodal collapse is driven by dominant singular values in the *accumulated representations* (top-1 SV explains up to 79% of variance at layer 16), which can emerge from accumulation dynamics regardless of individual weight ranks.

T-9's finding that Q/K routing matrices are lower-rank (0.31 effective rank) than V/O value matrices (0.52) has a geometric interpretation: routing (deciding where to attend) operates in a lower-dimensional subspace, while value extraction must manipulate the full representation geometry.

T-9 also found that gate\_proj effective rank increases in late layers (+0.166, p < 0.0001). In T-4, late layers show partial PR recovery (10–78) with massive norm growth, which is consistent with increased gating capacity being associated with expanding, high-magnitude representations.

### T-17 (Contrastive Trajectories): Geometry of Divergence

T-17 found that **shared-prefix** antonym pairs (e.g., "The Sun is a star" vs "The Sun is a planet") maintain high cosine similarity (>0.63) across layers 2–34, with a dip at layer 21. During the deep collapse (PR 1.6–7.5), two sequences processing different tokens are squeezed onto a near-one-dimensional manifold — their cosine similarity remains high because there is essentially only one direction available. The dip at layer 21 (PR=2.6) coincides with the beginning of PR recovery within the collapse region, where enough dimensions re-emerge to begin separating the antonyms.

T-17 also found a two-phase processing structure: antonym KL > synonym KL in layers 0–4 (semantic discrimination), then synonym KL > antonym KL in layers 5–35 (form commitment). A brief reversal at layers 21–23 (antonym > synonym) coincides with the deep collapse region, suggesting the dimensionality bottleneck disrupts the established processing order.

T-17's finding that layer 35 is a "universal discriminator" (all cosine similarities drop sharply) aligns directly with T-4's final-layer dispersal — the expansion to PR 119 and cosine drop to 0.09 maximally separates all token alternatives for the LM head.

### T-1 (Logit Lens): Prediction Quality vs Geometric Phase

T-1 found a four-phase architecture in prediction quality: representation building (L0–12, <1% accuracy), early semantics (L13–21), prediction formation (L22–28), and refinement (L29–35, reaching 99.5%). T-4's geometric phases provide a structural explanation:

- **L0–12 (<1% accuracy)**: Spans the early collapse and mid recovery. Representations are being geometrically restructured — first compressed, then expanded. Predictions are poor because the representation space has not settled into a stable structure.
- **L22–28 (rapid accuracy climb)**: Spans the deep collapse and beginning of late recovery. The extreme dimensionality bottleneck (PR 1.6–7.5) constrains representations to so few dimensions that ambiguity is eliminated — prediction accuracy climbs fastest here.
- **L35 (99.5% accuracy)**: The dispersal (PR 119, cosine 0.09) creates a well-separated landscape where the LM head can cleanly select the correct token.

T-1 also found that layer 0 actively *destroys* embedding predictions (mean rank 42K→77K), which aligns with the geometric expansion at layer 0 — the dimensionality and norm restructuring completely reorganizes the representational space.

### T-3 (Layer Swap Cost): Geometry Predicts Swap Tolerance

T-3 found that the cheapest swaps are exclusively adjacent late-middle pairs (layers 25–33, Δloss 0.06–0.10), while early layers (0–2) and layer 35 produce the costliest swaps (Δloss 10–23). T-4's geometry explains this:

- **Cheapest swaps (L25–L33)**: Late recovery region where PR and norms change gradually between adjacent layers — similar geometric signatures make swapping minimally disruptive.
- **Costliest swaps (L0–L2, L35)**: Layer 0 performs a unique geometric expansion; layer 35 performs a unique dispersal. These are geometrically singular transformations with no substitute.
- **T-3's 3-zone clustering** (early [0–6], middle [7–30], late [31–35]) aligns with T-4's geometric regimes, with the early/middle boundary at the dimensionality explosion (layer 6, PR jumps to 170) and the middle/late boundary near the end of the deep collapse recovery.

### Synthesis: The Geometric Processing Pipeline

Combining T-4 with other experiments reveals a coherent six-phase pipeline:

| Phase | Layers | PR | Key Geometry | Cross-Experiment Role |
|---|---|---|---|---|
| **Geometric expansion** | 0 | 68→112 | Norm 1→7.7 | Critical (99.6× knockout, T-2), linearizable (99.1%, T-7), destroys embedding predictions (T-1) |
| **First compression** | 1–5 | 6–43 | Top-1 SV 14–40% | Linearizable (91–99%, T-7) |
| **Distributed processing** | 6–15 | 120–178 | Top-1 SV 3–4% | Hub at L6 (21.7× knockout, T-2), locally linear but globally nonlinear (T-7), low Jacobian consistency (0.55–0.70) |
| **Second compression** | 16–24 | 1.6–7.5 | Top-1 SV 36–79% | Prediction accuracy begins steep climb (T-1), globally nonlinear (T-7) |
| **Output preparation** | 25–34 | 10–78 | Norm 139→568 | Cheapest swap region (T-3), high Jacobian consistency (0.76–0.88, T-7), gate\_proj rank increases (T-9) |
| **Dispersal** | 35 | 119 | Cosine 0.62→0.09, norm ↓391 | Universal discriminator (T-17), MLP-driven nonlinearity (T-7), 99.5% accuracy (T-1) |

## Practical & Architectural Implications

The following implications are **hypotheses grounded in T-4 observations and cross-experiment patterns**, not experimentally validated optimizations. They identify specific, testable predictions for future work.

### 1. Layer Pruning & Structured Distillation

The two compression bottlenecks pass information through as few as 1–2 effective dimensions, making intermediate layers within each bottleneck candidates for pruning or merging:

- **Layers 17–23**: PR stays below 7.5, top-1 SV explains 36–79% of variance, and downstream layers tolerate perturbation (T-3: adjacent swaps at Δloss 0.06–0.10). *Hypothesis*: these 7 layers could be replaced with 2–3 layers or a single linear projection capturing the dominant singular direction.
- **Layers 1–4**: Similarly low-dimensional (PR 6–43) and confirmed linearizable by T-7 (91–99% recovery). *Hypothesis*: these could be collapsed into a single learned linear map.
- **Do not prune layer 0, layer 6, or layer 35.** T-2 knockout ratios (99.6×, 21.7×) and T-4's unique geometric signatures at these layers indicate they perform irreplaceable transformations.

### 2. Quantization Strategy by Geometric Phase

Different geometric regimes have different precision profiles:

- **Bottleneck layers (1–5, 16–24)**: Variance concentrates on 1–7 axes, so signal-to-noise ratio is inherently high. *Hypothesis*: INT4 or lower should preserve the dominant singular direction with minimal degradation.
- **Distributed processing layers (6–15)**: PR 120–178 with low Jacobian consistency (T-7: 0.55–0.70) means computation is input-dependent across many dimensions. *Hypothesis*: these layers are most sensitive to quantization noise and should receive higher precision (INT8/FP8).
- **Late recovery layers (25–34)**: High Jacobian consistency (T-7: 0.76–0.88) suggests more uniform transformations tolerant of moderate quantization, though large norms (139–568) require careful scaling factor selection.
- **Layer 35**: The dispersal involves precise norm reduction and cosine collapse. *Hypothesis*: this benefits from higher precision (FP8/INT8 minimum).

A **mixed-precision scheme** aligned to geometric phases could outperform uniform quantization at the same average bit-width.

### 3. KV-Cache Compression

Low effective dimensionality in bottleneck regions implies key/value vector redundancy. Combined with T-9's finding that Q/K matrices are lower-rank (0.31) than V/O matrices (0.52):

- *Hypothesis*: **Layers 16–24** KV-cache entries could be compressed via low-rank projection or aggressive quantization, since representations at these layers project onto essentially one direction (PR 1.6–7.5).
- **Layers 6–15** require fuller KV-cache fidelity — high-dimensional representations (PR 120–178) carry input-dependent routing information.

### 4. Feature Extraction & Embedding Selection

For downstream tasks requiring sentence/token embeddings:

- **Always mean-center** representations. Since all anisotropy is mean-direction anisotropy (centered cosine ≈ 0), mean-centering produces near-isotropic embeddings better suited for cosine-similarity-based retrieval.
- **Layer 27** has peak category separation (0.21) before the late-layer anisotropic collapse, with moderate dimensionality (PR ≈ 21) and norms — a natural choice for semantic embeddings.
- **Layer 35** has maximum category separation (1.03) and near-isotropic geometry, but is optimized for next-token prediction.
- **Avoid layers 16–24** — PR 1.6–7.5 means representations are too compressed to preserve fine-grained distinctions.

### 5. Early Exit & Adaptive Computation

The geometric pipeline suggests natural exit points:

- **After layer 24** (end of deep collapse): The bottleneck has constrained representations — prediction accuracy climbs steeply through this region (T-1). An exit head here could capture tokens that resolved during compression.
- **The cost of skipping layer 35 is high**: any early-exit head at earlier layers must compensate for the anisotropic cone geometry (high cosine, large norms) — requiring a learned dispersal pre-projection rather than a simple linear head.

### 6. Architecture Design Observations

- **The bimodal bottleneck pattern may be functional**: the two compression phases mirror the information bottleneck principle — compress to discard irrelevant features, then expand to build task-relevant representations. Architectures with *explicit* bottleneck layers (reduced hidden dims or low-rank constraints) might achieve similar compression more parameter-efficiently. Whether this improves over standard uniform-width architectures remains to be tested.
- **Final-layer dispersal appears necessary for the LM head**: applying the LM head to layer 34 representations (cosine 0.62) instead of layer 35 (cosine 0.09) would face high inter-token similarity, likely degrading discrimination. Architectures that share or skip the final layer would need an alternative dispersal mechanism.
- **Norm management**: RMSNorm controls per-layer scale but does not prevent the superlinear norm accumulation (1→568) in the residual stream. Architectures with explicit norm management (residual scaling factors, periodic norm resets) might reduce the need for the final-layer correction.

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
