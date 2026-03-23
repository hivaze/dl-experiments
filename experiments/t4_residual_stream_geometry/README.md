# T-4: Residual Stream Geometry Across Depth

## Motivation & Research Question

How does the geometry of the hidden-state manifold change as representations flow through transformer layers? Specifically:

1. **Effective dimensionality**: How many dimensions do representations actually use at each layer?
2. **Isotropy**: Are representations uniformly distributed in the space, or clustered in a low-dimensional cone?
3. **Token clustering**: Do tokens from semantically similar prompts converge as depth increases?
4. **Anisotropy collapse**: Do representations become increasingly anisotropic in later layers?

This connects to the Layer Shuffle Recovery experiment (on Qwen3-1.7B) — if activation flow fingerprints work well for ordering layers, there must be a predictable geometric transformation at each layer. Understanding the geometry explains *why* those fingerprints are so discriminative. While the shuffle experiment uses a different model (1.7B, 28 layers), the geometric principles — smooth progression of norms, spectral structure, and per-channel residual signatures — should generalize across models with homogeneous decoder architectures.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, hidden\_dim=2560)
- **Data**: 28 pre-generated completions from `data/text_completions/`, covering 7 categories (factual, reasoning, linguistic, code, world\_knowledge, technical, rare)
- **Tokens analyzed**: 2,277 completion tokens (prompt/template tokens excluded)
- **Layers**: 37 measurement points — embedding output + 36 transformer layer outputs
- **Hardware**: NVIDIA B200, bf16 inference, float32 metric computation
- **Runtime**: ~63 seconds total

## Mathematical Framework

### Participation Ratio as Effective Dimensionality

Given a set of $N$ token representations $\{\mathbf{h}_1, \ldots, \mathbf{h}_N\} \subset \mathbb{R}^d$, form the mean-centered data matrix $\mathbf{H} \in \mathbb{R}^{N \times d}$ with rows $\mathbf{h}_i - \bar{\mathbf{h}}$. The SVD gives singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(N,d)} \ge 0$. Define the normalized energy distribution:

$$p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}$$

The **participation ratio** (inverse Simpson index of the energy distribution) is:

$$\text{PR} = \frac{\left(\sum_i \sigma_i^2\right)^2}{\sum_i \sigma_i^4} = \frac{1}{\sum_i p_i^2}$$

**Properties:**
- $\text{PR} = 1$ when all variance concentrates in a single direction ($p_1 = 1$)
- $\text{PR} = k$ when variance is uniformly distributed across exactly $k$ directions
- $\text{PR} = \min(N, d)$ when all singular values are equal (maximally isotropic)
- $\text{PR} = \exp(2H_2)$ where $H_2 = -\log\sum_i p_i^2$ is the Rényi entropy of order 2

The PR is more robust than rank (which is sensitive to numerical thresholds) and more interpretable than spectral entropy (which is unbounded). It directly answers: "how many orthogonal directions carry meaningful variance?"

### Isotropy via Cosine Similarity

For a set of vectors $\{\mathbf{h}_i\}$, the **mean pairwise cosine similarity** is:

$$\bar{c} = \mathbb{E}_{i \ne j}\left[\frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\| \cdot \|\mathbf{h}_j\|}\right]$$

For vectors drawn uniformly on $S^{d-1}$ (the unit sphere in $\mathbb{R}^d$), $\mathbb{E}[\bar{c}] = 0$ and $\text{Var}[\bar{c}] = \mathcal{O}(1/d)$. Any significant deviation from zero indicates anisotropy — the representations cluster in a sub-region of the sphere.

**Decomposition into mean-direction and intrinsic components.** Write $\mathbf{h}_i = \bar{\mathbf{h}} + \tilde{\mathbf{h}}_i$ where $\bar{\mathbf{h}}$ is the centroid. The cosine similarity decomposes as:

$$\frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\|\|\mathbf{h}_j\|} = \frac{\|\bar{\mathbf{h}}\|^2 + \langle \bar{\mathbf{h}}, \tilde{\mathbf{h}}_i + \tilde{\mathbf{h}}_j \rangle + \langle \tilde{\mathbf{h}}_i, \tilde{\mathbf{h}}_j \rangle}{\|\mathbf{h}_i\|\|\mathbf{h}_j\|}$$

When $\|\bar{\mathbf{h}}\| \gg \|\tilde{\mathbf{h}}_i\|$, the first term dominates and the cosine is approximately $\|\bar{\mathbf{h}}\|^2 / (\|\bar{\mathbf{h}}\|^2 + \|\tilde{\mathbf{h}}\|^2) \approx 1$. Computing $\bar{c}$ on the centered vectors $\tilde{\mathbf{h}}_i$ isolates the **intrinsic** anisotropy — any structure that remains after removing the shared mean direction.

### Spectral Flatness

The **spectral flatness** of the covariance eigenvalue spectrum $\{\lambda_i\}$ is the ratio of geometric to arithmetic mean:

$$\text{SF} = \frac{\left(\prod_i \lambda_i\right)^{1/k}}{\frac{1}{k}\sum_i \lambda_i} = \frac{\exp\left(\frac{1}{k}\sum_i \log \lambda_i\right)}{\frac{1}{k}\sum_i \lambda_i}$$

where $k$ is the number of positive eigenvalues. By the AM-GM inequality, $\text{SF} \in [0, 1]$, with equality to 1 iff all eigenvalues are equal (perfectly isotropic). A value near 0 indicates a "spiked" spectrum dominated by a few large eigenvalues. Unlike PR (which measures how many directions matter), spectral flatness measures how uniformly the variance is spread — two distributions with the same PR can have different spectral flatness if their energy distributions have different shapes.

### Cluster Separation

For tokens partitioned into $C$ categories, define:

$$\bar{c}_{\text{intra}} = \frac{1}{C} \sum_{c=1}^{C} \mathbb{E}_{i,j \in c}\left[\cos(\mathbf{h}_i, \mathbf{h}_j)\right], \qquad \bar{c}_{\text{inter}} = \mathbb{E}_{i \in c_1, j \in c_2, c_1 \ne c_2}\left[\cos(\mathbf{h}_i, \mathbf{h}_j)\right]$$

The **cluster separation ratio** is:

$$S = \frac{\bar{c}_{\text{intra}} - \bar{c}_{\text{inter}}}{|\bar{c}_{\text{inter}}|}$$

$S > 0$ means tokens within a category are more similar to each other than to tokens in other categories. $S = 0$ means no category structure. Note that $S$ is sensitive to the overall anisotropy level — in highly anisotropic layers where all cosines are large, $S$ can be small despite meaningful category structure (because both intra and inter are large).

## Methods

### 1. Hidden State Extraction
Forward-hook-based extraction at every layer, identical to T-1. For each of 28 prompts, we register hooks on all 36 `model.model.layers[i]` modules plus capture the embedding output. Only completion-token positions are retained (positions $\ge$ `prompt_token_count`), yielding 2,277 token vectors of dimension 2,560 per layer.

### 2. Effective Dimensionality (Participation Ratio)
For each layer's pooled matrix $\mathbf{H} \in \mathbb{R}^{2277 \times 2560}$, we mean-center and compute SVD. The participation ratio is computed from the squared singular values as defined above.

### 3. Isotropy Metrics
Two complementary measures:

- **Mean pairwise cosine similarity**: Sample 5,000 random token pairs and compute their cosine similarity. For isotropic representations this should be ~0; for anisotropic (cone-shaped) distributions, it will be high.
- **Spectral flatness**: Computed on the eigenvalues of the sample covariance matrix ($\sigma_i^2 / (N-1)$), ranging from 0 (one dominant direction) to 1 (perfectly isotropic).

We also compute cosine similarity on **centered** representations (mean-subtracted), which isolates intrinsic geometry from the shared-direction component.

### 4. Token Clustering by Category
Tokens are grouped by their prompt's category. We compute:

- **Intra-category similarity**: Mean cosine similarity between random token pairs *within* the same category
- **Inter-category similarity**: Mean cosine similarity between tokens from *different* categories
- **Cluster separation ratio**: $(c_{\text{intra}} - c_{\text{inter}}) / |c_{\text{inter}}|$ — higher means categories are more distinguishable
- **Centroid analysis**: Cosine similarity between category centroids at each layer

### 5. Anisotropy Decomposition
We decompose anisotropy into removable (mean-direction) and intrinsic components:

- **Raw cosine similarity** captures total anisotropy
- **Centered cosine similarity** captures intrinsic anisotropy after removing the shared mean direction
- **Norm statistics** track how representation magnitudes evolve across depth

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

Category separation generally increases through the middle layers, from near-zero at layer 0 to a peak of 0.21 at layer 27 (with minor fluctuations — e.g., a small dip from 0.12 at L15 to 0.11 at L16). However, **separation reverses in layers 28–34**: as norms grow superlinearly and raw cosine similarity climbs to 0.62, both intra- and inter-category similarity rise together, compressing the gap and dropping separation back to 0.06 at layer 34. The final layer 35 then produces a dramatic jump to **1.03** — the model's de-anisotropification disperses representations enough for category structure to become the dominant organizing principle, with intra-category similarity dropping to 0.15 while inter-category drops further to 0.08.

The centroid cosine heatmap at layer 35 reveals that `code` tokens are most distinctive (lowest cosine with other categories), while `factual` and `world_knowledge` remain similar to each other.

### Singular Value Spectra

![Singular Value Spectra — SV heatmap and cumulative variance curves](results/singular_value_spectra.png)

![Variance Explained — top-k variance explained across layers](results/variance_explained.png)

The cumulative variance plots reveal:
- At layers 1–2 (early collapse), the top-1 SV explains **~40%** of variance; at layers 3–5 the dominance decreases (31%→14%) as PR recovers
- At layers 16–24 (deep collapse), the top-1 SV explains **36–79%** of variance, peaking at layer 16 (79%) and decreasing monotonically as PR slowly recovers through this region
- At mid layers (6–15), variance is more distributed — top-1 explains only 3–4%, and top-10 explains ~25–30%
- The final layer 35 has one of the most distributed spectra — top-1 explains only 5%, with top-50 needed for ~72% variance

The SV heatmap shows a clear "bright band" at singular value index 0 during collapse layers, confirming the single-dominant-direction interpretation.

## Conclusions & Key Findings

1. **Bimodal dimensionality collapse**: The model has two distinct bottleneck regions (layers 1–5 and 16–24) where representations collapse to near-one-dimensional manifolds (PR as low as 1.6). This is not gradual — it's a sharp phase transition. The two collapse regions bookend the "mid recovery" plateau (layers 6–15, PR 120–178) where representations are high-dimensional and distributed.

2. **All anisotropy is mean-direction anisotropy**: Centered cosine similarity is ~0 everywhere (range −0.002 to +0.009). The high raw cosine similarity (~0.4–0.6) is entirely explained by a shared mean direction. Remove it, and representations are isotropic. This has practical implications — mean-centering is sufficient to "fix" anisotropy for downstream tasks like retrieval or clustering.

3. **Superlinear norm growth with final-layer correction**: Norms grow from 1.0 to 568 across 35 layers, but the final layer reduces norms to 391 and dramatically increases isotropy (raw cosine drops from 0.62 to 0.09). Layer 35 acts as a "de-anisotropifier" that disperses representations for the vocabulary projection. This is consistent with the unembedding matrix expecting more isotropic input.

4. **Non-monotonic category separation with final-layer spike**: Token clustering by semantic category increases through the middle layers (separation 0.03 at L0 → 0.21 at L27), but then reverses in layers 28–34 as the anisotropic cone dominates (both intra and inter similarity rise, compressing the gap to 0.06). The final layer breaks this by dispersing representations, producing a dramatic separation spike to 1.03. Code tokens are most distinctive at layer 35 (lowest cosine with other categories), while factual and world\_knowledge remain similar.

5. **Connection to Layer Shuffle Recovery**: The sharp geometric transitions at layers 1–5 and 16–24 explain why activation fingerprints are so discriminative for layer ordering — each layer has a distinct geometric signature (PR, norm, SV spectrum) that changes predictably across depth. Shuffling layers would disrupt these signatures, making mis-ordering detectable. (Note: the shuffle experiment uses Qwen3-1.7B with 28 layers; the geometric principles should transfer but the specific layer boundaries may differ.)

## Cross-Experiment Connections

### T-2 (Layer Knockout) + T-7 (Linearization Gap): Geometry Explains Criticality and Linearizability

T-2 found that layer 0 is catastrophically critical (99.6× loss ratio). T-7 showed it is also linearizable (99.1% recovery with a full-rank linear map). T-4 explains why both are true: layer 0 is the only layer that *increases* dimensionality (PR: 67.7 → 112.1) and dramatically increases norms (1.0 → 7.7). It performs a massive geometric expansion — projecting 68-dimensional token embeddings into a 112-dimensional working space with 7.7× norm amplification. This is a specific, learnable linear transformation, which is why linear replacement works but removal is catastrophic.

T-7's "bimodal linearity" finding — early and late layers are globally linearizable while middle layers (8–19) are not — maps onto T-4's geometry as follows:

| T-4 Geometry Region | Layers | PR | T-7 Linear Replacement |
|---|---|---|---|
| Early collapse (low PR) | 1–5 | 6–43 | 91–99% recovery |
| Mid recovery (high PR) | 6–15 | 120–178 | Mixed: L6 recovers (87%), L8–L12 fail (−2631% to −1067%) |
| Deep collapse (low PR) | 16–24 | 1.6–7.5 | Fails: −1796% to −101% |
| Late recovery | 25–34 | 10–78 | L32–L33 recover (87–88%), **L34 fails (−45%)** |

The general trend holds: **low-dimensionality regions tend to be linearizable, high-dimensionality regions tend to resist linear replacement**. When representations are squeezed into 1–6 effective dimensions, the layer's transformation is constrained enough to be captured by a fixed linear map. When representations span 120–178 dimensions, the transformation is input-dependent and globally nonlinear. However, the mapping is not strict — layer 6 (PR=170) is an exception that achieves 87% recovery despite high dimensionality, and layer 34 (PR=53) fails at −45% despite moderate dimensionality.

T-2 also found layer 6 is a computational hub (2nd most critical at 21.7×, appears in 4/5 top synergistic pairs). In T-4, layer 6 sits at the onset of the mid-recovery plateau (PR jumps from 42.7 at layer 5 to 170.3 at layer 6) — it's the layer where dimensionality first explodes, marking the transition from compressed to distributed representation.

### T-7 (Linearization Gap): Local Smoothness vs Global Geometry

T-7 found a U-shaped nonlinearity profile with middle layers (6–18) being most locally linear (gap ~0.13–0.15) but globally nonlinear (catastrophic replacement failure). T-7's Jacobian consistency data resolves this paradox when combined with T-4's geometry:

- **Layers 6–8**: Low Jacobian consistency (0.55–0.66) despite low linearization gap. The Jacobian is smooth at each operating point but *varies strongly across inputs*. T-4 shows these layers have high PR (170–178) — the many active dimensions create room for input-dependent behavior. (Note: layers 4–5 also have low consistency but lower PR of 20–43, so the PR-consistency connection is not purely about dimensionality.)
- **Layers 29–35**: High consistency (0.76–0.88), meaning the Jacobian is nearly the same regardless of input. T-4 shows norms are largest here (235–568), and the late-layer norm growth dominates over input-dependent variation, making the transformation effectively fixed.

The geometry also explains T-7's finding that MLP nonlinearity drives the late-layer spike (MLP gap 0.24 at layer 35): the final-layer de-anisotropification (PR: 53 → 119, cosine: 0.62 → 0.09 from L34 to L35) requires nonlinear dispersal that SwiGLU provides.

### T-9 (Weight Spectral Structure): Weight Rank vs Representation Dimensionality

T-9 found no significant correlation between weight effective rank and representation geometry (r = 0.157, p = 0.36). This is initially surprising — shouldn't low-rank weights produce low-dimensional representations? T-4 explains why not: the residual stream is an **accumulated sum** of all upstream layer contributions:

$$\mathbf{h}^{(\ell)} = \mathbf{h}^{(0)} + \sum_{i=1}^{\ell} f_i(\mathbf{h}^{(i-1)})$$

A single layer's weight rank constrains only its *incremental update* $f_i$ to the residual stream, not the total dimensionality of $\mathbf{h}^{(\ell)}$. The bimodal collapse at layers 1–5 and 16–24 is driven by dominant singular values in the *accumulated representations* (top-1 SV explains up to 79% of variance at layer 16), which can emerge from accumulation dynamics regardless of individual weight ranks.

T-9's key finding — that Q/K routing matrices are lower-rank (0.31) than V/O value matrices (0.52) across all 36 layers — has a geometric interpretation from T-4: routing (deciding where to attend) operates in a lower-dimensional subspace, while value extraction must manipulate the full representation geometry. During the collapse regions (PR 1.6–7.5), even Q/K's low-rank routing is operating on a near-one-dimensional manifold, which may explain why attention patterns become less input-specific in these layers.

T-9 also found that gate\_proj effective rank increases dramatically in late layers (+0.166, p < 0.0001). In T-4, late layers show partial PR recovery (10–78) with massive norm growth, suggesting the increased gating capacity is needed to manage the expanding, high-magnitude representations.

### T-17 (Contrastive Trajectories): Geometry of Divergence

T-17 found that **shared-prefix** antonym pairs (e.g., "The Sun is a star" vs "The Sun is a planet") maintain high cosine similarity (>0.63) across layers 2–34, with a dip at layer 21. T-4 explains the mechanism: during the deep collapse region (layers 16–24, PR 1.6–7.5), representations are squeezed onto a near-one-dimensional manifold. Two sequences processing different tokens ("star" vs "planet") are forced into the same narrow subspace — their cosine similarity remains high because there's essentially only one direction to point in. The dip at layer 21 (PR = 2.6) coincides with the beginning of the PR recovery within the collapse region, where enough dimensions re-emerge to begin separating the antonyms.

T-17 also found a two-phase processing structure in prefix-controlled experiments: antonym KL > synonym KL in layers 0–4 (semantic discrimination), then synonym KL > antonym KL in layers 5–35 (form commitment). A brief reversal at layers 21–23 (antonym > synonym) coincides with T-4's deep collapse region, suggesting the extreme dimensionality bottleneck forces a re-evaluation of semantic content.

T-17's finding that layer 35 is a "universal discriminator" (all cosine similarities drop sharply) aligns directly with T-4's final-layer geometry: PR expands to 118.7, raw cosine drops to 0.09, and category separation jumps to 1.03. The model deliberately disperses representations at the final layer, maximally separating all token alternatives for the LM head.

### T-1 (Logit Lens): Prediction Quality vs Geometric Phase

T-1 found a four-phase architecture in prediction quality: representation building (L0–12, <1% accuracy), early semantics (L13–21), prediction formation (L22–28), and refinement (L29–35, reaching 99.5%). The geometric phases from T-4 provide a structural explanation:

- **L0–12 (<1% accuracy)**: Spans the early collapse (PR 6–43) and mid recovery (PR 120–178). Representations are being geometrically restructured — first compressed, then expanded into a high-dimensional working space. Predictions are poor because the model hasn't committed to a direction in representation space.
- **L22–28 (rapid accuracy climb)**: Spans the deep collapse (PR 1.6–7.5) and beginning of late recovery. The extreme dimensionality bottleneck *forces* commitment — representations are squeezed into so few dimensions that the model must resolve ambiguity, which is exactly when prediction accuracy climbs fastest.
- **L35 (99.5% accuracy)**: The de-anisotropification (PR 119, cosine 0.09) creates a well-separated representational landscape where the LM head can cleanly select the correct token.

T-1 also found that layer 0 actively *destroys* embedding predictions (mean rank 42K → 77K), which aligns with T-4's geometric expansion at layer 0 — the dimensionality increase from 68 to 112 with 7.7× norm amplification completely restructures the representational space.

### T-3 (Layer Swap Cost): Geometry Predicts Swap Tolerance

T-3 found that the cheapest swaps are exclusively adjacent late-middle pairs (layers 25–33, Δloss 0.06–0.10), while early layers (0–2) and layer 35 produce the costliest swaps (Δloss 10–23). T-4's geometry explains this asymmetry:

- **Cheapest swaps (L25–L33)**: These layers sit in the late recovery region where PR increases gradually (10 → 78) and norms grow smoothly (139 → 460). Adjacent layers have similar geometric signatures, so swapping them causes minimal disruption.
- **Costliest swaps (L0–L2, L35)**: Layer 0 performs the unique geometric expansion (PR 68→112, norm 1→7.7). Layer 35 performs the unique de-anisotropification (PR 53→119, cosine 0.62→0.09). These are geometrically singular transformations with no substitute.
- **T-3's 3-zone clustering** (early [0–6], middle [7–30], late [31–35]) aligns with T-4's geometric regimes, with the early/middle boundary at the dimensionality explosion (layer 6, PR jumps to 170) and the middle/late boundary near the end of the deep collapse recovery.

### Synthesis: The Geometric Processing Pipeline

Combining T-4 with other experiments reveals a coherent geometric processing pipeline:

1. **Layer 0 — Geometric expansion**: Projects embeddings from ~68 to ~112 effective dimensions with 7.7× norm amplification. Critical (99.6× knockout, T-2) but linearizable (99.1% recovery, T-7). Destroys embedding prediction signal (T-1).
2. **Layers 1–5 — First compression**: Collapses to 6–43 dimensions. Linearizable (91–99% recovery, T-7), high local nonlinearity (gap 0.17–0.25, T-7). The model squeezes out irrelevant embedding dimensions.
3. **Layers 6–15 — Distributed processing**: Expands to 120–178 dimensions. Layer 6 is the critical hub (21.7× knockout, T-2; appears in 4/5 top synergistic pairs). Locally linear but globally nonlinear (T-7). Low Jacobian consistency (0.55–0.70) — computation is highly input-dependent.
4. **Layers 16–24 — Second compression**: Collapses to 1.6–7.5 dimensions. The most extreme bottleneck in the network (top-1 SV explains up to 79% of variance). Forces representations through an information bottleneck, discarding alternatives and committing to a specific interpretation. Prediction accuracy begins its steep climb (T-1). Globally nonlinear (T-7).
5. **Layers 25–34 — Output preparation**: Partial recovery to 10–78 dimensions with dramatic norm growth (139→568). High Jacobian consistency for layers 29+ (0.76–0.88, T-7) — a more uniform, input-independent transformation. Mostly linearizable (L32–33 at 87–88%, T-7), though L34 is a notable exception (−45%). Cheapest swap region (T-3). Late-layer gate\_proj rank increases (T-9).
6. **Layer 35 — De-anisotropification**: Disperses representations (PR 119, cosine 0.09, norm ↓ to 391) and maximally separates semantic categories (separation 1.03). Universal discriminator (T-17). MLP-driven nonlinearity (gap 0.24, T-7). Achieves 99.5% top-1 accuracy (T-1).

## Practical & Architectural Implications

The geometric processing pipeline revealed by T-4, combined with findings from T-1 through T-17, yields concrete guidance for model optimization, deployment, and architecture design.

### 1. Layer Pruning & Structured Distillation

The two compression bottlenecks (layers 1–5, PR 6–43; layers 16–24, PR 1.6–7.5) pass information through as few as 1–2 effective dimensions. This means the intermediate layers within each bottleneck are performing incremental refinements on a near-one-dimensional manifold — making them prime candidates for pruning or merging.

- **Layers 17–23** are the most compressible region: PR stays below 7.5, top-1 SV explains 36–79% of variance, and T-3 shows adjacent swaps in layers 25–33 are cheap (Δloss 0.06–0.10), meaning downstream layers tolerate perturbation. A distilled model could replace these 7 layers with 2–3 layers (or even a single linear projection capturing the dominant singular direction) with minimal quality loss.
- **Layers 1–4** are similarly compressible (PR 6–43) and T-7 confirms they are linearizable (91–99% recovery). These could be collapsed into a single learned linear map during distillation.
- **Do not prune layer 0, layer 6, or layer 35.** Layer 0 performs the irreplaceable geometric expansion (T-2: 99.6× knockout ratio). Layer 6 is the critical computational hub (T-2: 21.7× knockout, 4/5 top synergistic pairs). Layer 35 performs de-anisotropification essential for LM head accuracy.

### 2. Quantization Strategy by Geometric Phase

Different geometric regimes have fundamentally different precision requirements:

- **Bottleneck layers (1–5, 16–24)**: Representations live in 1–7 effective dimensions. These layers are inherently low-precision-tolerant — quantizing to INT4 or even INT3 should preserve the dominant singular direction with minimal degradation. The signal-to-noise ratio is naturally high because variance concentrates on so few axes.
- **Distributed processing layers (6–15)**: PR 120–178 with low Jacobian consistency (T-7: 0.55–0.70) means computation is highly input-dependent across many dimensions. These layers are most sensitive to quantization noise — prefer INT8 or FP8 here.
- **Late recovery layers (25–34)**: High Jacobian consistency (T-7: 0.76–0.88) means transformations are more uniform. Despite high norms (139–568), the near-linear behavior suggests moderate quantization (INT6–INT8) is safe, though the large norm magnitudes require careful scaling factor selection.
- **Layer 35**: The de-anisotropification involves a deliberate norm reduction (568→391) and cosine collapse (0.62→0.09). This precision-sensitive dispersal benefits from higher precision (FP8 or INT8 minimum).

This suggests a **mixed-precision quantization** scheme aligned to geometric phases could outperform uniform quantization at the same average bit-width.

### 3. KV-Cache Compression

The low effective dimensionality in bottleneck regions implies that key/value vectors at those layers are highly redundant. Combined with T-9's finding that Q/K routing matrices are lower-rank (0.31 effective rank) than V/O matrices (0.52):

- **Layers 16–24**: With PR as low as 1.6, key vectors at these layers project onto essentially one direction. KV-cache entries could be compressed via low-rank projection (rank 4–8 should suffice) or aggressive quantization without meaningful attention degradation.
- **Layers 6–15**: High-dimensional representations (PR 120–178) require fuller KV-cache fidelity. Compression here risks losing the input-dependent routing that makes these layers globally nonlinear (T-7).

### 4. Feature Extraction & Embedding Selection

For downstream tasks requiring sentence/token embeddings:

- **Always mean-center** representations before use. T-4 shows all anisotropy is mean-direction anisotropy (centered cosine ≈ 0 everywhere). Mean-centering trivially eliminates this, producing near-isotropic embeddings that are better suited for cosine-similarity-based retrieval and clustering.
- **Use layer 27 for semantic embeddings**: Peak category separation (0.21) before the late-layer anisotropic collapse. Representations are geometrically well-structured (PR ≈ 60) with moderate norms.
- **Use layer 35 for discriminative tasks**: Maximum category separation (1.03) and near-isotropic geometry (cosine 0.09). However, this is optimized for next-token prediction — fine-tuning may be needed for other tasks.
- **Avoid layers 16–24 for embeddings**: PR 1.6–7.5 means representations are too compressed to preserve fine-grained distinctions. Even after mean-centering, the effective information content is minimal.

### 5. Early Exit & Adaptive Computation

The geometric pipeline suggests natural exit points for adaptive-depth inference:

- **After layer 15** (end of distributed processing, PR ~178): For easy inputs where the model has already committed to a prediction (T-1: some tokens reach high confidence by mid-layers), the high-dimensional representation may contain enough signal for a lightweight exit head. However, T-1 shows <1% accuracy at layer 12 for the general case, so early exit here is only viable with a confidence threshold.
- **After layer 24** (end of deep collapse): The information bottleneck has forced commitment — prediction accuracy climbs steeply through this region (T-1). An exit head at layer 24–25 could capture tokens that resolved during the compression phase.
- **The cost of skipping layer 35** is high: de-anisotropification is necessary for clean LM head discrimination. Any early-exit head at earlier layers must compensate for the anisotropic cone geometry (high cosine similarity, large norms) — likely requiring a learned normalization or dispersal pre-projection.

### 6. Architecture Design Insights

The bimodal bottleneck pattern appears functional rather than accidental:

- **Deliberate information bottlenecks**: The two compression phases (layers 1–5 and 16–24) mirror the information bottleneck principle — compress to discard irrelevant information, then expand to build task-relevant representations. This suggests architectures that *explicitly* design bottleneck layers (e.g., via reduced hidden dimensions or low-rank constraints) might achieve similar information compression more parameter-efficiently.
- **The final-layer dispersal is architecturally necessary**: The LM head's unembedding matrix expects near-isotropic input (T-4: layer 35 drops cosine from 0.62 to 0.09). Architectures that skip or share the final layer must account for this — directly applying the LM head to layer 34 representations would face a 0.62 mean cosine similarity, degrading token discrimination.
- **Hub layers deserve over-parameterization**: Layer 6 occupies a unique geometric position (onset of high-dimensional processing, PR jumps from 43 to 170) and is critical for model function (T-2). Allocating more parameters to hub layers (wider MLP, more attention heads) while narrowing bottleneck layers could improve parameter efficiency.
- **Norm management is an implicit architectural constraint**: The superlinear norm growth (1→568 across 35 layers) with a final correction (568→391 at layer 35) suggests that RMSNorm alone does not prevent norm accumulation — it only controls per-layer scale. Architectures with explicit norm management (e.g., periodic norm reset layers, or residual scaling factors that decay with depth) might reduce the need for the final-layer dispersal and enable more uniform geometric profiles.

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
