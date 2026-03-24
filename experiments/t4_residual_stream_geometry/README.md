# T-4: Residual Stream Geometry Across Depth

## Motivation & Research Question

How does the geometry of the hidden-state manifold change as representations flow through transformer layers? Specifically:

1. **Effective dimensionality**: How many dimensions do representations actually use at each layer?
2. **Isotropy and anisotropy structure**: Are representations uniformly distributed in the space, or clustered in a low-dimensional cone? If anisotropic, is it intrinsic or removable?
3. **Token clustering**: Do tokens from semantically similar prompts converge as depth increases?
4. **Layer impact**: How much does each layer change the residual stream — in magnitude, direction, and relative contribution?
5. **Residual persistence**: How strongly do signals from earlier layers survive through later processing? Which layers' contributions persist to the final output?

This connects to the Layer Shuffle Recovery experiment (on Qwen3-1.7B) — if activation flow fingerprints work well for ordering layers, there must be a predictable geometric transformation at each layer. Understanding the geometry explains *why* those fingerprints are so discriminative.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, hidden\_dim=2560)
- **Data**: 50 pre-generated completions from `data/text_completions/`, covering 7 categories (factual, reasoning, linguistic, code, world\_knowledge, technical, rare)
- **Tokens analyzed**: 4,094 completion tokens (prompt/template tokens excluded)
- **Layers**: 37 measurement points — embedding output + 36 transformer layer outputs
- **Hardware**: NVIDIA B200, bf16 inference, float32 metric computation
- **Runtime**: ~143 seconds total

## Mathematical Framework

### Participation Ratio as Effective Dimensionality

Given a set of $N$ token representations $\lbrace\mathbf{h}_1, \ldots, \mathbf{h}_N\rbrace \subset \mathbb{R}^d$, form the mean-centered data matrix $\mathbf{H} \in \mathbb{R}^{N \times d}$ with rows $\mathbf{h}_i - \bar{\mathbf{h}}$. The SVD gives singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r \ge 0$ where $r = \min(N, d)$. Define the normalized energy distribution:

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

For a set of vectors $\lbrace\mathbf{h}_i\rbrace$, the **mean pairwise cosine similarity** is:

$$\bar{c} = \mathbb{E}_{i \ne j}\left[\frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\| \cdot \|\mathbf{h}_j\|}\right]$$

For vectors drawn uniformly on $S^{d-1}$ (the unit sphere in $\mathbb{R}^d$), $\mathbb{E}[\bar{c}] = 0$ and $\text{Var}[\bar{c}] = \mathcal{O}(1/d)$. Any significant deviation from zero indicates anisotropy — the representations cluster in a sub-region of the sphere.

**Decomposition into mean-direction and intrinsic components.** Write $\mathbf{h}_i = \bar{\mathbf{h}} + \tilde{\mathbf{h}}_i$ where $\bar{\mathbf{h}} = \frac{1}{N}\sum_i \mathbf{h}_i$ is the centroid (so $\sum_i \tilde{\mathbf{h}}_i = \mathbf{0}$). The inner product decomposes as:

$$\langle \mathbf{h}_i, \mathbf{h}_j \rangle = \|\bar{\mathbf{h}}\|^2 + \langle \bar{\mathbf{h}},\, \tilde{\mathbf{h}}_i + \tilde{\mathbf{h}}_j \rangle + \langle \tilde{\mathbf{h}}_i, \tilde{\mathbf{h}}_j \rangle$$

When $\|\bar{\mathbf{h}}\| \gg \|\tilde{\mathbf{h}}_i\|$ for typical $i$, the first term dominates and:

$$\frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\|\|\mathbf{h}_j\|} \approx \frac{\|\bar{\mathbf{h}}\|^2}{\|\bar{\mathbf{h}}\|^2 + \mathbb{E}[\|\tilde{\mathbf{h}}\|^2]} \to 1$$

Computing $\bar{c}$ on the centered vectors $\tilde{\mathbf{h}}_i$ isolates the **intrinsic** anisotropy — any structure that remains after removing the shared mean direction.

### Spectral Flatness

The **spectral flatness** of the covariance eigenvalue spectrum $\lbrace\lambda_i\rbrace_{i=1}^{m}$ (where $m$ is the number of positive eigenvalues, with $\lambda_i = \sigma_i^2 / (N-1)$) is the ratio of geometric to arithmetic mean:

$$\text{SF} = \frac{\left(\prod_{i=1}^{m} \lambda_i\right)^{1/m}}{\frac{1}{m}\sum_{i=1}^{m} \lambda_i} = \frac{\exp\left(\frac{1}{m}\sum_{i=1}^{m} \ln \lambda_i\right)}{\frac{1}{m}\sum_{i=1}^{m} \lambda_i}$$

By the AM-GM inequality, $\text{SF} \in (0, 1]$, with equality to 1 iff all eigenvalues are equal (perfectly isotropic). A value near 0 indicates a "spiked" spectrum dominated by a few large eigenvalues.

**Relationship to PR.** Both PR and SF measure spectral concentration, but differently. PR counts how many directions carry meaningful variance (an effective count), while SF measures how uniformly the variance is spread (a shape statistic). Two distributions with the same PR can have different SF if their energy distributions have different shapes — e.g., a flat-then-zero spectrum and a smoothly decaying spectrum can yield the same PR but different SF.

### Cluster Separation

For tokens partitioned into $C$ categories, define:

$$\bar{c}_{\text{intra}} = \frac{1}{C} \sum_{c=1}^{C} \mathbb{E}_{i,j \in c,\, i \ne j}\left[\cos(\mathbf{h}_i, \mathbf{h}_j)\right], \qquad \bar{c}_{\text{inter}} = \mathbb{E}_{i \in c_1,\, j \in c_2,\, c_1 \ne c_2}\left[\cos(\mathbf{h}_i, \mathbf{h}_j)\right]$$

The **cluster separation ratio** is:

$$S = \frac{\bar{c}_{\text{intra}} - \bar{c}_{\text{inter}}}{|\bar{c}_{\text{inter}}| + \epsilon}$$

where $\epsilon = 10^{-10}$ prevents division by zero when inter-category similarity vanishes. $S > 0$ means tokens within a category are more similar to each other than to tokens in other categories. Note that $S$ is sensitive to the overall anisotropy level — in highly anisotropic layers where all cosines are large, $S$ can be small despite meaningful category structure (because both intra and inter are large).

### Layer Impact Metrics

Each transformer layer $\ell$ adds an update to the residual stream: $\mathbf{h}^{(\ell)} = \mathbf{h}^{(\ell-1)} + f_\ell(\mathbf{h}^{(\ell-1)})$, where $\boldsymbol{\delta}_\ell = f_\ell(\mathbf{h}^{(\ell-1)})$ is the layer's update. We measure four properties of this update:

- **Update magnitude**: $\|\boldsymbol{\delta}_\ell\|$ — how much the layer changes the residual stream in absolute terms
- **Directional preservation**: $\cos(\mathbf{h}^{(\ell)}, \mathbf{h}^{(\ell-1)})$ — does the output point in roughly the same direction as the input? Values near 1 mean the layer makes a small angular perturbation; values near 0 mean a large redirect
- **Relative update size**: $\|\boldsymbol{\delta}_\ell\| / \|\mathbf{h}^{(\ell)}\|$ — the update magnitude relative to the resulting representation. Values $< 1$ mean the residual dominates; values $> 1$ mean the update overwhelms the residual
- **Update-residual alignment**: $\cos(\boldsymbol{\delta}_\ell, \mathbf{h}^{(\ell-1)})$ — is the update aligned with (+), orthogonal to (0), or opposing (-) the existing residual? Positive alignment reinforces the current direction (growing norms); negative alignment redirects or disperses

### Residual Persistence

To quantify how signals persist across depth, we track:

- **Embedding persistence**: $\cos(\mathbf{h}^{(\ell)}, \mathbf{h}^{(\text{emb})})$ — how much of the original embedding direction survives at layer $\ell$
- **Final alignment**: $\cos(\mathbf{h}^{(\ell)}, \mathbf{h}^{(L)})$ — how aligned each layer's output is with the final representation
- **Update survival**: $\cos(\boldsymbol{\delta}_\ell, \mathbf{h}^{(L)})$ — does layer $\ell$'s update point toward the final output?
- **Cumulative drift**: $\cos(\mathbf{h}^{(\ell)}, \mathbf{h}^{(\ell-k)})$ for gap sizes $k$ — how rapidly the representation direction changes over multiple layers
- **Residual decomposition**: Since $\mathbf{h}^{(L)} = \mathbf{h}^{(\text{emb})} + \sum_{\ell=0}^{L-1} \boldsymbol{\delta}_\ell$, we project each $\boldsymbol{\delta}_\ell$ onto $\hat{\mathbf{h}}^{(L)}$ (the unit vector in the final direction) to measure each layer's signed contribution to the final representation

## Methods

### 1. Hidden State Extraction
Forward-hook-based extraction at every layer, identical to T-1. For each of 50 prompts, we register hooks on all 36 `model.model.layers[i]` modules plus capture the embedding output. Only completion-token positions are retained (positions $\ge$ `prompt_token_count`), yielding 4,094 token vectors of dimension 2,560 per layer.

### 2. Effective Dimensionality (Participation Ratio)
For each layer's pooled matrix $\mathbf{H} \in \mathbb{R}^{4094 \times 2560}$, we mean-center and compute SVD. The participation ratio is computed from the squared singular values as defined above.

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

### 5. Layer Impact Analysis
For each transformer layer, compute the update vector $\boldsymbol{\delta}_\ell = \mathbf{h}^{(\ell)} - \mathbf{h}^{(\ell-1)}$ and measure its magnitude, directional preservation, relative size, and alignment with the existing residual (see mathematical framework above).

### 6. Residual Persistence Analysis
Track embedding persistence, final-output alignment, update survival, cumulative drift across gap sizes (1, 2, 4, 8, 16 layers), and decompose the final representation into per-layer contributions via projection.

### 7. Update Correlation Matrix
Compute mean update vectors per layer (averaged across tokens), then measure the pairwise cosine similarity between all layer pairs — revealing which layers push the residual stream in similar or opposing directions.

## Results

### Effective Dimensionality

![Geometry Overview — dimensionality, isotropy, anisotropy decomposition, and norms across layers](results/geometry_overview.png)

| Region | Layers | PR Range | Interpretation |
|--------|--------|----------|----------------|
| Embedding | emb | 73.4 | Moderate — token embeddings span ~73 effective dimensions |
| Early spike | 0 | 124.9 | First layer *increases* dimensionality |
| Early collapse | 1–5 | 6.0–43.0 | Dramatic collapse — layers 1–2 reduce to just ~6 effective dims |
| Mid recovery | 6–15 | 146.9–204.7 | Recovery to high-dimensional plateau, peaking at layer 8 |
| Deep collapse | 16–24 | 2.3–16.8 | Second major collapse — PR drops to **2.3** at layer 16 |
| Late recovery | 25–34 | 23.9–127.4 | Gradual recovery through final layers |
| Final layer | 35 | 160.1 | Sharp dimensionality expansion at the output |

The **bimodal collapse pattern** (layers 1–5 and 16–24) is striking. The deep collapse at layer 16 (PR=2.3) means representations are nearly one-dimensional — almost all variance sits on a single axis. This suggests two critical "bottleneck" regions where the model compresses information maximally before expanding it again.

### Isotropy / Anisotropy

The raw mean cosine similarity hovers around **0.37–0.51** throughout most layers (moderately anisotropic), peaking at **0.63** at layer 34 before dramatically dropping to **0.09** at the final layer 35.

The **centered** cosine similarity is essentially **zero** (~−0.002 to +0.008) at every layer. This is a key finding: **all observed anisotropy is due to a shared mean direction, not intrinsic geometric clustering**. After removing the mean, token representations are nearly perfectly isotropic at every layer.

Spectral flatness follows a non-monotonic trajectory: the embedding starts high (0.35), drops sharply at layer 0 (0.08), then generally increases through the network — reaching 0.13–0.19 in layers 6–15, dipping to 0.04 during the deep collapse (layers 16–17), and steadily climbing to 0.20–0.22 by layers 30–35. The pattern tracks the PR trajectory: collapse regions have low SF (spiked spectra), recovery regions have higher SF (more distributed spectra).

### Representation Norms

Norms grow **superlinearly** across depth:
- Embedding: 1.00
- Layer 10: 40.5
- Layer 20: 67.7
- Layer 30: 282.0
- Layer 34: 570.7

Combined with high raw cosine similarity, this confirms the **anisotropic cone** pattern: representations extend along an increasingly narrow cone with growing norms. The final layer (35) breaks this pattern — norms drop to 387.6 and cosine drops to 0.09 (see Layer Impact below for the mechanism).

### Layer Impact on Residual Stream

![Layer Impact — update magnitude, directional preservation, relative update size, update-residual alignment](results/layer_impact.png)

#### Update Magnitude

| Layer | ‖delta‖ | Interpretation |
|-------|---------|----------------|
| 0 | 7.7 | Moderate — transforms embedding |
| 1–5 | 4.5–11.9 | Small — incremental refinement |
| 6–15 | 12.6–17.3 | Medium-sized, relatively consistent updates |
| 16–24 | 15.2–46.9 | Growing — accelerating norm contribution |
| 25–34 | 43.6–190.9 | Large and escalating — dominant norm builders |
| 35 | **513.9** | Massive — the largest single-layer update by far |

Layer 35's update (‖delta‖=513.9) is **2.7x the size of the residual it receives** (‖h(34)‖=570.7), making it the only layer whose update exceeds its input in magnitude.

#### Directional Preservation

Most layers preserve the direction of their input remarkably well: cos(h(l), h(l-1)) is **0.91–0.96** for layers 1–34. This means each layer makes a relatively small angular perturbation while growing the norm. Two layers stand out:

- **Layer 0**: cos = 0.11 — nearly orthogonal to the embedding. This layer completely redirects the representation into a new subspace, consistent with its role as a geometric expansion (PR: 73→125).
- **Layer 35**: cos = 0.47 — substantial redirection. The only layer besides L0 that significantly changes the direction of the residual stream.

#### Relative Update Size

The ratio ‖delta‖ / ‖h(l)‖ is remarkably stable at **0.29–0.44** for layers 1–34 — each layer's update is about 30–40% the size of the resulting representation. This means the residual stream always dominates over any single layer's contribution, providing stability.

Two exceptions:
- **Layer 0**: ratio = 0.99 — the update is essentially the entire representation (the embedding contributes very little to the output norm)
- **Layer 35**: ratio = **1.38** — the update is *larger* than the resulting representation, which is only possible because the update partially opposes the residual (see below)

#### Update-Residual Alignment

This metric reveals the most nuanced picture of what each layer does to the residual stream:

| Region | Layers | cos(delta, h(l-1)) | Interpretation |
|--------|--------|-------------------|----------------|
| Initial redirect | 0 | −0.02 | Orthogonal — embedding is irrelevant to update direction |
| Early reinforcement | 1–5 | +0.36 to +0.39 | **Positive** — updates reinforce the residual direction, growing norms |
| Mid transition | 6–8 | +0.19 → +0.02 | Shifting from reinforcement to orthogonal |
| Mid orthogonal/opposing | 9–15 | −0.14 to −0.09 | **Weakly opposing** — updates push slightly against the residual |
| Deep collapse transition | 16–17 | −0.05 | Nearly orthogonal |
| Late reinforcement | 18–34 | +0.02 to +0.44 | Increasingly **positive** — strong reinforcement drives superlinear norm growth |
| Final dispersal | 35 | **−0.73** | Strongly **opposing** — the update actively pushes against the residual |

This reveals layer 35's mechanism: its update actively opposes the residual direction (cos = −0.73), which is why norms drop from 571 to 388 despite the update being the largest of any layer. The layer doesn't just add variance in new directions — it actively *subtracts* from the dominant direction that accumulated over layers 18–34.

The transition from opposing (L9–15) to reinforcing (L18–34) coincides with the onset of the deep collapse, suggesting the bottleneck is where the model switches from exploring diverse directions to committing to a dominant direction that will be refined through the late layers.

### Residual Persistence

![Residual Persistence — embedding persistence, final alignment, update survival, cumulative drift](results/residual_persistence.png)

#### Embedding Persistence

Cosine similarity between each layer's output and the original embedding drops rapidly:
- Layer 0: 0.11 (already mostly destroyed)
- Layer 5: 0.03
- Layers 6–35: 0.00–0.05

**The embedding direction is essentially erased by layer 5.** The original token identity encoded by the embedding table does not persist as a directional signal — it is completely overwritten by the transformer layers. This means all directional information in the later residual stream comes from the transformer layers' accumulated updates, not from the embedding.

#### Alignment with Final Output

Cosine similarity between each layer's output and the final representation (h(35)) grows monotonically:
- Layer 0: 0.04
- Layer 10: 0.08
- Layer 20: 0.17
- Layer 25: 0.32
- Layer 30: 0.45
- Layer 34: 0.47
- Layer 35: 1.00 (by definition)

This shows a **gradual convergence**: the representation slowly rotates toward the final output direction throughout the network, with the most rapid alignment happening in layers 22–35 (from 0.23 to 1.00). There is no sudden jump — the final direction is built incrementally.

#### Which Layers' Updates Survive to the Final Output?

![Residual Decomposition — signed projection and alignment fraction](results/residual_decomposition.png)

The signed projection of each layer's update onto the final direction ($\langle \boldsymbol{\delta}_\ell, \hat{\mathbf{h}}^{(L)} \rangle$) reveals which layers actually contribute to the final representation:

| Layer | Signed Projection | % of Final Norm | Interpretation |
|-------|-------------------|-----------------|----------------|
| emb | 0.5 | 0.1% | Negligible — embedding barely contributes to final direction |
| 0–15 | 0.3–2.3 | 0.1–0.6% | Minimal — early/mid layers contribute almost nothing to the final direction |
| 16–21 | 2.7–6.5 | 0.7–1.7% | Small but growing contributions |
| 22–24 | 12.2–17.4 | 3.1–4.5% | Significant contributions begin |
| 25–29 | 16.5–26.3 | 4.3–6.8% | Substantial — each layer adds meaningfully |
| 30–34 | 31.4–55.7 | 8.1–14.4% | Dominant — these 5 layers account for ~50% of the final norm |
| 35 | **123.4** | **31.8%** | Single largest contributor |

**Layers 30–35 collectively account for ~80% of the final representation's magnitude along its own direction.** Layers 0–15 together contribute less than 5%. This means the "early processing" done by the first half of the network is almost entirely orthogonal to the final output direction — it builds intermediate structure that is later transformed into the final direction by late layers.

The fraction of each layer's update that aligns with the final direction also grows monotonically: early layers project only 3–6% of their update onto the final direction, while layer 35 projects 24%. This means early layers' updates are almost entirely orthogonal to the eventual output — they build structure in directions that are later consumed and redirected.

### Token Clustering

![Token Clustering — intra/inter-category similarity, separation ratio, and centroid heatmap](results/token_clustering.png)

| Layer | Intra-cat | Inter-cat | Separation |
|-------|-----------|-----------|------------|
| emb   | 0.077     | 0.069     | 0.11       |
| 0     | 0.439     | 0.431     | 0.02       |
| 11    | 0.438     | 0.395     | 0.11       |
| 22    | 0.400     | 0.359     | 0.11       |
| 27    | 0.398     | 0.343     | **0.16**   |
| 34    | 0.635     | 0.612     | 0.04       |
| 35    | 0.131     | 0.076     | **0.73**   |

Category separation generally increases through the middle layers, from near-zero at layer 0 to a peak of 0.16 at layer 27. However, **separation reverses in layers 29–34**: as norms grow superlinearly and raw cosine similarity climbs to 0.63, both intra- and inter-category similarity rise together, compressing the gap and dropping separation back to 0.04 at layer 34. The final layer 35 then produces a jump to **0.73** — the dispersal of representations lets category structure become the dominant organizing principle, with intra-category similarity dropping to 0.13 while inter-category drops further to 0.08.

### Update Correlation Matrix

![Update Correlation Matrix — cosine similarity between mean layer updates](results/update_correlation.png)

The correlation matrix between mean layer updates reveals clear **block structure**:

1. **Early block (L0–5)**: Updates are mutually correlated (cos 0.3–0.8), forming a coherent group. These layers work together to build the initial representation.
2. **Mid block (L6–15)**: Updates are correlated among themselves but only weakly with the early block. The boundary at layer 6 (where PR jumps from 43 to 197) marks a transition to a different processing mode.
3. **Late block (L17–34)**: A large, strongly correlated block — these layers push in similar directions, driving the superlinear norm growth. The deep collapse layers (16–19) form a transition zone.
4. **Layer 35 is anti-correlated with all other layers** — the entire column/row shows blue (negative) values. This confirms quantitatively that layer 35's update opposes the direction established by all previous layers.

The block boundaries align with the geometric phase transitions: early collapse → mid recovery → deep collapse → late recovery → dispersal. Layers within the same geometric phase push in similar directions; layers in different phases push in different or opposing directions.

### Singular Value Spectra

![Singular Value Spectra — SV heatmap and cumulative variance curves](results/singular_value_spectra.png)

![Variance Explained — top-k variance explained across layers](results/variance_explained.png)

The cumulative variance plots reveal:
- At layers 1–2 (early collapse), the top-1 SV explains **~40%** of variance; at layers 3–5 the dominance decreases (31%→22%→14%) as PR recovers
- At layers 16–24 (deep collapse), the top-1 SV explains **36–79%** of variance, peaking at layer 16 (79%) and decreasing monotonically as PR slowly recovers through this region
- At mid layers (6–15), variance is more distributed — top-1 explains only 3–4%, and top-10 explains ~25–30%
- The final layer 35 has one of the most distributed spectra — top-1 explains only 4%, with top-50 needed for ~68% variance

The SV heatmap shows a clear "bright band" at singular value index 0 during collapse layers, confirming the single-dominant-direction interpretation.

## Conclusions & Key Findings

1. **Bimodal dimensionality collapse**: Two distinct bottleneck regions (layers 1–5, PR 6–43; layers 16–24, PR 2.3–16.8) bookend a high-dimensional processing plateau (layers 6–15, PR 147–205). These are sharp phase transitions, not gradual.

2. **All anisotropy is mean-direction anisotropy**: Centered cosine similarity is ~0 everywhere — mean-centering produces near-isotropic representations at every layer.

3. **Layer 0 and layer 35 are geometric singularities**: Layer 0 completely redirects the embedding (cos(io) = 0.11, update ratio = 1.0). Layer 35 actively opposes the accumulated residual (cos(delta, prev) = −0.73, update ratio = 1.38). All other layers preserve direction with cos > 0.91 and updates at 30–40% of the residual.

4. **The embedding is erased by layer 5**: All directional information in the later residual stream comes from transformer updates, not from the embedding.

5. **Late layers dominate the final representation**: Layers 30–35 contribute ~80% of the final representation's magnitude along its own direction. Layers 0–15 contribute <5% — the early network builds intermediate structure orthogonal to the final output.

6. **Layer updates form three correlated blocks** (early/mid/late) with a strongly anti-correlated layer 35, aligning with the geometric phase boundaries. This explains why activation fingerprints (Layer Shuffle Recovery) are discriminative for layer ordering.

## Cross-Experiment Connections

Throughout this section, T-4 geometric data (PR, norms, SV spectra, impact metrics) are as reported in the Results tables above. Only cross-experiment findings and their geometric interpretations are stated here.

### T-2 (Layer Knockout) + T-7 (Linearization Gap): Geometry Explains Criticality and Linearizability

T-2 found that layer 0 is catastrophically critical (99.6x loss ratio). T-7 showed it is also linearizable (99.1% recovery with a full-rank linear map). T-4 explains why both are true: layer 0 is the only layer that *increases* dimensionality (PR: 73→125) and completely redirects the representation (cos(io) = 0.11). This geometric expansion is a specific, learnable linear transformation — which is why linear replacement works but removal is catastrophic.

T-7's "bimodal linearity" finding maps onto T-4's geometry:

| T-4 Geometry Region | Layers | PR | T-7 Linear Replacement |
|---|---|---|---|
| Early collapse (low PR) | 1–5 | 6–43 | 91–99% recovery |
| Mid recovery (high PR) | 6–15 | 147–205 | Mixed: L6 recovers (87%), L8–L12 fail |
| Deep collapse (low PR) | 16–24 | 2.3–16.8 | Fails: −1796% to −101% |
| Late recovery | 25–34 | 24–127 | L32–L33 recover (87–88%), **L34 fails (−45%)** |

The general trend: **low-PR regions tend to be linearizable, high-PR regions tend to resist linear replacement**. When representations are squeezed into 2–6 effective dimensions, the layer's transformation is constrained enough to be captured by a fixed linear map. When representations span 150–200 dimensions, the transformation is input-dependent and globally nonlinear.

T-2 also found layer 6 is a computational hub (2nd most critical at 21.7x, appears in 4/5 top synergistic pairs). In T-4, layer 6 sits at the onset of the mid-recovery plateau (PR jumps from 43 at layer 5 to 197 at layer 6) — it marks the transition from compressed to distributed representation. The update correlation matrix shows layer 6 at the boundary between the early and mid correlation blocks.

### T-7 (Linearization Gap): Local Smoothness vs Global Geometry

T-7 found a U-shaped nonlinearity profile with middle layers (6–18) being most locally linear (gap ~0.13–0.15) but globally nonlinear (catastrophic replacement failure). T-7's Jacobian consistency data resolves this paradox when combined with T-4's geometry:

- **Layers 6–8**: Low Jacobian consistency (0.55–0.66) despite low linearization gap. The Jacobian is smooth at each operating point but *varies strongly across inputs*. These layers have high PR (197–205) — the many active dimensions create room for input-dependent behavior.
- **Layers 29–35**: High consistency (0.76–0.88), meaning the Jacobian is nearly the same regardless of input. Norms are largest here (236–571), and the late-layer norm growth dominates over input-dependent variation. The update-residual alignment is strongly positive (cos = 0.29–0.44), confirming these layers perform a consistent, reinforcing transformation.

T-7 also found that MLP nonlinearity drives the late-layer spike (MLP gap 0.24 at layer 35). The final-layer dispersal requires nonlinear transformation that SwiGLU provides — the strongly negative update-residual alignment (cos = −0.73) shows this is an active reversal, not just noise.

### T-9 (Weight Spectral Structure): Weight Rank vs Representation Dimensionality

T-9 found no significant correlation between weight effective rank and representation geometry (r = 0.157, p = 0.36). T-4 explains why: the residual stream is an **accumulated sum** of all upstream layer contributions:

$$\mathbf{h}^{(\ell)} = \mathbf{h}^{(0)} + \sum_{i=1}^{\ell} f_i\left(\mathbf{h}^{(i-1)}\right)$$

A single layer's weight rank constrains only its *incremental update* $f_i$, not the total dimensionality of $\mathbf{h}^{(\ell)}$. The bimodal collapse is driven by dominant singular values in the *accumulated representations* (top-1 SV explains up to 79% of variance at layer 16), which emerge from accumulation dynamics regardless of individual weight ranks.

### T-17 (Contrastive Trajectories): Geometry of Divergence

T-17 found that **shared-prefix** antonym pairs maintain high cosine similarity (>0.63) across layers 2–34, with a dip at layer 21. During the deep collapse (PR 2.3–16.8), two sequences processing different tokens are squeezed onto a near-one-dimensional manifold — their cosine similarity remains high because there is essentially only one direction available. The update-residual alignment data adds nuance: during the collapse (L16–24), layers are reinforcing a shared dominant direction (cos(delta, prev) > 0), which further compresses both sequences toward the same axis.

T-17's finding that layer 35 is a "universal discriminator" (all cosine similarities drop sharply) aligns directly with T-4's final-layer dispersal mechanism.

### T-1 (Logit Lens): Prediction Quality vs Geometric Phase

T-1 found a four-phase architecture in prediction quality: representation building (L0–12, <1% accuracy), early semantics (L13–21), prediction formation (L22–28), and refinement (L29–35, reaching 99.5%). T-4's geometric phases provide a structural explanation:

- **L0–12 (<1% accuracy)**: Spans the early collapse and mid recovery. Updates are nearly orthogonal to the final output (update-to-final cos < 0.04). The representation is being restructured in directions that won't contribute to the final prediction.
- **L22–28 (rapid accuracy climb)**: Coincides with the end of the deep collapse and beginning of late recovery. Update alignment to final jumps from 0.08 to 0.25 — these layers start building toward the output. The signed projection onto final also accelerates here (12→26 per layer).
- **L35 (99.5% accuracy)**: The dispersal creates a well-separated landscape via opposing update (cos = −0.73), letting the LM head cleanly select the correct token.

### T-3 (Layer Swap Cost): Geometry Predicts Swap Tolerance

T-3 found that the cheapest swaps are exclusively adjacent late-middle pairs (layers 25–33, delta-loss 0.06–0.10), while early layers (0–2) and layer 35 produce the costliest swaps (delta-loss 10–23). T-4's geometry and impact metrics explain this:

- **Cheapest swaps (L25–L33)**: The update correlation matrix shows these layers are strongly mutually correlated — they push in similar directions with similar magnitudes. Directional preservation is consistently high (cos > 0.95). Swapping adjacent layers within this correlated block is minimally disruptive because both layers do approximately the same thing.
- **Costliest swaps (L0–L2, L35)**: Layer 0 is a geometric singularity (cos(io) = 0.11, complete redirect). Layer 35 is anti-correlated with all other layers. These are geometrically unique transformations with no substitute.
- **T-3's 3-zone clustering** (early [0–6], middle [7–30], late [31–35]) aligns with T-4's update correlation blocks and geometric regimes.

### Synthesis: The Geometric Processing Pipeline

Combining T-4 with other experiments reveals a coherent six-phase pipeline:

| Phase | Layers | PR | Key Geometry | Impact Signature | Cross-Experiment Role |
|---|---|---|---|---|---|
| **Geometric expansion** | 0 | 73→125 | Norm 1→7.7 | cos(io)=0.11, ratio=1.0 | Critical (99.6x, T-2), linearizable (99.1%, T-7) |
| **First compression** | 1–5 | 6–43 | Top-1 SV 14–40% | Reinforcing (cos(d,h)=+0.37) | Linearizable (91–99%, T-7) |
| **Distributed processing** | 6–15 | 147–205 | Top-1 SV 3–4% | Weakly opposing (cos(d,h)=−0.10) | Hub at L6 (21.7x, T-2), low Jacobian consistency |
| **Second compression** | 16–24 | 2.3–16.8 | Top-1 SV 36–79% | Transitioning to reinforcing | Prediction accuracy begins climb (T-1) |
| **Output preparation** | 25–34 | 24–127 | Norm 139→571 | Strongly reinforcing (cos(d,h)=+0.35) | Cheapest swap region (T-3), ~50% of final norm |
| **Dispersal** | 35 | 160 | Cosine 0.63→0.09 | Opposing (cos(d,h)=−0.73), ratio=1.38 | Universal discriminator (T-17), 99.5% accuracy (T-1) |

## Practical & Architectural Implications

The following implications are **hypotheses grounded in T-4 observations and cross-experiment patterns**, not experimentally validated optimizations. They identify specific, testable predictions for future work.

### 1. Layer Pruning & Structured Distillation

The two compression bottlenecks pass information through as few as 2–3 effective dimensions, making intermediate layers within each bottleneck candidates for pruning or merging:

- **Layers 17–23**: PR stays below 10, top-1 SV explains 36–79% of variance, and their updates are highly correlated (update correlation >0.5). *Hypothesis*: these 7 layers could be replaced with 2–3 layers or a single linear projection capturing the dominant singular direction.
- **Layers 1–4**: Similarly low-dimensional (PR 6–20) and confirmed linearizable by T-7 (91–99% recovery). *Hypothesis*: these could be collapsed into a single learned linear map.
- **Do not prune layer 0, layer 6, or layer 35.** T-2 knockout ratios (99.6x, 21.7x) and T-4's unique geometric signatures at these layers indicate they perform irreplaceable transformations.

### 2. Quantization Strategy by Geometric Phase

Different geometric regimes have different precision profiles:

- **Bottleneck layers (1–5, 16–24)**: Variance concentrates on 2–6 axes, so signal-to-noise ratio is inherently high. *Hypothesis*: INT4 or lower should preserve the dominant singular direction with minimal degradation.
- **Distributed processing layers (6–15)**: PR 147–205 with weakly opposing updates means computation is input-dependent across many dimensions. *Hypothesis*: these layers are most sensitive to quantization noise and should receive higher precision (INT8/FP8).
- **Late recovery layers (25–34)**: Strongly reinforcing updates (cos = +0.35) with high directional preservation (cos(io) > 0.95) suggest consistent, well-conditioned transformations tolerant of moderate quantization, though large norms (139–571) require careful scaling.
- **Layer 35**: The dispersal involves precise norm reduction, direction reversal, and cosine collapse. *Hypothesis*: this benefits from higher precision (FP8/INT8 minimum).

### 3. KV-Cache Compression

Low effective dimensionality in bottleneck regions implies key/value vector redundancy. Combined with T-9's finding that Q/K matrices are lower-rank (0.31) than V/O matrices (0.52):

- *Hypothesis*: **Layers 16–24** KV-cache entries could be compressed via low-rank projection or aggressive quantization, since representations at these layers project onto essentially one direction (PR 2.3–16.8).
- **Layers 6–15** require fuller KV-cache fidelity — high-dimensional representations (PR 147–205) carry input-dependent routing information.

### 4. Feature Extraction & Embedding Selection

For downstream tasks requiring sentence/token embeddings:

- **Always mean-center** representations. Since all anisotropy is mean-direction anisotropy (centered cosine ~0), mean-centering produces near-isotropic embeddings better suited for cosine-similarity-based retrieval.
- **Layer 27** has peak category separation (0.16) before the late-layer anisotropic collapse, with moderate dimensionality (PR ~49) and norms — a natural choice for semantic embeddings.
- **Layer 35** has maximum category separation (0.73) and near-isotropic geometry, but is optimized for next-token prediction.
- **Avoid layers 16–24** — PR 2.3–16.8 means representations are too compressed to preserve fine-grained distinctions.

### 5. Early Exit & Adaptive Computation

The geometric pipeline suggests natural exit points:

- **After layer 24** (end of deep collapse): The bottleneck has constrained representations — prediction accuracy climbs steeply through this region (T-1). An exit head here could capture tokens that resolved during compression.
- **The cost of skipping layer 35 is high**: any early-exit head at earlier layers must compensate for the anisotropic cone geometry (high cosine, large norms) — requiring a learned dispersal pre-projection rather than a simple linear head. The opposing update mechanism of layer 35 (cos(delta, prev) = −0.73) is essential for separating tokens for the LM head.

### 6. Architecture Design Observations

- **The bimodal bottleneck pattern may be functional**: the two compression phases mirror the information bottleneck principle — compress to discard irrelevant features, then expand to build task-relevant representations. Architectures with *explicit* bottleneck layers (reduced hidden dims or low-rank constraints) might achieve similar compression more parameter-efficiently.
- **Final-layer dispersal appears necessary for the LM head**: applying the LM head to layer 34 representations (cosine 0.63) instead of layer 35 (cosine 0.09) would face high inter-token similarity, degrading discrimination. Architectures that share or skip the final layer would need an alternative dispersal mechanism.

## Usage

```bash
# Prerequisites: completions must exist at data/text_completions/qwen3-4b-instruct-2507/
poetry run python experiments/t4_residual_stream_geometry/run.py
```

Output in `experiments/t4_residual_stream_geometry/results/`:
- `summary.json` — All per-layer metrics, impact data, persistence data, update correlations, and singular value spectra
- `geometry_overview.png` — 4-panel plot: dimensionality, isotropy, anisotropy decomposition, norms
- `token_clustering.png` — Intra/inter-category similarity, separation ratio, centroid heatmap
- `singular_value_spectra.png` — SV heatmap and cumulative variance curves
- `variance_explained.png` — Top-k variance explained across layers
- `layer_impact.png` — 4-panel plot: update magnitude, directional preservation, relative update size, update-residual alignment
- `residual_persistence.png` — 4-panel plot: embedding persistence, final alignment, update survival, cumulative drift
- `update_correlation.png` — Layer update correlation heatmap
- `residual_decomposition.png` — Per-layer signed contribution and alignment fraction to final output
