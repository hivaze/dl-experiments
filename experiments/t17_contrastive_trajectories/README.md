# T-17: Contrastive Completion Trajectories

## Motivation & Research Question

T-1 (logit lens) showed that token predictions evolve through four phases across depth, and T-4 (residual stream geometry) revealed how the hidden-state manifold changes layer-by-layer. But both experiments used a single greedy completion per prompt. A natural follow-up: **how do the model's internal representations differ when processing semantically similar vs. opposite completions?**

Specifically:
1. **At which layer do antonym completions first diverge in hidden-state space?** If the prompt is "Is the Sun a star or a planet?" and we force-decode "star" vs "planet", at what depth does the model's residual stream clearly separate the two?
2. **Do synonym completions maintain similar trajectories despite different surface forms?** "Fast", "swift", and "rapid" mean the same thing — do their hidden states converge at semantic layers even though their token embeddings differ?
3. **Does meaning crystallize before form?** If synonym trajectories stay similar longer than antonym trajectories, it suggests the model builds semantic representations first and only resolves surface-level token choice in later layers.
4. **How does the divergence profile differ across relationship types?** Antonyms, synonyms, style variants, and unrelated completions should each produce distinct divergence signatures.

## Theoretical Framework

### Residual Stream as a Trajectory in Representation Space

A transformer with $L$ layers maps an input token sequence to a sequence of hidden states via the residual stream. For a given token position $t$, the hidden state after layer $\ell$ is:

$$h_t^{(\ell)} = h_t^{(0)} + \sum_{i=1}^{\ell} f_i(h_t^{(i-1)})$$

where $h_t^{(0)}$ is the token embedding and $f_i$ is the residual update from layer $i$ (attention + MLP). Given two completions $A$ and $B$ for the same prompt, the hidden states at the **first diverging token position** $t^\ast$ (the "pivot") follow two trajectories:

$$\text{Trajectory A:} \quad h_{t^\ast}^{(0,A)}, \; h_{t^\ast}^{(1,A)}, \; \ldots, \; h_{t^\ast}^{(L,A)}$$
$$\text{Trajectory B:} \quad h_{t^\ast}^{(0,B)}, \; h_{t^\ast}^{(1,B)}, \; \ldots, \; h_{t^\ast}^{(L,B)}$$

These trajectories start from different embeddings (different token IDs at position $t^\ast$) but are conditioned on the same context (identical prefix). We measure their relationship at each layer $\ell$ via:

**Cosine similarity** (directional alignment):
$$\text{cos}(\ell) = \frac{\langle h_{t^\ast}^{(\ell,A)}, \; h_{t^\ast}^{(\ell,B)} \rangle}{\|h_{t^\ast}^{(\ell,A)}\| \cdot \|h_{t^\ast}^{(\ell,B)}\|}$$

**Normalized L2 distance** (magnitude-sensitive divergence):
$$d(\ell) = \frac{\|h_{t^\ast}^{(\ell,A)} - h_{t^\ast}^{(\ell,B)}\|}{\frac{1}{2}(\|h_{t^\ast}^{(\ell,A)}\| + \|h_{t^\ast}^{(\ell,B)}\|)}$$

### KL Divergence as Functional Divergence

Cosine similarity and L2 distance operate in hidden-state space, but two hidden states can be geometrically close yet produce very different next-token distributions, or geometrically distant yet agree on the output. KL divergence on the logit distribution captures the *functionally relevant* difference — how differently the model would act on the two representations if forced to decode at that point.

At each layer, we project the hidden state through the final layer norm $\text{RMSNorm}$ and unembedding matrix $W_U$ to get logits:

$$z^{(\ell)} = W_U \cdot \text{RMSNorm}(h_{t^\ast}^{(\ell)})$$

The KL divergence $D_{\text{KL}}(p_A^{(\ell)} \| p_B^{(\ell)})$ between the resulting softmax distributions measures how distinguishable the two completions are at each layer from the model's output perspective. If antonyms (different meanings, e.g. "star" vs "planet") show higher KL than synonyms (same meaning, e.g. "fast" vs "swift") in early layers, this suggests the model resolves semantic content before surface form. If synonyms show higher KL in late layers, this suggests form-level commitment happens late. These are empirical predictions tested below.

### Linear CKA as Representational Similarity

Cosine similarity and L2 distance are defined for individual vectors — they compare the hidden state at a single token position. But completions span multiple tokens, and the *relationship between* token positions (e.g., whether the model encodes similar relative structure across the sequence) is invisible to pointwise metrics. We need a measure that asks: "do these two completions induce the same *pattern of similarities* across their token positions?" — i.e., representational similarity rather than pointwise similarity.

We use **Linear Centered Kernel Alignment** (CKA) (Kornblith et al., 2019) because it satisfies three properties that alternatives lack: (1) **invariance to orthogonal transformations and isotropic scaling** — if two representations encode the same structure but in rotated or rescaled coordinate systems, CKA still returns 1.0, unlike naive Frobenius-norm comparisons; (2) **sensitivity to representational geometry** — unlike Procrustes distance or CCA, CKA captures whether the similarity structure (which tokens are close to which) is preserved, not just whether individual dimensions align; (3) **computational simplicity** — linear CKA reduces to a ratio of squared Frobenius norms with no eigendecomposition or iterative alignment, making it tractable for 36 layers × 111 groups.

$$\text{CKA}(X, Y) = \frac{\| Y^{T} X \|_F^{2}}{\| X^{T} X \|_F \cdot \| Y^{T} Y \|_F}$$

where $X \in \mathbb{R}^{n \times d}$ and $Y \in \mathbb{R}^{n \times d}$ are the hidden-state matrices (tokens × hidden dim) for completions $A$ and $B$ respectively, after centering. CKA = 1 means the representations encode the same relational structure across token positions; CKA = 0 means the similarity patterns are unrelated. This complements the pointwise metrics: cosine similarity tells us whether individual positions align, while CKA tells us whether the completions organize information the same way across the full sequence.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 layers, GQA, SwiGLU, bf16)
- **Data**: `data/text_completions/contrastive_pairs.json` — 111 prompt groups with hand-crafted completion pairs/triples (v3: balanced immediate-divergence samples)
- **Relationship types**: antonym (67 groups), synonym (29 groups), style (8 groups), unrelated (7 groups)
- **Prefix control**: Each group tagged as `immediate` (diverge at token 1) or `shared` (shared completion prefix). This allows comparing antonyms vs synonyms with matched prefix structure.
- **Hardware**: Single B200 GPU (bf16), ~37s total runtime
- **Completion strategy**: Teacher forcing (no generation) — feed each completion's tokens as input, collect hidden states at every layer

### Data Design

Each group contains a prompt and 2–3 short completions (1–8 tokens) with a known semantic relationship:

| Type | Example Prompt | Completions | Design Principle |
|---|---|---|---|
| **Antonym (shared)** | "Is the Sun a star or a planet?" | "The Sun is a **star**" / "The Sun is a **planet**" | Shared completion prefix, differ at a single pivot word |
| **Antonym (immediate)** | "Is the Sun a star or a planet? One word." | "**Star**" / "**Planet**" | Diverge at first token — prefix-matched with synonym pairs |
| **Synonym (immediate)** | "Describe the speed of a cheetah in one word." | "**Fast**" / "**Swift**" / "**Rapid**" | Diverge immediately at first completion token |
| **Synonym (shared)** | "Describe the speed of a cheetah." | "A cheetah is **fast**" / "A cheetah is **swift**" | Shared prefix, diverge at synonym — prefix-matched with antonym pairs |
| **Style** | "Explain what gravity is." | Formal / Casual / Technical registers | Same information, different verbosity and word choice |
| **Unrelated** | "What is the capital of France?" | "**Paris**" / "**Elephant**" / "**42**" | Correct answer vs semantically nonsensical |

**Prefix control (v2→v3)**: The original v1 data had a structural confound: all antonym pairs shared completion prefixes while most synonym pairs diverged immediately. This made direct KL comparisons unreliable. Version 2 introduced prefix-controlled groups; version 3 balances sample sizes:
- **48 immediate-divergence antonym pairs** (IDs 50–110): single-word answers that diverge at token 1, matching synonym prefix structure. 48 pairs = 48 synonym immediate pairs.
- **13 shared-prefix synonym pairs** (IDs 63–75): multi-word answers with shared prefix before the synonym, matching antonym prefix structure

This creates a balanced 2×2 design (relationship × prefix structure) enabling fair comparison with equal sample sizes in the critical immediate-divergence condition.

## Methods

### Method 1: Contrastive Logit Lens

Extension of T-1. For each completion in a group, run the full prompt + completion through the model with teacher forcing. At each layer $\ell$, project the residual stream at the pivot position through the final LM head (RMSNorm + unembedding) to get logit distributions $p_A^{(\ell)}$ and $p_B^{(\ell)}$.

**Primary metric**: $D_{\text{KL}}(p_B^{(\ell)} \| p_A^{(\ell)})$ — how distinguishable are the two completions' logit distributions at each layer?

### Method 2: Hidden-State Representational Similarity

Extension of T-4. For each prompt group, collect the residual stream vectors after each layer for all completions.

**Metrics per layer per pair:**
- **Cosine similarity** at the first diverging token position
- **Normalized L2 distance** at the diverging position
- **Linear CKA** over completion tokens

### Method 3: Divergence Curve Analysis

Aggregate per-layer metrics into **divergence curves** — one curve per relationship type, showing how similarity evolves across depth.

### Method 4: Pivot Token Analysis

Track the hidden state specifically at the pivot token position across layers. Measure cosine similarity of the pivot token's representation between the two completions, plus a baseline from a shared-prefix token (which should remain ~1.0 in a causal model).

## Results

### Divergence Curves

![Divergence Curves](results/divergence_curves.png)

The four-panel overview reveals the core dynamics:

**Cosine similarity** (top-left): All relationship types follow an inverted-U trajectory — similarity rises from the embedding layer, plateaus in mid layers, then drops at the final layer. The general ordering is antonym ≈ synonym > style > unrelated, with antonym and synonym trading places depending on layer.

**KL divergence** (top-right, log scale): All types start with high KL (~14–17) at layer 0, decrease through mid layers, then rise again in late layers. Style completions show the highest late-layer KL (~20 at L34), while antonyms and synonyms are moderate (~10–12 at L34). Note: the all-groups aggregation mixes shared-prefix and immediate-divergence pairs; see the prefix-controlled analysis below for the fair comparison.

**L2 distance** (bottom-left): Mirrors cosine similarity inversely. Unrelated pairs have the highest L2 throughout.

**CKA** (bottom-right): Completion-level representational similarity. Antonyms maintain high CKA (>0.85 through L34, dropping to 0.70 at L35). Synonym and style CKA also remain relatively high, dropping more gradually in late layers.

### KL Divergence: Synonym vs Antonym

![KL Crossover](results/kl_crossover.png)

The left panel shows KL divergence across layers; the right panel shows the delta KL(synonym) − KL(antonym) per layer.

| Layer Range | Synonym KL vs Antonym KL | Observation |
|---|---|---|
| **0–3** | Antonym > Synonym | Antonyms are slightly more divergent (e.g., 16.9 vs 15.4 at L0). This reflects the v3 data composition: 48/67 antonym groups are immediate-divergence, which have higher KL than shared-prefix pairs |
| **4–17** | Mixed | The two types alternate, with synonym generally slightly higher |
| **18–23** | Antonym > Synonym | Antonyms become clearly more divergent (e.g., 4.1 vs 3.0 at L21) |
| **24–35** | Synonym > Antonym | Synonym KL rises as the model commits to surface forms; by layer 34, synonyms are 1.2× more divergent (12.3 vs 10.1) |

**Note on the all-groups aggregation**: This comparison mixes shared-prefix and immediate-divergence pairs with different proportions per relationship type (antonyms: 48 immediate + 19 shared; synonyms: 48 immediate + 39 shared). The all-groups KL curves are therefore confounded by prefix structure. The prefix-controlled analysis below isolates the genuine semantic effect by comparing only immediate-divergence pairs with matched sample sizes.

### Pivot Token Analysis

![Pivot Token Trajectories](results/pivot_token_trajectories.png)

**Left panel**: Aggregate pivot token similarity by relationship type. Antonym and synonym pivots show broadly similar cosine similarity ranges (antonym: 0.54–0.75, synonym: 0.51–0.77 across layers 0–34). With v3's balanced data (many immediate-divergence antonyms), the two types overlap substantially — immediate-divergence pairs lack the shared-context advantage that previously inflated antonym similarity.

**Right panel**: Individual traces reveal high variance within both types. Some antonym pairs diverge sharply at specific layers (visible as individual red traces dipping to 0.3), while others stay similar throughout. This suggests certain semantic distinctions are resolved at specific layers rather than gradually.

### Relationship Heatmap

![Relationship Heatmap](results/relationship_heatmap.png)

Each row is a group-pair, columns are layers (embedding through layer 35). Groups are sorted by relationship type (synonym | antonym | style | unrelated), separated by horizontal lines.

Key observations:
- **Antonym block** (second section): consistently green (high similarity) through layers 5–34, with a sharp red column at the embedding layer and layer 35
- **Synonym block** (first section): more heterogeneous — some synonym pairs maintain high similarity throughout, others diverge in mid-late layers
- **Style and unrelated blocks**: generally lower similarity (more orange/red), especially in late layers
- **Layer 35 (final)**: universally red across all types — the final layer specializes representations for specific token prediction, destroying inter-completion similarity

### Quantitative Summary (All Groups)

| Metric | Synonym (29 groups, 87 pairs) | Antonym (67 groups, 67 pairs) | Style (8 groups, 24 pairs) | Unrelated (7 groups, 21 pairs) |
|---|---|---|---|---|
| Peak cosine similarity | 0.766 (L19) | 0.747 (L16) | 0.671 (L19) | 0.649 (L34) |
| Min cosine similarity | 0.182 (emb) | 0.187 (emb) | 0.103 (emb) | 0.032 (emb) |
| KL at layer 0 | 15.4 | 16.9 | 13.8 | 16.7 |
| KL at layer 16 (min) | 2.5 | 2.3 | 3.1 | 3.6 |
| KL at layer 34 | 12.3 | 10.1 | 20.2 | 10.0 |
| Final-layer cosine (L35) | 0.409 | 0.458 | 0.289 | 0.198 |

### Prefix-Controlled Results

![Prefix-Controlled Analysis](results/prefix_controlled.png)

The central analysis of T-17 v3. By comparing immediate-divergence pairs with **balanced sample sizes** (n=48 antonym pairs, n=48 synonym pairs), we eliminate both the prefix-length confound and the sample-size imbalance.

| Layer Range | Immediate Antonym KL vs Synonym KL | Interpretation |
|---|---|---|
| **0–4** | **Antonym > Synonym** (18.4 vs 15.5 at L0, ratio 1.19×) | Antonyms (different meanings) produce more divergent logit distributions — the model distinguishes meanings before forms |
| **5–20** | Synonym > Antonym | Synonyms become more divergent as form-level differences emerge in mid layers |
| **21–23** | Antonym > Synonym | Brief reversal in the prediction-formation zone |
| **24–35** | Synonym ≫ Antonym (16.4 vs 12.0 at L34, ratio 1.36×) | Form commitment: synonyms diverge as the model selects specific output tokens |

**Key numbers for immediate-divergence pairs (n=48 each):**

| Layer | Antonym (imm.) | Synonym (imm.) | Ratio |
|---|---|---|---|
| 0 | 18.36 | 15.45 | 1.19× (ant > syn) |
| 2 | 8.86 | 7.09 | 1.25× (ant > syn) |
| 4 | 5.33 | 5.03 | 1.06× (ant > syn) |
| 16 | 2.32 | 2.74 | 0.84× (syn > ant) |
| 34 | 12.04 | 16.37 | 0.74× (syn > ant) |

The early-layer antonym > synonym signal is consistent and strengthens with balanced sampling (1.19× at L0 vs 1.16× in v2 with n=13). The late-layer synonym dominance is also stronger (1.36× at L34 vs 1.13× in v2).

**Comparison with shared-prefix antonyms:** Shared-prefix antonym KL at L0 is 13.2 (lower than both immediate conditions), confirming that shared context reduces KL regardless of semantic relationship.

## Conclusions & Key Findings

### 1. The Meaning-vs-Form Hypothesis Is Confirmed (With Balanced Prefix Control)

With prefix-length confound removed and balanced sample sizes (n=48 each), the early-layer prediction **is confirmed**: antonym KL exceeds synonym KL at layers 0–4 with a consistent ratio of 1.06–1.25×. The model genuinely distinguishes antonyms (different meanings) more than synonyms (same meaning) in the first 5 layers, consistent with early semantic processing.

The crossover at layer 5 is clean — after this point, synonym KL exceeds antonym KL as the model begins committing to surface forms. The late-layer synonym dominance (1.36× at L34) reflects genuine form commitment, not a prefix confound artifact.

There's also a secondary antonym > synonym phase at layers 21–23, which persists across both v2 (n=13) and v3 (n=48), suggesting it's a real phenomenon rather than noise.

### 2. Prefix Structure Is a Dominant Confound

Comparing the four conditions (antonym/synonym × shared/immediate) reveals that prefix structure explains more variance in KL divergence than semantic relationship:

| Condition | KL at L0 | KL at L34 |
|---|---|---|
| Antonym (shared prefix, n=19) | 13.2 | 5.4 |
| Antonym (immediate, n=48) | 18.4 | 12.0 |
| Synonym (shared prefix, n=39) | 15.3 | 7.2 |
| Synonym (immediate, n=48) | 15.5 | 16.4 |

Shared-prefix pairs have consistently lower KL than immediate pairs regardless of relationship type. At L34, the prefix effect (shared→immediate: +6.6 for antonyms, +9.2 for synonyms) is larger than the relationship effect (antonym→synonym: +4.4 for immediate, +1.8 for shared). This highlights a methodological lesson: **any experiment comparing contrastive completions must control for prefix structure**.

### 3. Context Dominates Token Identity in the Residual Stream

Shared-prefix antonym pairs (e.g., "star" vs "planet" after identical context "The Sun is a ") maintain cosine similarity > 0.63 across layers 2–34 (typically > 0.70, with a dip to 0.64 at layer 21). The residual stream at a given position is overwhelmingly determined by context, not by the specific token embedded there. This confirms the "residual stream as information highway" view — the token embedding is a small perturbation on a context-dominated representation.

### 4. Layer 35 Is a Universal Discriminator

All relationship types show a dramatic cosine similarity drop at the final layer (35). This aligns with T-1's finding that layer 35 is where the model makes its final token prediction — it must maximally separate all alternatives. T-4 explains the geometry: layer 35 expands dimensionality (PR jumps to 119) and reduces mean cosine similarity from 0.62 to 0.09, dispersing representations for the vocabulary projection.

### 5. The KL "Smile" Pattern

KL divergence follows a U-shape for all conditions: high at layer 0 (embeddings are very different → logits are noisy), decreasing through mid layers (representations converge), then rising again in late layers (form commitment). The U-shape is steeper for immediate-divergence pairs than shared-prefix pairs, confirming that shared context dampens KL throughout.

### 6. Two-Phase Processing with Mid-Layer Reversal

The immediate-divergence data reveals a clear two-phase structure:
1. **Layers 0–4 (semantic discrimination)**: Antonyms produce more divergent logits than synonyms (ratio 1.06–1.25×). The model processes meaning before form.
2. **Layers 5–35 (form commitment)**: Synonyms become increasingly more divergent, peaking at 1.36× at L34.

A brief reversal at layers 21–23 (antonym > synonym) is reproducible across v2 and v3. This coincides with T-4's deep dimensionality collapse region (PR 1.6–7.5), where representations are squeezed into very few dimensions — possibly forcing a re-evaluation of semantic content before final form commitment.

## Usage

```bash
# No prerequisites — contrastive_pairs.json is hand-crafted, no model inference for data
poetry run python experiments/t17_contrastive_trajectories/run.py
```

Optional flags:
```bash
--device cuda:0          # GPU selection (default: cuda:0)
--output-dir results/    # Output directory (default: experiments/t17_contrastive_trajectories/results/)
```

Runtime: ~37s on a single B200 GPU.

Output:
- `results/summary.json` — Per-layer metrics aggregated by relationship type
- `results/prefix_summary.json` — Per-layer metrics aggregated by relationship × prefix_group
- `results/full_results.json` — Per-pair, per-layer metrics for all 111 groups
- `results/divergence_curves.png` — Four-panel overview (cosine, KL, L2, CKA)
- `results/kl_crossover.png` — Synonym vs antonym KL (all pairs)
- `results/prefix_controlled.png` — Prefix-controlled meaning-vs-form test
- `results/pivot_token_trajectories.png` — Pivot token analysis
- `results/relationship_heatmap.png` — Group × layer cosine heatmap

## Connections to Other Experiments

- **T-1 (Logit Lens)**: T-17 extends the logit lens to *contrastive* settings — instead of asking "when does the correct token appear?", we ask "when do alternative tokens separate?" The two-phase pattern (semantic discrimination L0–4, form commitment L5–35) aligns with T-1's finding that the final 4 layers account for the bulk of accuracy gains.
- **T-4 (Residual Stream Geometry)**: T-17 uses the same geometric tools (cosine similarity, norms, CKA) but applied to *paired* completions rather than single-sequence statistics. T-4's bimodal dimensionality collapse (layers 1–5 and 16–24) explains why shared-prefix antonyms maintain high cosine similarity — during collapse regions, representations are squeezed into so few dimensions that different tokens are forced into the same subspace. The layer 35 universal discriminator behavior in T-17 aligns with T-4's final-layer de-anisotropification (PR 119, cosine 0.09).
- **T-3 (Layer Swap Cost)**: The form-commitment layers (24–35) where synonym KL rises should correspond to expensive swap regions in T-3's cost matrix — swapping these layers would disrupt the model's surface-form selection.
- **T-7 (Linearization Gap)**: The mid-layer reversal (layers 21–23, antonym > synonym) overlaps with T-7's rising nonlinearity region, suggesting that this reversal involves nonlinear computation. The late layers (24–35) where form commitment occurs have high Jacobian consistency (T-7), consistent with a more uniform transformation applied regardless of content.
