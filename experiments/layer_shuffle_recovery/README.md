# Layer Shuffle Recovery

Can we recover the correct order of transformer decoder layers after shuffling them, without knowing the original order?

## Setup

**Model**: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) — a standard decoder-only causal LM.

| Property | Value |
|----------|-------|
| Architecture | Qwen3ForCausalLM |
| Decoder layers | 28 (homogeneous, all full-attention) |
| Hidden size | 2048 |
| Attention | GQA — 16 query heads, 8 KV heads, head_dim=128 |
| MLP | SwiGLU (gate_proj + up_proj → SiLU → down_proj), intermediate=6144 |
| Normalization | RMSNorm (pre-norm: input_layernorm, post_attention_layernorm, per-head q/k_norm) |
| Params per layer | ~25M |
| Total size | ~1.7B params, ~3.4GB in bf16 |

All 28 layers are randomly permuted (seed=42). Each method attempts to recover the original ordering.

## Calibration data

Dataset-based methods require calibration data — text that the model can produce meaningful loss gradients on.

**Generation pipeline**:
1. Load 64 questions from [OpenAI GSM8K](https://huggingface.co/datasets/openai/gsm8k) (grade-school math)
2. Generate completions using vLLM rollouts from the **original** (unshuffled) Qwen3-1.7B at temperature=0.7, top_p=0.9, max_tokens=256
3. Concatenate prompt + completion, tokenize with padding to length 256
4. Save as `input_ids.pt` (shape: 64 × 256)

Using the model's own rollouts ensures the calibration text lies in the model's distribution — loss gradients are most informative on text the model would naturally produce. GSM8K provides diverse reasoning chains with varied token patterns.

## Evaluation metrics

### Why Kendall tau?

**Kendall tau** (τ) measures rank correlation between two orderings. It counts the fraction of concordant vs discordant pairs:

```
τ = (concordant_pairs - discordant_pairs) / total_pairs
```

A **concordant pair** (i,j) means both orderings agree on who comes first. τ ranges from -1 (perfectly reversed) through 0 (random) to +1 (identical ordering).

This is the right metric because:
- It captures **global ordering quality**, not just local correctness
- A recovered order with τ=0.9 has the right relative ordering for 95% of all layer pairs, even if few layers land in their exact position
- **Positional accuracy** (% of layers at exact correct position) is too strict — shifting one layer cascades errors across all displaced positions, giving 0% accuracy for a nearly-perfect recovery
- **Mean displacement** complements τ by showing how far off misplaced layers are on average

## Results

Model: Qwen/Qwen3-1.7B, 28 layers, seed=42.

| Method                    | Type    | Kendall τ | Accuracy | Avg Displ. |  Time |
|---------------------------|---------|----------:|---------:|-----------:|------:|
| pairwise_bubble           | dataset |   +1.0000 |   100.0% |       0.00 | 18.9s |
| remove_reinsert           | dataset |   +1.0000 |   100.0% |       0.00 | 72.9s |
| activation_flow           | math    |   +0.9735 |    67.9% |       0.36 |  0.2s |
| ensemble_rank             | math    |   +0.8360 |    35.7% |       1.64 |  0.1s |
| weight_stats_continuity   | math    |   +0.8148 |    39.3% |       1.86 |  0.0s |
| simulated_annealing       | dataset |   +0.7831 |    17.9% |       2.50 | 18.7s |
| tsp_full_weights          | math    |   +0.7407 |    25.0% |       2.71 |  0.0s |
| greedy_cosine             | math    |   +0.6931 |     3.6% |       3.43 |  0.0s |
| svd_spectrum              | math    |   +0.6614 |     3.6% |       3.50 |  0.0s |
| layernorm_progression     | math    |   +0.6349 |    42.9% |       3.29 |  0.0s |
| fiedler_spectral          | math    |   +0.5820 |    28.6% |       3.93 |  0.0s |
| causal_pairwise           | math    |   +0.3915 |     0.0% |       6.14 |  0.3s |
| greedy_perplexity         | dataset |   +0.3016 |     0.0% |       6.86 |  1.1s |

## Methods (ordered worst → best)

### 13. Greedy Perplexity — τ = 0.30 | 0% accuracy | dataset

Build the ordering one layer at a time. At each position, try every remaining candidate layer, pick the one giving lowest next-token prediction loss.

**Algorithm**: Cache the hidden state after all previously-decided layers. For each candidate `c` at position `k`:

```
h_c = Layer_c(cached_hidden)
logits = lm_head(RMSNorm(h_c))
loss_c = CrossEntropy(logits[:, :-1], input_ids[:, 1:])
Pick c* = argmin(loss_c)
```

Uses cached prefix for efficiency: N + (N-1) + ... + 1 = 406 single-layer forwards total.

**Why it fails**: After only `k` layers (instead of 28), the hidden state is "undercooked" — the model wasn't trained for partial execution. The loss signal is very noisy, especially for early positions where most of the model is missing. The method commits to greedy choices it can never undo.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  2  5  0 26 25 18  8  4 13 10 16  3  9 17  7 19 27  6 11 12 14 15  1 20 22 21 24 23
            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0/28 correct — complete scramble
```

Every position is wrong. The greedy choices compound: picking the wrong first layer poisons all subsequent decisions.

---

### 12. Causal Pairwise Test — τ = 0.39 | 0% accuracy | math

For each pair (i, j), pass random input through layer i then j, and compare to j then i. The "correct" order should produce more stable activations.

**Algorithm**: Pre-compute single-layer outputs for all 28 layers on random input `h ~ N(0, 0.01)`. For each pair:

```
h_ij = Layer_j(Layer_i(h))
h_ji = Layer_i(Layer_j(h))
If std(h_ij) ≤ std(h_ji): layer i "wins" over j
```

Count total wins per layer, rank by win count (more wins = earlier position). C(28,2) = 378 pair tests.

**Why it fails**: The hypothesis that correct ordering produces lower activation variance is weak in a residual architecture. Each layer adds a small perturbation via the residual connection `output = input + attention(input) + mlp(input)`, so the output std is dominated by the input magnitude regardless of ordering.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  8 11  0 16  9 12 10  1  5 18 15 13  6  7 17 20 21 14  4 22 19 24 23  2 27  3 25 26
            ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  0/28 correct — pairwise stability is not order-sensitive
```

---

### 11. Fiedler Spectral Ordering — τ = 0.58 | 28.6% accuracy | math

Use spectral graph theory for a globally optimal ordering instead of greedy chain-building.

**Algorithm**: Build a similarity graph using a Gaussian kernel over weight-stats L2 distances:

```
S[i,j] = exp(-D²[i,j] / (2σ²))    where σ = median(D)
```

Compute the graph Laplacian `L = diag(S·1) - S`. The **Fiedler vector** is the eigenvector corresponding to the 2nd smallest eigenvalue of `L`. Sorting nodes by Fiedler values gives a 1-D embedding that minimizes a global balanced-cut objective — similar nodes are placed adjacent.

**Why it partially works**: The Fiedler vector provides a principled global ordering (unlike greedy methods that can lock in early mistakes). But the Gaussian kernel bandwidth σ is set heuristically to the median distance, and the method can't capture the directed nature of the layer sequence.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  8 13 14  6  7 11  9  1 10 12  5  3  2  0  4 16 15 17 18 19 20 21 22 23 24 27 25 26
            ░░░░░░░░░░░░░░░░░████████░░░  8/28 — late layers (17-24) correct, early layers scrambled
```

Gets layers 17-24 right (high weight-norm separation) but fails on 0-16 where the similarity graph is nearly uniform.

---

### 10. LayerNorm Progression — τ = 0.63 | 42.9% accuracy | math

RMSNorm parameters evolve smoothly across depth — adjacent layers should have similar norm weights.

**Algorithm**: For each layer, concatenate all norm weight vectors:

```
v_i = [input_layernorm.weight,        # (2048,)
       post_attention_layernorm.weight, # (2048,)
       q_norm.weight,                   # (128,)
       k_norm.weight]                   # (128,)
```

Total: 4352 dims per layer. Compute pairwise **L2 distance** matrix, solve open-path TSP via nearest-neighbor + 2-opt.

**Key detail**: Must use L2, not cosine distance. Cosine is scale-invariant and destroys the magnitude progression signal — an earlier version using cosine distance got τ = -0.41 (worse than random, i.e. reversed).

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered: 14 13 11  9  8  3  6  5  0  1  4  2  7 10 12 16 15 17 18 19 20 21 22 23 24 25 26 27
            ░░░░░░█░░░░░░░░░░███████████  12/28 — layers 17-27 perfect, 0-16 scrambled
```

All errors are in layers 0-16. The norm weight vectors for these layers are so similar that L2 distances within this group provide weak ordering signal. Layers 17+ have rapidly diverging norm parameters.

---

### 9. SVD Spectrum Similarity — τ = 0.66 | 3.6% accuracy | math

The singular value spectrum of weight matrices changes gradually between adjacent layers.

**Algorithm**: For each layer, compute top-32 singular values of 4 major weight matrices on GPU:

```
σ_i = concat([svdvals(q_proj)[:32],
              svdvals(k_proj)[:32],
              svdvals(v_proj)[:32],
              svdvals(gate_proj)[:32]])    # 128-d vector
```

Build L2 distance matrix, solve TSP.

**Why it partially works**: SVD spectra capture the effective rank and energy distribution of each linear transformation, which evolves across depth as the model learns increasingly abstract features. Using 4 matrices instead of only q_proj (an earlier version) improved from τ ≈ 0 to τ ≈ 0.66 by providing more discriminative information.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  2  1  3  4  7  6  9  5 10  8 12 14 15 17 18 19 20 16 11 13 26 25 21 24 22 23 27  0
            ░█░░░░░░░░░░░░░░░░░░░░░░░░░░  1/28 — relative order roughly correct but shifted
```

Only layer 1 lands exactly. The chain is roughly in the right shape but systematically offset, particularly in the tail (layers 20-27 are shuffled among themselves).

---

### 8. Greedy Cosine Chain — τ = 0.69 | 3.6% accuracy | math

Start from the most "isolated" layer and greedily chain by highest cosine similarity.

**Algorithm**: Flatten all ~25M weights per layer into one vector. Compute cosine similarity matrix. Identify the layer with the lowest average similarity to all others (likely an endpoint of the chain). Greedily extend the chain by always picking the most similar unused layer.

**Difference from TSP on Full Weights (Method 6)**: Same distance matrix, but no 2-opt refinement and only one starting point (the most isolated). Simpler but slightly worse since greedy can lock in early mistakes that 2-opt would fix.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  1  2  3  5  4  8  7 10  9 11 12 13 14 16 15 17 18 19 20 21 22 26 25 24 23 27  6  0
            ░░░░█░░░░░░░░░░░░░░░░░░░░░░░  1/28 — mostly off-by-1 pairs: (1↔2), (7↔8), (15↔16)
```

The chain shape is correct (smooth τ = 0.69) but local pairs are frequently swapped. Without 2-opt, these adjacent inversions persist. Layers 0 and 6 are placed at the wrong end entirely.

---

### 7. TSP on Full Weights — τ = 0.74 | 25.0% accuracy | math

Treat each layer's full parameter vector as a point in ~25M-dimensional space. Adjacent layers should be closest.

**Algorithm**: Flatten all weights per layer into a single vector. Compute pairwise cosine distance matrix. Solve the open-path Traveling Salesman Problem:

1. **Nearest-neighbor heuristic**: Try all 28 starting points, build path by always visiting the nearest unvisited node, keep the shortest
2. **2-opt improvement**: Repeatedly try reversing sub-segments of the path; accept if total distance decreases

**Why cosine > L2 here**: For 25M-dimensional vectors, L2 distance is dominated by the norm difference (which is the same signal as Method 1). Cosine distance is scale-invariant and captures structural similarity in the weight patterns, providing complementary information.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  0  5  6  8  3  4  1  2  7 10  9 11 12 13 14 15 17 19 21 22 24 26 25 23 27 20 18 16
            █░░░░░░░░░░█████░░░░░░░█░░░░  7/28 — two clean stretches, but early and late scrambled
```

Layers 11-15 and 23 are exactly correct. The 2-opt refinement fixes many inversions that greedy_cosine misses, but it can't resolve the deep early-layer confusion (layers 1-10 mixed up) or the reversed tail (16, 18, 20 at the end).

---

### 6. Simulated Annealing — τ = 0.78 | 17.9% accuracy | dataset

Start from the best math method result, propose random swaps, accept/reject by the Metropolis criterion on next-token prediction loss.

**Algorithm**: Seeded from activation_flow result (τ ≈ 0.97). Low initial temperature T₀ = 0.3 (the seed is already good — we're fine-tuning, not exploring). 1500 iterations:

```
1. Propose swap: 75% near-adjacent (offset ±1 or ±2), 25% random
2. Compute Δloss = loss(new_order) - loss(current_order)
3. Accept if Δloss < 0, or with probability exp(-Δloss / T)
4. T = T₀ · (1 - iter/1500)
```

Track best-ever ordering across all iterations.

**Why it underperforms bubble sort**: SA explores stochastically from a near-perfect seed. Most random swaps make things worse, and the probabilistic acceptance criterion can undo good states. Deterministic bubble-sort sweeps are more efficient when the seed has only a few errors.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  0  2  3  4  5  8 10 11  9  6 13  7 14 15 16 17 19 18 20 12  1 21 23 24 22 25 26 27
            █░░░░░░░░░░░░░░░░░░░░█░░░███  5/28 — random swaps moved layers 1 and 12 far away
```

Layers 1 and 12 ended up at positions 20 and 19 — the stochastic acceptance at high temperature accepted bad swaps that moved them far from their correct positions, then couldn't recover.

---

### 5. Weight Statistics Continuity — τ = 0.81 | 39.3% accuracy | math

Aggregate scalar statistics per layer evolve smoothly across depth.

**Algorithm**: For each layer, compute a 4-d feature vector over all parameters:

```
f_i = [‖W_i‖_F, mean(W_i), std(W_i), max|W_i|]
```

Build pairwise L2 distance matrix `D[i,j] = ‖f_i - f_j‖₂`. Solve open-path TSP.

**Why it works well**: Training creates a monotonic progression in weight magnitudes — later layers develop larger weights as they carry more abstracted representations. Layers 17-27 have norms 314-1140 with clear separation. **Limitation**: Layers 0-16 cluster in a narrow band (248-265), making fine-grained ordering within this group difficult with only 4 scalar features.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  0  4  2  5  1 10 12  7  6 14  8 13 11  9  3 16 15 17 18 19 20 21 22 23 24 27 25 26
            █░█░░░░█░░░░░░░░░████████░░░  11/28 — layers 17-24 nailed, early layers mixed
```

The "norm plateau" problem is clearly visible: layers 0-16 are shuffled within their group (norms 248-265, nearly indistinguishable), while layers 17-24 with large norm gaps are placed correctly. Layers 25-27 are swapped among themselves.

---

### 4. Ensemble Rank Averaging — τ = 0.84 | 35.7% accuracy | math

Average rank positions across multiple distance metrics for a robust consensus ordering.

**Algorithm**: Compute 4 independent chains using different distance matrices (weight stats L2, SVD spectrum L2, full-weight cosine, norm weights L2). Each chain assigns a position to each layer. For each shuffled layer s:

```
avg_pos[s] = mean(position_of_s in chain_k, for k = 1..4)
```

Sort layers by average position.

**Why it helps**: Different metrics capture different aspects of the layer progression — weight norms are best for late layers (17-27), SVD spectra capture mid-range structure, cosine distance captures local neighborhoods, norm weights capture normalization patterns. Averaging cancels out per-method biases and errors.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  5  4  2  1  6  3  8  7  9 14 10  0 12 11 13 15 17 19 16 18 20 21 22 24 25 23 26 27
            ░░█░░░░█░░█░█░░█░░░░███░░░██  10/28 — scattered exact matches, consistent relative order
```

More exact positions than any single constituent method. The errors that remain (e.g., 0↔5 swap, 16↔19 region) are cases where all 4 metrics agree on the wrong relative position.

---

### 3. Activation Flow Analysis — τ = 0.97 | 67.9% accuracy | math | BEST MATH METHOD

Each layer transforms activations in a unique way. Instead of looking at static weights, pass random hidden states through each layer independently and fingerprint the transformation using high-dimensional channel-wise vectors.

**Algorithm**: For each of 4 random probes at different scales (0.05, 0.1, 0.5, 1.0):

```
For probe p with scale s:
    h ~ N(0, s²), shape (2, 32, 2048)
    For each layer i:
        h' = Layer_i(h, use_cache=False)
        r = h' - h                          # residual contribution
        μ_c = mean(r, dims=(batch, seq))     # channel-wise mean, shape (2048,)
        σ_c = std(r, dims=(batch, seq))      # channel-wise std, shape (2048,)

fingerprint_i = concat(μ_c, σ_c for all 4 probes)  # 16384-d vector
```

Build L2 distance matrix on 16384-d fingerprints, solve TSP.

**Why this dominates all other math methods**: Layers 0-16 have nearly identical total weight norms (span of 17 units out of ~250), making them indistinguishable by any scalar statistic. But each layer modifies *specific channels differently* — layer 3 might amplify channels 100-200 while layer 4 amplifies channels 150-250. The full 2048-d channel-wise residual vector captures this unique functional signature. Multi-scale probing (4 different input magnitudes) tests different activation regimes, adding robustness against scale-dependent effects in the SwiGLU MLP and attention mechanisms.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  0  2  1  5  3  4  6  8  7  9 10 11 12 13 14 16 15 17 18 19 20 21 22 23 24 25 26 27
            █░░░░░█░░██████░░███████████  19/28 — only adjacent-pair swaps remain
```

The remaining 9 errors are all **adjacent-pair inversions**: (1↔2), (3↔5, 4→shifted), (7↔8), (15↔16). The method gets the correct neighborhood for every layer but can't resolve which of two very similar adjacent layers comes first — that requires loss-based comparison with actual data.

---

### 2. Remove-Reinsert Refinement — τ = 1.00 | 100% accuracy | dataset

Starting from the best math method result, refine by removing each layer and reinserting at the optimal position.

**Algorithm**: Seeded from activation_flow (τ ≈ 0.97). Uses 32 calibration samples of length 196 for strong signal. For each pass over all 28 positions:

```
For position k = 0..27:
    removed = order[k]
    remaining = order[:k] + order[k+1:]     # 27 elements
    For each insertion position p = 0..27:
        trial = remaining[:p] + [removed] + remaining[p:]
        losses[p] = full_forward_loss(trial)
    order = insert removed at p* = argmin(losses)
```

Repeat passes until no moves improve loss (typically 1-2 passes).

**Why it achieves 100%**: Starting from a near-perfect seed (only ~2-3 misplacements), the loss landscape around the correct order is informative — the model is nearly functional, so adding/removing a single layer at the wrong position causes measurable loss increase. Remove-reinsert can fix large displacements in one move (e.g., a layer displaced by 8 positions), unlike bubble sort which would need 8 passes. Uses 32 x 196 = 6272 tokens of calibration data for robust loss estimates.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
            ████████████████████████████  28/28 — perfect recovery
```

---

### 1. Pairwise Bubble Refinement — τ = 1.00 | 100% accuracy | dataset | BEST OVERALL

Starting from the best math method result, refine by deterministic bubble-sort sweeps: test each adjacent pair, swap if it reduces loss.

**Algorithm**: Seeded from activation_flow (τ ≈ 0.97). Uses 32 calibration samples of length 196.

```
Phase 1 — Adjacent bubble passes (up to 5 passes):
    For i = 0..26:
        trial = swap(order[i], order[i+1])
        If loss(trial) < loss(order) - ε:
            order = trial
    Stop when a full pass makes 0 swaps

Phase 2 — Non-adjacent sweep:
    For all pairs (i, j) where j > i+1:
        trial = swap(order[i], order[j])
        If loss(trial) < loss(order) - ε:
            order = trial
    Repeat until no improvement
```

**Why it achieves 100%**: The seed has τ = 0.97 with only ~2-3 layer misplacements. Adjacent bubble passes fix local inversions (layer k+1 placed before layer k). The non-adjacent sweep catches any remaining distant misplacements. Each full forward pass through the near-correct 28-layer model produces meaningful next-token prediction loss, so even swapping two similar early layers produces a detectable loss difference with 32 x 196 tokens. Faster than remove-reinsert (19s vs 73s) because it only tests swaps, not all possible insertion positions.

```
ideal:      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
recovered:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27
            ████████████████████████████  28/28 — perfect recovery
```

## Key findings

1. **Math-only methods can achieve τ = 0.97** using high-dimensional activation fingerprints. The key insight is that channel-wise residual vectors discriminate layers that are indistinguishable by scalar statistics.

2. **Loss-based refinement achieves perfect recovery** when seeded from a good math result. The critical insight is that dataset methods fail completely when starting from random (the model is too broken for meaningful loss gradients) but excel at fine-tuning a near-correct order.

3. **Weight norm plateau is the main obstacle**: Layers 0-16 have nearly identical total norms (248-265), making them impossible to separate with aggregate statistics. Multi-probe channel-wise fingerprints solve this.

4. **The optimal pipeline** is: activation_flow (0.2s) → pairwise_bubble (19s) = **perfect recovery in ~19 seconds**.

## Chain orientation

Math methods produce unordered chains (could be forward or reversed). Orientation is determined by computing the **Spearman rank correlation** between position-in-chain and total weight norm for all layers in the chain. Trained transformers have weight norms that correlate positively with depth (later layers are larger), so positive correlation means the chain is oriented correctly; negative means it should be reversed.

## Usage

### 1. Generate calibration data (one-time)

```bash
poetry run python experiments/layer_shuffle_recovery/generate_calibration.py
```

### 2. Run the experiment

```bash
poetry run python experiments/layer_shuffle_recovery/run.py
```

Results saved to `experiments/layer_shuffle_recovery/results/results.json`.

## Implementation details

- **layer_idx corruption**: After shuffling `model.model.layers`, each attention module retains its original `layer_idx` (used for KV cache). All forward passes use `use_cache=False`.
- **Weight features are cached**: A `WeightFeatures` class precomputes all distance matrices once, shared across methods 1-5 and 7-8.
- **SVD runs on GPU**: 2048×2048 SVD on GPU is ~10x faster than CPU.
- **Greedy perplexity caches prefix hidden states**: Only 1 new layer forward per candidate.
- **Dataset methods are seeded**: Pairwise bubble, SA, and remove-reinsert all start from the best math method result. Without seeding, they all fail (τ < 0.3) because loss is meaningless when the model is completely broken.
