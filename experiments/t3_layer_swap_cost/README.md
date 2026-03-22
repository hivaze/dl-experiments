# T-3: Layer Swap Cost Matrix

## Motivation & Research Question

**How interchangeable are individual decoder layers?** For every pair of layers (i, j), swap their positions and measure loss degradation. This produces a 36×36 "swap cost" matrix — a functional distance between layers that reveals which layers are tightly bound to their position and which are freely relocatable.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, bf16)
- **Hardware**: NVIDIA B200, single GPU (`cuda:0`)
- **Evaluation data**: 28 question/instruction-format prompts with greedy completions (vLLM, temp=0, max 512 tokens). System message prevents echo repetition. Loss computed **only on completion tokens**.
- **Total pairs**: 630 unique swaps (36 choose 2)
- **Baseline loss**: 0.0877
- **Runtime**: ~4 minutes

## Methods

For each unique pair (i, j) with i < j:
1. Swap layers i and j in the model's `ModuleList`
2. Forward the full prompt+completion, compute cross-entropy loss only on completion token positions (with `use_cache=False`)
3. Record Δloss = swapped_loss − baseline_loss

Analysis includes per-layer average swap cost, distance dependence, hierarchical clustering, and interchangeability assessment.

## Results

### Most costly swaps

| Swap | Δ Loss | Loss |
|------|--------|------|
| (0, 35) | +22.83 | 22.92 |
| (6, 35) | +17.81 | 17.89 |
| (2, 22) | +16.28 | 16.37 |
| (7, 35) | +16.27 | 16.36 |
| (2, 28) | +16.18 | 16.27 |

Layer 35 and layers 0–2 dominate the costliest swaps. The (0,35) swap is catastrophic — 261x baseline loss.

### Cheapest swaps

| Swap | Δ Loss | Ratio |
|------|--------|-------|
| (32, 33) | +0.059 | 1.67x |
| (26, 27) | +0.068 | 1.77x |
| (25, 26) | +0.072 | 1.83x |
| (27, 28) | +0.082 | 1.94x |
| (31, 32) | +0.096 | 2.09x |

The cheapest swaps are exclusively **adjacent late-middle pairs** (layers 25–33). Even the cheapest swap (32↔33) still causes a 67% loss increase — there are **no truly interchangeable pairs** with completion-token evaluation.

### Interchangeability

**0 of 630 pairs** (0%) have <1% loss increase — down from 148 pairs (23.5%) with prompt-token evaluation. Every single layer swap measurably disrupts the model's ability to reproduce its own greedy output. The "fungible middle" identified with prompt-token loss was an artifact of measuring on trivially-predictable template tokens.

### Per-layer position sensitivity

| Region | Layers | Avg Swap Cost |
|--------|--------|---------------|
| Most sensitive | 2, 1, 35, 0 | 9.5–11.1 |
| Early transition | 3–6 | 5.3–8.7 |
| Middle | 7–30 | 2.5–3.7 |
| Late | 31–34 | 3.2–6.8 |

The U-shape persists but is much steeper: the most sensitive layers (0–2, 35) have 4x the swap cost of the middle layers (vs 3x with prompt-token loss). Layer 2 is now the most position-sensitive (avg cost 11.15), surpassing even layer 35 (10.36).

### Swap cost vs layer distance

| Distance | Avg Δ Loss | Ratio vs baseline |
|----------|-----------|-------------------|
| \|i−j\| = 1 | +0.57 | 7.5x |
| \|i−j\| = 5 | +2.08 | 24.7x |
| \|i−j\| = 10 | +3.60 | 42.1x |
| \|i−j\| = 20 | +7.29 | 84.1x |
| \|i−j\| = 35 | +22.83 | 261x |

Even adjacent swaps (distance 1) cause a 7.5x average loss increase — the model is extremely sensitive to layer ordering.

### Hierarchical clustering (Ward linkage)

With 4 clusters:
| Cluster | Layers | Role |
|---------|--------|------|
| Singleton | [0] | Embedding interface — unique |
| Early | [1, 2, 3, 4, 5, 6] | Early processing |
| Middle | [7–30] | Core computation (24 layers) |
| Late | [31–35] | Output preparation |

The clustering structure is identical to prompt-token evaluation — the architectural zones are robust even though the sensitivity within them is much higher.

## Conclusions & Key Findings

1. **Zero interchangeable pairs**: With completion-token evaluation, every swap causes at least 67% loss increase. The 148 "interchangeable" pairs (23.5%) found with prompt-token loss were entirely an artifact of measuring on easily-predictable template tokens.

2. **The 3-zone architecture is real but not fungible**: The same clustering (early/middle/late) appears with both evaluation methods, but the middle layers are not interchangeable — they're just less position-sensitive than boundary layers. "Less sensitive" ≠ "interchangeable."

3. **Adjacent swaps in the late middle are cheapest**: Layers 25–33 can be swapped with their neighbors for ~67–94% loss increase — painful but not catastrophic. This suggests some local redundancy in the output preparation layers.

4. **Layer 2 is the most position-sensitive**: With completion-token loss, layer 2 surpasses even layer 35 in average swap cost (11.15 vs 10.36). This was hidden by prompt-token evaluation where layer 2 appeared moderately sensitive.

5. **Distance dependence is steeper**: Swap cost scales more aggressively with distance under completion-token evaluation. Even distance-1 swaps cause 7.5x loss increase on average.

6. **Practical implication**: Pipeline parallelism with arbitrary layer reordering is infeasible when the model must maintain coherent generation quality. Only adjacent swaps in the 25–33 range are remotely viable, and even those degrade quality substantially.

## Artifacts

- `results/swap_cost_raw.json` — Full 36×36 swap loss and delta matrices
- `results/swap_cost_analysis.json` — Clustering, per-layer costs, equivalence classes
- `results/swap_cost_heatmap.png` — Heatmap of swap cost matrix
- `results/swap_cost_heatmap_log.png` — Log-scale heatmap
- `results/swap_cost_pct_change.png` — Percentage loss change heatmap
- `results/swap_cost_vs_distance.png` — Scatter + mean with ±1 std band
- `results/per_layer_avg_swap_cost.png` — Bar chart of per-layer position sensitivity
- `results/swap_cost_dendrogram.png` — Hierarchical clustering dendrogram

## Usage

```bash
# Generate completions first (one-time):
poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Instruct-2507

# Run experiment:
poetry run python experiments/t3_layer_swap_cost/run.py
```
