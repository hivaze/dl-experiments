# T-2: Layer Knockout / Criticality Mapping

## Motivation & Research Question

Which layers in a transformer are truly critical, and which are redundant? In a 36-layer decoder-only model, every layer adds parameters and compute — but do they all contribute equally to the output? This experiment answers that question through systematic ablation: single-layer knockout, pair knockout (to detect inter-layer circuits), and greedy pruning (to find the structured pruning ceiling).

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, bf16, 2560 hidden dim)
- **Hardware**: NVIDIA B200 (183GB), `cuda:1`
- **Evaluation data**: 50 question/instruction-format prompts with greedy completions generated via vLLM (temp=0, max 512 tokens). System message prevents echo repetition.
- **Loss metric**: Cross-entropy computed **only on completion tokens** (positions after the prompt). This is far more sensitive than prompt-token loss — the model must predict its own continuation exactly, leaving no room for redundancy.
- **Baseline loss**: 0.0894 (very low — the model reproduces its own greedy output nearly perfectly)

## Methods

### Part 1: Single Layer Knockout

For each of the 36 layers $i$, we physically remove it from the model's `ModuleList` and run inference on all calibration data. The loss ratio $\frac{L_{\text{knockout}_i}}{L_{\text{baseline}}}$ quantifies how critical layer $i$ is.

**Why remove rather than zero out?** Removing the layer is equivalent to replacing it with the identity function on the residual stream (since each transformer layer computes $h_{i+1} = h_i + \text{Attn}(h_i) + \text{MLP}(h_i)$, skipping layer $i$ means $h_{i+1} = h_i$). This is the cleanest ablation — no artifacts from zeroed-out attention patterns or MLP outputs.

### Part 2: Pair Layer Knockout

We test 94 selected layer pairs and compute **synergy**: $S(i,j) = \Delta L_{i,j} - (\Delta L_i + \Delta L_j)$

- $S > 0$ (super-additive): the pair is worse together than the sum of their individual damages — they form a circuit where each layer's output depends on the other.
- $S < 0$ (sub-additive): removing both is less damaging than expected — the layers partially compensate for each other.
- $S \approx 0$ (additive): the layers contribute independently.

**Pair selection strategy**: Adjacent pairs (all 35), critical×critical (top 5 choose 2), redundant×redundant (bottom 5 choose 2), critical×redundant (top 3 × bottom 3), and evenly spaced (step 4). This covers local interactions, hub detection, and long-range coupling with 94 pairs instead of 630.

### Part 3: Greedy N-Layer Pruning

Iteratively remove the least-critical remaining layer (recomputing criticality at each step, since removing one layer changes the criticality landscape). Stop when loss exceeds 3× baseline. This gives an upper bound on structured pruning: how many layers can you remove while keeping the model functional?

## Results

### Single Layer Knockout — Criticality Profile

![Single layer knockout criticality profile](results/single_knockout_overview.png)

| Layer | Loss | Ratio | Notes |
|-------|------|-------|-------|
| 0 | 8.905 | **99.6×** | Catastrophically critical |
| 6 | 1.940 | **21.7×** | Second most critical |
| 1 | 0.707 | 7.9× | |
| 12 | 0.644 | 7.2× | |
| 10 | 0.633 | 7.1× | |
| ... | ... | ... | |
| 32 | 0.283 | 3.2× | |
| 34 | 0.254 | 2.8× | |
| 3 | 0.246 | 2.7× | |
| 2 | 0.218 | **2.4×** | Least critical |

**Key finding**: With completion-token loss, **every layer is critical** — even the least critical layer (2) causes a 2.4× loss increase. The model's ability to reproduce its own greedy output is tightly dependent on all 36 layers.

Layer 0 is catastrophically critical (~100×) — removing it alone makes the model nearly non-functional. This makes sense: layer 0 is the first to process raw embeddings and must establish the residual stream's initial structure (subword → contextual representation). All downstream layers depend on its output distribution.

Layer 6 is similarly devastating (22×), and as the pair analysis shows, it serves as a computational hub.

### Pair Knockout — Synergy

![Pair knockout heatmaps — loss delta and synergy matrices](results/pair_knockout_heatmap.png)

Most super-additive pairs:

| Pair | Synergy | Notes |
|------|---------|-------|
| (5, 6) | +3.53 | Layers 5+6 form a critical circuit |
| (6, 7) | +2.04 | Layer 6 hub |
| (1, 6) | +1.70 | Layer 6 hub |
| (1, 2) | +1.60 | Early layers interdependent |
| (6, 12) | +1.45 | Layer 6 long-range synergy |

Layer 6 appears in 4 of the top 5 synergistic pairs — it's the central hub of the model's critical circuitry. The (5,6) pair has the highest synergy (+3.53), meaning their combined removal is 3.53 units of loss *worse* than the sum of their individual effects. This implies a tightly coupled computational circuit: layer 5 produces intermediate representations that layer 6 specifically consumes, and neither can compensate for the other's absence.

The (6,12) pair is particularly interesting — these layers are 6 apart, suggesting long-range functional coupling. Layer 6 may produce features that are specifically consumed by layer 12, bypassing the intervening layers via the residual stream.

### Greedy Pruning

| Step | Layer Removed | Loss | Ratio |
|------|---------------|------|-------|
| 1 | 2 | 0.218 | 2.4× |
| 2 | 34 | 0.329 | 3.7× → **stopped** (>3×) |

**Only 1 layer can be removed** before loss triples. The model needs all 36 layers to reproduce its own greedy decoding trajectory.

## Conclusions & Key Findings

1. **Every layer is essential**: Even the "least critical" (layer 2) causes 2.4× loss increase. No layer is redundant for self-completion.

2. **Layer 0 is catastrophically critical (~100×)**: Removing it alone takes loss from 0.09 to 8.91, making the model nearly non-functional. It establishes the residual stream's initial structure from raw embeddings.

3. **Layer 6 is the model's critical hub**: 2nd most critical for knockout (22×), appears in 4/5 top synergistic pairs. The (5,6) circuit is the strongest inter-layer dependency found (+3.53 synergy).

4. **Zero redundancy on self-generated text**: Only 1 layer can be removed before loss triples. The model has no spare capacity for reproducing its own greedy output — every layer contributes.

5. **Synergy reveals circuits**: Super-additive pairs cluster around layer 6, which forms tightly coupled circuits with layers 5, 7, 1, and 12. The (6,12) long-range coupling (+1.45) suggests functional dependencies that bypass 5 intervening layers via the residual stream.

For layer replacement and linearization analysis building on these knockout results, see T-7.

## Usage

```bash
# Generate completions first (one-time):
poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Instruct-2507

# Run experiment (Parts 1-3):
poetry run python experiments/t2_layer_knockout/run.py
```

Results in `experiments/t2_layer_knockout/results/`:
- `results.json` — all data (knockouts, pairs, pruning)
- `single_knockout_overview.png` — criticality profile
- `pair_knockout_heatmap.png` — pair loss delta and synergy heatmaps
- `greedy_pruning_curve.png` — loss trajectory
