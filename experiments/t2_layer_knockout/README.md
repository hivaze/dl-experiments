# T-2: Layer Knockout / Criticality Mapping

## Motivation & Research Question

Which layers in a transformer are truly critical, and which are redundant? By skipping each layer one at a time (and pairs), we can map the "criticality profile" of the model. We also compare knockout criticality with causal tracing (activation patching) to see if the "most critical" layer is the same as the "causal bottleneck" for factual recall.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 decoder layers, bf16)
- **Hardware**: NVIDIA B200, cuda:1
- **Evaluation data**: 28 question/instruction-format prompts with greedy completions (vLLM, temp=0, max 512 tokens). System message prevents echo repetition. Loss computed **only on completion tokens**, providing a much more sensitive measure of disruption.
- **Patching prompts**: 8 prompts with known targets (factual, arithmetic, linguistic) using raw tokenization
- **Baseline loss**: 0.0877

## Methods

### Part 1: Single Layer Knockout
Remove each of the 36 layers and measure cross-entropy loss on completion tokens. The loss ratio vs baseline quantifies criticality.

### Part 2: Pair Layer Knockout
Test 94 selected layer pairs. Compute **synergy** = actual delta - sum of individual deltas.

### Part 3: Greedy N-Layer Pruning
Iteratively remove the least-critical remaining layer. Stop when loss exceeds 3x baseline.

### Part 4: Weight Norm vs Criticality Correlation

### Part 5: Causal Tracing (Activation Patching)
Corrupt subject token embeddings, then restore clean hidden states at individual layers to find where factual associations are stored.

## Results

### Single Layer Knockout — Criticality Profile

| Layer | Loss | Ratio | Notes |
|-------|------|-------|-------|
| 0 | 8.850 | **100.9x** | Catastrophically critical |
| 6 | 2.044 | **23.3x** | Second most critical |
| 1 | 0.722 | 8.2x | |
| 10 | 0.569 | 6.5x | |
| 12 | 0.561 | 6.4x | |
| ... | ... | ... | |
| 34 | 0.247 | 2.8x | |
| 3 | 0.241 | 2.7x | |
| 2 | 0.209 | **2.4x** | Least critical |

**Key finding**: With completion-token loss, **every layer is critical** — even the least critical layer (2) causes a 2.4x loss increase. The model's ability to reproduce its own greedy output is tightly dependent on all 36 layers.

Layer 0 is catastrophically critical (101x) — removing it alone makes the model nearly non-functional. Layer 6 is similarly devastating (23x).

### Pair Knockout — Synergy

Most super-additive pairs:

| Pair | Synergy | Notes |
|------|---------|-------|
| (5, 6) | +3.63 | Layers 5+6 form a critical circuit |
| (6, 7) | +1.81 | Layer 6 hub |
| (1, 6) | +1.79 | Layer 6 hub |
| (6, 12) | +1.66 | Layer 6 long-range synergy |
| (1, 2) | +1.63 | Early layers interdependent |

Layer 6 appears in 4 of the top 5 synergistic pairs — it's the central hub of the model's critical circuitry.

### Greedy Pruning

| Step | Layer Removed | Loss | Ratio |
|------|---------------|------|-------|
| 1 | 2 | 0.209 | 2.4x |
| 2 | 3 | 0.323 | 3.7x → **stopped** (>3x) |

**Only 1 layer can be removed** before loss triples. The model has essentially zero redundancy when evaluated on its own completions.

### Weight Norm vs Criticality

| Metric | Value |
|--------|-------|
| Pearson r | -0.115 (p=0.503) |
| Spearman ρ | -0.147 (p=0.393) |

**No significant correlation.** Weight norm does not predict criticality.

### Causal Tracing — Bottleneck Layers

| Type | Mean Bottleneck | Layers |
|------|----------------|--------|
| Arithmetic | 5.0 | [4, 6] |
| Linguistic | 6.7 | [19, 0, 1] |
| Factual | 9.0 | [1, 14, 12] |

Arithmetic associations are stored earliest (~layers 4-6), consistent across evaluation methods. Layer 6 continues to appear as a key knowledge storage site.

## Conclusions & Key Findings

1. **Every layer is essential**: Even the "least critical" (layer 2) causes 2.4x loss increase. No layer is redundant for self-completion.

2. **Layer 0 is ~101x critical**: Removing it is catastrophic (loss goes from 0.09 to 8.85). It performs irreplaceable input processing.

3. **Layer 6 is the model's critical hub**: It's the 2nd most critical for knockout (23x), appears in 4/5 top synergistic pairs, and is a causal bottleneck for arithmetic.

4. **Zero redundancy on self-generated text**: Only 1 layer can be removed before loss triples (vs 12 with prompt-token loss). The model has no spare capacity for reproducing its own greedy output — every layer contributes.

5. **Synergy reveals circuits**: Layers (5,6) have the highest synergy (+3.63), forming a tightly coupled computational unit. Layer 6's synergy with distant layer 12 hints at long-range functional coupling.

## Usage

```bash
# Generate completions first (one-time):
poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Instruct-2507

# Run experiment:
poetry run python experiments/t2_layer_knockout/run.py
```

Results in `experiments/t2_layer_knockout/results/`:
- `results.json` — all data (knockouts, pairs, pruning, correlations, patching)
- `single_knockout_overview.png` — 4-panel criticality profile
- `pair_knockout_heatmap.png` — pair loss delta and synergy heatmaps
- `greedy_pruning_curve.png` — loss trajectory
- `activation_patching.png` — recovery curves and knockout vs patching comparison
