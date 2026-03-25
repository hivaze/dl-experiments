# T-15: Normalization Layer Analysis & Replacement

## Motivation & Research Question

Modern LLMs universally use **RMSNorm** (Root Mean Square Normalization) rather than the original **LayerNorm**. The key difference: RMSNorm skips mean centering, normalizing only by root-mean-square magnitude. But is mean centering actually redundant in trained models? Can we swap normalization layers in a pretrained model without retraining? And how critical is QK-norm (the additional normalization Qwen3 applies to Q/K projections inside attention)?

**Key questions:**
1. How do normalization methods compare in latency on B200 GPUs — and does `torch.compile` close the gap between unfused RMSNorm and fused LayerNorm?
2. Is RMSNorm-to-LayerNorm a "free" swap (near-zero quality loss)?
3. Are residual stream means near-zero relative to their std, confirming mean centering is redundant?
4. How catastrophic is removing QK-norm, and what happens to Q/K vector magnitudes?
5. Which individual layers are most dependent on normalization?

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 layers, hidden=2560, head_dim=128, GQA 32q/8kv)
- **Hardware**: NVIDIA B200 (183GB), CUDA 12.8, bf16 precision
- **Evaluation**: 20 pre-generated greedy completions (cross-entropy loss on completion tokens)
- **Baseline**: loss = 0.0862 nats, PPL = 1.09
- **Seed**: 42
- **Runtime**: ~40 seconds total

## Mathematical Framework

### The Role of Normalization in Transformers

**Why normalize at all?** A transformer layer applies a residual update:

$$\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + g(\mathbf{x}_\ell)$$

where $g$ is the attention + MLP computation. Without normalization, the magnitude of $\mathbf{x}_\ell$ grows with depth (each layer adds to the residual stream). After 36 layers, the hidden states can grow so large that floating-point precision is lost. Normalization constrains the scale of $\mathbf{x}$ before it enters each sublayer, keeping the computation numerically stable.

**Pre-norm architecture.** Modern LLMs (including Qwen3) use **pre-norm**: normalization is applied *before* attention and MLP, not after. Concretely, each Qwen3 layer computes:

$$\mathbf{h} = \mathbf{x} + \text{Attn}(\text{RMSNorm}(\mathbf{x}))$$

$$\mathbf{x}_{\text{out}} = \mathbf{h} + \text{MLP}(\text{RMSNorm}(\mathbf{h}))$$

This means the normalization layers control what distribution the attention and MLP sublayers see as input. The residual stream itself (the $\mathbf{x}$ flowing through skip connections) is *not* normalized -- it grows freely. Only the "side channels" feeding into attention and MLP are normalized.

### RMSNorm vs LayerNorm

Both normalize a hidden state vector $\mathbf{x} \in \mathbb{R}^d$ ($d = 2560$) so that downstream layers receive inputs of predictable scale. They differ in one key operation: whether they center by the mean.

**LayerNorm** computes the per-token mean and variance, centers the input, then scales:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$. The learnable parameters $\gamma$ (scale) and $\beta$ (shift) have $d$ elements each. The output has zero mean and unit variance (before $\gamma$ and $\beta$ are applied).

**RMSNorm** skips the mean centering and the bias entirely:

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2}$$

The output has unit RMS (root-mean-square = 1) but its mean can be nonzero. RMSNorm has half the learnable parameters ($\gamma$ only, no $\beta$) and skips one reduction operation (mean computation) per call.

### The Redundancy Condition: When RMSNorm $\approx$ LayerNorm

The key mathematical relationship is between the RMS and the standard deviation:

$$\text{RMS}(\mathbf{x})^2 = \frac{1}{d}\sum_i x_i^2 = \sigma^2 + \mu^2$$

This is the bias-variance decomposition of the second moment. It follows that:

$$\text{RMS}(\mathbf{x}) = \sigma \sqrt{1 + r^2}, \qquad r = \frac{\mu}{\sigma}$$

where $r$ is the **mean-to-std ratio**. When $r \ll 1$ (the mean is small relative to the spread), $\sqrt{1 + r^2} \approx 1$ and $\text{RMS} \approx \sigma$. In this regime:

$$\text{RMSNorm}(\mathbf{x}) \approx \gamma \odot \frac{\mathbf{x}}{\sigma} \approx \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \gamma \odot \frac{\mu}{\sigma}$$

The first term matches LayerNorm (up to the missing bias $\beta$). The second term, $\gamma \mu / \sigma$, is a constant offset that enters every token's representation identically -- the model can learn to absorb this offset into the subsequent linear projections.

The question this experiment answers: **what is $r$ in practice at each layer of a trained model?**

### Learned Scale Parameters as Feature Gating

The scale vector $\gamma \in \mathbb{R}^d$ is initialized to all-ones but trained jointly with the model. If $\gamma_i \approx 0$ for some dimension $i$, then that dimension is effectively silenced before entering the sublayer -- RMSNorm normalizes it, then $\gamma_i$ scales it back to near-zero. Conversely, $\gamma_i \gg 1$ amplifies dimension $i$ relative to others.

This means the norm layer functions as a **learnable per-dimension gate**: the normalization puts all dimensions on equal footing, and $\gamma$ then selectively amplifies or suppresses dimensions before attention or MLP. The degree to which $\gamma$ has drifted from its initialization tells us how much the model relies on this gating mechanism.

### QK-Norm: Stabilizing Attention Logits

Qwen3 applies additional RMSNorm to Q and K projections inside attention. After the linear projections $W_q$ and $W_k$ map the hidden state to query and key vectors, these are reshaped to per-head format and normalized:

$$\mathbf{q}_h = \text{RMSNorm}(W_q \mathbf{x}|_h), \qquad \mathbf{k}_h = \text{RMSNorm}(W_k \mathbf{x}|_h)$$

where the subscript $h$ denotes the slice for attention head $h$, and the Q/K vectors have dimension $d_h = 128$.

**Why this matters.** Attention scores are the scaled dot product of Q and K vectors, then passed through softmax. If Q and K magnitudes are unbounded:
- Large $\|\mathbf{q}\| \cdot \|\mathbf{k}\|$ produces attention logits with extreme variance
- Softmax saturates: one position gets probability $\approx 1$, all others $\approx 0$
- Gradients vanish through the saturated softmax (gradient $\approx 0$ at the extremes)
- The model loses its ability to attend to multiple positions

QK-norm ensures that Q and K vectors have approximately unit RMS (magnitude $\approx \sqrt{d_h}$), keeping attention logits in a range where softmax produces well-distributed probabilities. This is especially important at depth, where residual stream magnitudes grow.

### Why Cross-Sequence Normalization Fails

GroupNorm (with the wrapper permutation) and BatchNorm both normalize statistics that include the sequence dimension. For a hidden state tensor $X \in \mathbb{R}^{B \times S \times d}$:

- **LayerNorm/RMSNorm** compute statistics over $d$ (the feature dimension) independently for each token at each position. Token at position 5 is normalized using only its own 2560 features.
- **GroupNorm (wrapped)** permutes to $\mathbb{R}^{B \times d \times S}$ and normalizes over $(d/G, S)$ jointly. This means each token's normalization depends on what other tokens in the sequence contain -- a token at position 5 is normalized using features from positions 1 through $S$.
- **BatchNorm** normalizes each feature across the batch and sequence dimensions jointly, computing global per-feature statistics.

The fundamental problem: transformer layers process tokens autoregressively. The hidden state at position $t$ should depend only on positions $\leq t$ (causal invariant). Cross-sequence normalization breaks this by making position $t$'s normalized value depend on positions $> t$. Even outside of autoregressive constraints, it entangles tokens that should be processed independently, creating information flow paths that the model's weights were never trained to use.

## Methods

### 1. Normalization Benchmarks
Measured forward and backward pass latency for 7 normalization variants: RMSNorm (unfused Python), RMSNorm (`torch.compile`'d), `nn.LayerNorm` (fused CUDA kernel), BatchNorm, GroupNorm, InstanceNorm, and Identity. Tested across 9 configurations (hidden sizes 2048/2560/4096 x sequence lengths 512/2K/8K), batch size 4, bf16, 50 iterations after warmup.

### 2. Activation Statistics
Collected per-layer statistics (mean, std, abs_max, kurtosis, skewness) of both the *inputs* and *outputs* of each normalization layer. This reveals what normalization actually does to the activation distribution at each layer.

### 3. QK-Norm Ablation
Replaced Q-norm and K-norm with Identity, measuring loss impact and post-norm Q/K vector magnitudes (hooked on the norm output, not the projection output as before).

### 4. Norm Replacement
Replaced all RMSNorm layers with alternative types (LayerNorm, BatchNorm, GroupNorm with 1/8/32/64 groups, Identity), transferring learned scale weights. BatchNorm includes a warmup pass to accumulate running statistics before evaluation.

### 5. Per-Layer Norm Sensitivity
Replaced normalization in one layer at a time with Identity to identify which layers are most dependent on normalization.

### 6. Residual Stream Mean Analysis
Measured the |mean|/std ratio at each layer to determine whether mean centering is redundant in practice.

## Results

### Benchmark Latency (B200, bf16)

At Qwen3-4B's native dimensions (hidden=2560, seq=2048):

| Method | Forward (ms) | Backward (ms) | Total (ms) | vs Compiled RMS |
|--------|-------------|---------------|-----------|----------------|
| **RMSNorm (unfused)** | 0.128 | 0.356 | 0.484 | 2.52x slower |
| **RMSNorm (compiled)** | 0.058 | 0.135 | **0.192** | 1.00x |
| **LayerNorm (fused)** | 0.048 | 0.180 | 0.228 | 1.19x slower |
| **BatchNorm** | 0.102 | 0.127 | 0.229 | 1.19x slower |
| **GroupNorm** | 0.191 | 0.202 | 0.392 | 2.04x slower |
| **InstanceNorm** | 0.151 | 0.167 | 0.318 | 1.66x slower |
| **Identity** | 0.005 | 0.071 | 0.075 | -- |

**Key insight**: The unfused Python RMSNorm is 2.5x slower than `torch.compile`'d RMSNorm, which in turn is **16% faster** than the fused `nn.LayerNorm` kernel. This is the correct comparison: when both have comparable kernel optimization, RMSNorm's algorithmic simplicity (skipping mean computation) provides a genuine but modest speedup. The previous version's claim that "LayerNorm is 2x faster" was an artifact of comparing a fused kernel against unfused Python.

At large scale (hidden=4096, seq=8192), compiled RMSNorm (0.54ms) is **1.67x faster** than fused LayerNorm (0.91ms), confirming the speedup grows with problem size.

### Norm Replacement Quality Impact

| Replacement | Loss (nats) | PPL | Loss Ratio | Verdict |
|------------|------------|------|-----------|---------|
| **Baseline (RMSNorm)** | 0.0862 | 1.09 | 1.00x | -- |
| **LayerNorm** | 0.1205 | 1.13 | **1.40x** | Near-lossless |
| **BatchNorm** | 11.98 | 159,760 | 138.9x | Catastrophic |
| **GroupNorm (1 group)** | 11.90 | 147,560 | 138.0x | Catastrophic |
| **GroupNorm (8 groups)** | 12.85 | 380,634 | 149.0x | Catastrophic |
| **GroupNorm (32 groups)** | 13.15 | 512,593 | 152.4x | Catastrophic |
| **GroupNorm (64 groups)** | 13.27 | 577,287 | 153.8x | Catastrophic |
| **Identity (no norm)** | NaN | NaN | NaN | Total collapse |

**RMSNorm to LayerNorm is near-free** (1.40x loss ratio, PPL 1.09 to 1.13). This is the *only* viable zero-shot swap — every other replacement causes catastrophic degradation.

**Why GroupNorm fails**: The `GroupNormWrapper` permutes input to `(B, hidden, seq)` for `nn.GroupNorm`, which normalizes over `(hidden/G, seq)` jointly. This means it normalizes *across the sequence dimension* — fundamentally breaking the transformer's per-token independence. Each token's representation gets contaminated by statistics from other tokens at a different processing stage than attention. Even GroupNorm with 1 group (which normalizes over all of hidden + seq jointly) fails catastrophically.

**Why BatchNorm fails**: BatchNorm normalizes per-feature across the batch+sequence dimension, computing running mean/variance statistics. This introduces cross-sample information leakage and produces different behavior in train vs eval mode.

### QK-Norm Ablation

| Condition | Loss (nats) | PPL | Loss Ratio |
|-----------|------------|------|-----------|
| With QK-norm | 0.0862 | 1.09 | 1.00x |
| Without QK-norm | 11.71 | 121,234 | **135.7x** |

**QK-norm is structurally essential.** The post-norm Q/K magnitude comparison reveals dramatic divergence:

| Layer | Q max (with norm) | Q max (without) | K max (with norm) | K max (without) |
|-------|------------------|----------------|------------------|----------------|
| 0 | 11.5 | 2.0 | **314.0** | 1.0 |
| 16 | 13.9 | 13.4 | 26.4 | 5.7 |
| 24 | 10.0 | 27.8 | 27.4 | 16.3 |
| 32 | 17.6 | **65.5** | 15.5 | 29.1 |

The pattern reverses across depth: in early layers, QK-norm *amplifies* magnitudes (layer 0 K: 1.0 to 314.0); in late layers, it *suppresses* them (layer 32 Q: 65.5 to 17.6). This bidirectional control keeps attention logits in a numerically stable range throughout the network. Without it, late-layer Q magnitudes grow to 65+ while early-layer K magnitudes collapse, causing attention score miscalibration.

### Activation Statistics

**Normalization as compression.** The norm input vs output std comparison reveals that RMSNorm dramatically compresses the residual stream:

| Layer Range | Input Std | Output Std | Compression Ratio |
|------------|-----------|-----------|------------------|
| 0-5 (early) | 0.02 - 0.67 | 0.03 - 0.10 | 1.5x - 6.5x |
| 6 (transition) | 0.67 | 0.14 | 4.8x |
| 7-35 (deep) | 19.4 - 21.7 | 0.23 - 4.44 | 5x - 84x |

The residual stream std jumps from 0.67 to **19.4** at layer 6 (a 29x increase in one layer), then remains at 19-22 for all subsequent layers. Meanwhile, the norm *output* std grows gradually from 0.23 to 4.44 across layers 7-35. The norm is doing increasingly heavy lifting as depth increases — compressing a 19-22 std input down to match what the attention/MLP sublayers expect.

**Extreme kurtosis.** Activation kurtosis reaches **170,000** in deep layers (compared to 3.0 for a Gaussian). This indicates the residual stream has extremely heavy tails — most values are near zero, with rare but extreme outliers. This is a known phenomenon in transformer activations and partly explains why normalization is essential: without it, these outliers would dominate the downstream computation.

### Per-Layer Norm Sensitivity

Replacing normalization in a single layer with Identity reveals a clear pattern:

| Sensitivity | Layers | Loss Ratio Range |
|-------------|--------|-----------------|
| **Least sensitive** | 4, 6, 3, 5, 0 | 14.6x - 56.5x |
| **Mid-range** | 1, 2, 8-18 | 80x - 170x |
| **Most sensitive** | 7, 22, 24, 25, 29 | 194x - 226x |

**No layer is "norm-optional"** — even the least sensitive layer (layer 4) sees a 14.6x loss increase when its norms are removed. But the sensitivity varies by 15x across layers. The most sensitive layers (22, 24, 25, 29) are in the late-middle of the network, suggesting this is where the model does its most numerically delicate computation.

Layer 7 stands out: it's the first layer *after* the layer-6 residual stream explosion (std jumps from 0.67 to 19.4), making normalization especially critical to tame the suddenly-large activations.

### Residual Stream Mean Analysis

| Layer Range | Avg |Mean| | Avg Std | |Mean|/Std Ratio |
|------------|------------|---------|----------------|
| 0 (embed) | 0.0002 | 0.019 | 0.0127 |
| 1-6 (early) | 0.002 - 0.014 | 0.17 - 0.62 | 0.013 - 0.022 |
| 7-35 (deep) | 0.046 - 0.165 | 3.3 - 12.6 | 0.012 - 0.015 |

The |mean|/std ratio is **remarkably constant at ~0.013-0.015** across all 36 layers. While the raw means grow from 0.0002 to 0.165 (an 800x increase), the std grows proportionally. This means mean centering corrects approximately the same fraction of the signal at every layer — about 1.3% of the std. This is small enough that the model is robust to the omission (confirming the 1.40x loss ratio for the LayerNorm swap), but non-zero enough that the swap isn't perfectly lossless.

### Norm Weight Distribution

The learned RMSNorm scale parameters ($\gamma$) have drifted significantly from their initialization of 1.0:

| Norm Location | Mean($\gamma$) | Std($\gamma$) | Within 10% of 1.0 | Within 50% of 1.0 |
|--------------|-------------|------------|-------------------|-------------------|
| Input LayerNorm (pre-attn) | 1.010 | **1.331** | 7.5% | 31.3% |
| Post-Attn LayerNorm (pre-MLP) | 0.796 | 0.380 | 11.5% | 80.0% |

The input layernorm has extremely high weight variance (std=1.33), meaning some dimensions are scaled by 3-4x while others are near zero. **Only 7.5% of pre-attention scale parameters remain near initialization** — the model aggressively reshapes the feature distribution entering attention. The pre-MLP norm is more conservative (80% within 50% of 1.0), suggesting attention output distributions are closer to what the MLP expects.

## Conclusions & Key Findings

1. **RMSNorm to LayerNorm is a viable zero-shot swap** (1.40x loss ratio). This is the *only* replacement that preserves model quality. The residual stream |mean|/std ratio of ~1.3% explains why: mean centering corrects a non-zero but small bias.

2. **Compiled RMSNorm is genuinely faster than fused LayerNorm** (16% at hidden=2560, 67% at hidden=4096). The previous version's contrary finding was an artifact of comparing unfused Python against a fused CUDA kernel.

3. **QK-norm is as critical as the main normalization layers** (136x loss increase when removed). It plays a bidirectional role: amplifying small K vectors in early layers and suppressing large Q vectors in late layers, keeping attention logits calibrated.

4. **All normalization that crosses the sequence dimension fails catastrophically.** GroupNorm and BatchNorm both normalize across tokens, violating the transformer's per-token processing model. This is a fundamental architectural constraint, not a weight compatibility issue.

5. **No individual layer can survive without normalization.** Even the least sensitive layer (layer 4) sees 14.6x loss increase. The most sensitive layers (22, 24, 25, 29) are in the late-middle of the network. Layer 7, immediately after the layer-6 residual stream explosion, is also highly sensitive.

6. **Norm scale parameters are heavily learned** — only 7.5% of pre-attention scales remain near initialization, suggesting the model uses normalization as a learned feature gating mechanism.

7. **Residual stream kurtosis reaches 170,000** (vs 3.0 for Gaussian), confirming extreme outlier structure in deep transformer activations. This explains why normalization is structurally essential at inference time, not just a training aid.

## Usage

```bash
poetry run python experiments/t15_normalization/run.py
```

**Prerequisites**: Pre-generated completions in `data/text_completions/qwen3-4b-instruct-2507/completions.json` (from `generate_completions.py`).

**Runtime**: ~40 seconds total.

**Outputs**: `experiments/t15_normalization/results/`
- `results.json` -- All quantitative data
- `benchmark_latency.png` -- Norm type latency comparison (including compiled RMSNorm)
- `replacement_losses.png` -- Loss/PPL after each replacement (log scale)
- `activation_stats.png` -- Per-layer activation statistics (mean, std, kurtosis)
- `qk_norm_ablation.png` -- Post-norm Q/K magnitudes with and without QK-norm
- `norm_weight_distribution.png` -- Learned scale parameter distribution
- `residual_means.png` -- |Mean|/Std ratio across layers
- `replacement_activation_profiles.png` -- Residual stream std under each replacement
- `per_layer_sensitivity.png` -- Per-layer norm sensitivity (Identity replacement)
- `norm_io_comparison.png` -- Norm input vs output std (compression ratio)
