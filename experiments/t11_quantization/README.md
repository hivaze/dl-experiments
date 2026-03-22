# T-11: Quantization Methods Comparative Analysis

## Motivation & Research Question

Quantization compresses model weights to lower bit-widths, reducing memory and (sometimes) improving throughput. But which layers and weight matrices tolerate compression, and which break? Standard quantization benchmarks report whole-model numbers. We have something better: **per-layer linearity data (T-7) and per-matrix spectral rank data (T-9)**. Can these structural properties predict quantization sensitivity?

**Primary questions:**
1. Which layers/matrices are most sensitive to quantization noise?
2. Does T-7 linearity gap predict quantization robustness?
3. Does T-9 spectral rank predict quantization robustness?
4. Can spectral/linearity-informed mixed-precision beat uniform quantization?

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 homogeneous decoder layers, 2560 hidden dim)
- **Hardware**: NVIDIA B200 (183GB VRAM), CUDA 12.8
- **Evaluation**: WikiText-2 test set, 4096 tokens, single forward-pass perplexity (loss in float32 for precision)
- **Quantization simulation**: Symmetric per-group RTN (round-to-nearest), group_size=128
- **Real quantization**: bitsandbytes NF4/INT8, torchao INT8 weight-only
- **Prior data**: T-7 summary.json (per-layer linearity gaps), T-9 summary.json (per-layer/matrix spectral rank)

## Methods

### Phase 1: Per-Layer RTN Sensitivity
For each of 36 layers independently, quantize all 7 weight matrices to {8, 6, 4, 3, 2}-bit via RTN simulation, measure perplexity delta. This isolates each layer's individual sensitivity.

### Phase 1b: Per-Matrix Sensitivity
For each of 36 layers and 7 matrix types (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj), quantize that single matrix to 3-bit and measure perplexity delta. This identifies which matrix types are most critical.

### Phase 2: Full-Model Methods Comparison
Apply quantization to the entire model and measure perplexity, memory, and generation speed:
- **RTN** (simulated): 8-bit, 4-bit, 3-bit
- **bitsandbytes NF4**: Real 4-bit NormalFloat with double quantization
- **bitsandbytes INT8**: Real 8-bit mixed-precision decomposition
- **torchao INT8**: Weight-only INT8 quantization
- **GPTQ W4A16** (via llmcompressor): Calibration-based 4-bit weight quantization (256 samples from WikiText-2)
- **GPTQ W8A16** (via llmcompressor): Calibration-based 8-bit weight quantization

### Phase 3: Spectral-Informed Mixed Precision
Design per-layer bit-width recipes using T-7/T-9 data, all targeting ~4-bit average:
- **Uniform 4-bit** (baseline)
- **T-9 spectral**: Higher effective rank $\to$ more bits
- **T-7 linearity**: Higher linearity gap $\to$ more bits
- **Combined T-7+T-9**: Weighted average of both signals
- **Sensitivity oracle**: Direct from measured 2-bit sensitivity (upper bound)
- **First/last protected**: Layers 0 and 35 at 8-bit, rest adjusted
- **Per-matrix mixed**: Q/K at 3-bit, V/O/MLP at 5-bit (~4.4-bit avg)

### Phase 4: Correlation Analysis
Spearman rank correlation between quantization sensitivity and T-7/T-9 metrics, plus visualization.

## Results

### 1. Per-Layer Sensitivity: The Early-Layer Cliff

| Layer | 8-bit | 6-bit | 4-bit | 3-bit | 2-bit |
|-------|-------|-------|-------|-------|-------|
| **0** | +0.00 | -0.00 | +0.01 | +0.03 | **+20.5** |
| **1** | +0.01 | +0.02 | -0.01 | +0.13 | **+1126.6** |
| **2** | +0.00 | +0.01 | +0.02 | **+15.20** | **+3827.6** |
| **3** | -0.00 | -0.00 | -0.02 | +0.21 | **+277.1** |
| **4** | +0.00 | +0.00 | +0.03 | +0.10 | +8.1 |
| 5-20 | ~0 | ~0 | ~0 | <0.15 | **0.2-1.4** |
| 22 | -0.00 | +0.01 | +0.09 | +0.31 | +1.3 |
| **35** | +0.00 | +0.01 | +0.10 | **+2.29** | **+5.95** |

**Key finding**: Quantization sensitivity follows a steep exponential decay from the embedding layers. Layers 0-3 are catastrophically sensitive at 2-bit (PPL increases of 20-3828x), while mid-layers (8-20) are remarkably robust (PPL delta < 0.7 even at 2-bit). Layer 35 is a secondary sensitivity hotspot.

The model is **essentially lossless at 4-bit per-layer RTN** — no single layer causes more than 0.1 PPL degradation. The cliff appears between 3-bit and 2-bit.

### 2. Per-Matrix Sensitivity: gate_proj Dominates

Average per-matrix sensitivity at 3-bit RTN (PPL delta):

| Matrix | Mean PPL Δ | Std | Most Sensitive Layer |
|--------|-----------|-----|---------------------|
| **gate_proj** | **0.247** | 1.29 | Layer 2: +7.87 |
| down_proj | 0.052 | 0.25 | Layer 35: +1.54 |
| up_proj | 0.035 | 0.05 | Layer 34: +0.13 |
| v_proj | 0.007 | 0.04 | Layer 22: +0.08 |
| k_proj | 0.007 | 0.03 | Layer 33: +0.07 |
| o_proj | 0.003 | 0.02 | Layer 20: +0.06 |
| q_proj | 0.001 | 0.02 | Layer 22: +0.04 |

**gate_proj is 50x more sensitive than attention projections.** This makes sense: the SwiGLU gating mechanism ($\text{SiLU}(W_g x) \odot W_u x$) means gate_proj's errors get amplified through a nonlinear activation and element-wise multiplication. Quantization noise in the gate directly corrupts which features are selected.

Layer 2's gate_proj alone causes +7.87 PPL at 3-bit — accounting for most of the layer's total sensitivity. Layer 35's down_proj (+1.54) is the output bottleneck.

### 3. Full-Model Quantization Methods

| Method | PPL | PPL Δ | Memory | Gen Speed | Quant Time | Notes |
|--------|-----|-------|--------|-----------|-----------|-------|
| BF16 (baseline) | 9.252 | — | 8.05 GB | 86.9 tok/s | — | |
| **GPTQ W8A16** | **9.271** | **+0.02** | (deploy) | (deploy) | 179s | Best quality overall |
| RTN 8-bit | 9.273 | +0.02 | (sim) | 86.7 | — | Simulated, lossless |
| **bnb INT8** | **9.298** | **+0.05** | 12.47 GB | 19.5 | — | Mixed-precision decomp. |
| **torchao INT8** | **9.319** | **+0.07** | 12.87 GB | 45.2 | — | Best 8-bit speed |
| **bnb NF4** | **9.345** | **+0.09** | 10.72 GB | 53.7 | — | Best practical 4-bit |
| **GPTQ W4A16** | **9.482** | **+0.23** | (deploy) | (deploy) | 327s | Calibration-based 4-bit |
| RTN 4-bit | 9.741 | +0.49 | (sim) | 86.7 | — | Simulated |
| RTN 3-bit | 214.7 | +205.4 | (sim) | 86.7 | — | Catastrophic |

**Calibration-based methods dramatically outperform naive RTN.** At 4-bit, bnb NF4 achieves only +0.09 PPL delta vs GPTQ's +0.23 and RTN's +0.49. NF4's NormalFloat data type is specifically designed for the bell-curved weight distribution of trained neural networks, which explains its advantage over uniform integer quantization.

GPTQ W4A16 (+0.23) is better than RTN (+0.49) thanks to Hessian-guided optimal weight rounding, but falls short of NF4 at the same nominal bit-width. GPTQ W8A16 is essentially lossless (Δ = +0.02), matching RTN 8-bit.

Note: RTN speed appears identical to BF16 because it's simulated (weights stored as BF16 after quantize/dequantize). GPTQ memory/speed marked "(deploy)" because llmcompressor stores models in compressed_tensors format — real deployment memory/speed depends on the serving engine (vLLM, ONNX Runtime).

The bnb INT8 memory is *higher* than BF16 (12.47 vs 8.05 GB) due to mixed-precision decomposition overhead. torchao INT8 shows the same pattern. For this 4B model, the metadata overhead outweighs the int8 savings; the benefit materializes for larger models.

### 4. Mixed-Precision Recipes: Simple Heuristics Win

| Recipe | PPL Δ | Avg Bits | Description |
|--------|-------|----------|-------------|
| **first_last_protected** | **+0.36** | 4.0 | Layers 0,35 at 8b, rest 3-4b |
| sensitivity_oracle | +0.41 | 4.03 | Informed by measured 2-bit sensitivity |
| uniform_4bit | +0.49 | 4.0 | All layers at 4-bit |
| qk3_vmul5 | +0.56 | 4.43 | Q/K=3b, V/MLP=5b |
| t7_linearity | +46.9 | 3.97 | Gap-proportional bits |
| t7_t9_combined | +211.9 | 4.0 | Combined spectral+linearity |
| t9_spectral | +13941 | 4.0 | Rank-proportional bits |

**The simple "protect first and last layers" heuristic beats every spectral/linearity-informed recipe** and even beats the sensitivity oracle. The spectral-informed recipe fails catastrophically because it assigns 2-bit to low-rank early layers — exactly the layers that are most sensitive.

**Why spectral recipes fail**: T-9 showed early layers have *low* effective rank (especially Q/K projections at ~0.25 ratio), which the recipe interprets as "highly compressible." But low rank in early layers means errors in the compact representation propagate through all 36 downstream layers. Low rank $\neq$ quantization-safe.

The per-matrix recipe (Q/K=3b, V/MLP=5b) performs *worse* than uniform 4-bit despite using 0.43 more bits on average. Splitting bit-budgets by matrix type adds complexity without benefit.

### 5. Correlation with T-7 and T-9

#### T-7 Linearity Gap Strongly Predicts Sensitivity

| Metric | Spearman $\rho$ | p-value | Significance |
|--------|----------------|---------|--------------|
| T-7 MLP gap | +0.709 | < 0.0001 | *** |
| T-7 perturb gap (total) | +0.679 | < 0.0001 | *** |
| T-7 attention gap | +0.444 | 0.007 | ** |
| T-7 nonlinearity order | +0.183 | 0.285 | NS |

**Layers with higher linearity gap (more nonlinear) are more sensitive to quantization.** The MLP gap is the strongest predictor ($\rho$ = 0.71), consistent with gate_proj being the most sensitive matrix. The correlation is driven by early layers (0-4) which have both the highest linearity gaps (T-7: U-shaped profile peaking at layers 0-1) and the highest quantization sensitivity.

#### T-9 Spectral Rank Does NOT Predict Layer-Level Sensitivity

| Metric | Spearman $\rho$ | p-value | Significance |
|--------|----------------|---------|--------------|
| T-9 avg effective rank | -0.228 | 0.182 | NS |
| T-9 q_proj rank | -0.111 | 0.518 | NS |
| T-9 k_proj rank | -0.202 | 0.237 | NS |
| T-9 gate_proj rank | -0.168 | 0.328 | NS |
| **T-9 up_proj rank** | **-0.509** | **0.002** | ** |

The only significant predictor from T-9 is up_proj rank ($\rho$ = -0.51, p = 0.002): layers where up_proj uses less capacity (lower rank) are more sensitive. This makes sense — lower-rank up_proj means fewer active dimensions, so quantization noise has a larger relative effect.

#### Per-Matrix Sensitivity vs Rank

| Matrix | $\rho$ (rank vs sensitivity) | p-value | Interpretation |
|--------|----------------------------|---------|----------------|
| **v_proj** | **+0.381** | **0.022** | Higher rank = MORE sensitive |
| gate_proj | -0.309 | 0.067 | Trend: lower rank = more sensitive |
| q_proj | -0.193 | 0.259 | NS |
| up_proj | -0.084 | 0.627 | NS |

The v_proj result is counterintuitive: higher-rank V matrices are *more* sensitive to 3-bit quantization. This may reflect that high-rank V matrices are using more of their capacity for meaningful computation, leaving less redundancy to absorb quantization noise.

## Conclusions & Key Findings

### 1. The Early-Layer Vulnerability
Layers 0-3 are catastrophically sensitive to aggressive quantization. A single layer quantized to 2-bit can increase perplexity by 3,800x. This is a **propagation effect**: errors introduced at layer 0 compound through all 36 downstream layers, while errors at layer 20 only propagate through 16 layers.

### 2. gate_proj is the Achilles' Heel
The SwiGLU gate projection is 50x more sensitive than attention projections. Quantization noise in the gate corrupts the feature selection mechanism nonlinearly. Any quantization scheme should prioritize gate_proj precision.

### 3. Linearity Predicts, Spectral Rank Does Not (at Layer Level)
T-7's linearity gap is a strong predictor of quantization sensitivity ($\rho$ = 0.71***). T-9's spectral rank is not ($\rho$ = -0.23, NS). This resolves a theoretical question: **quantization sensitivity is driven by nonlinear amplification (activations), not by weight matrix structure.**

The spectral rank becomes relevant at the *matrix type* level (gate_proj, up_proj), but not for comparing layers against each other.

### 4. Simple Heuristics Beat Sophisticated Recipes
"Protect the first and last layers" ($\Delta$PPL = +0.36) outperforms every spectral-informed, linearity-informed, and even oracle-informed mixed-precision recipe at the same average bit budget. This suggests that for Qwen3-4B at ~4-bit average:
- Just keep layers 0 and 35 at 8-bit
- Quantize everything else uniformly to 3-4-bit
- Don't bother with per-matrix mixed precision

### 5. NF4 is the Practical Winner at 4-bit
bitsandbytes NF4 achieves PPL within 0.09 of BF16 at 4-bit, dramatically better than GPTQ (+0.23) and naive RTN (+0.49). NF4's NormalFloat data type is specifically designed for the bell-curved weight distribution of trained neural networks, giving it an advantage over uniform integer quantization. For production deployment, NF4 with double quantization is the clear choice at 4-bit; at 8-bit, all calibrated methods are essentially lossless (Δ < 0.07).

### Connection to T-7 / T-9 / Blog Post

The depth profile from T-7 showed a U-shaped linearity gap: high at layers 0-1 (embedding projection), minimum at 8-18 (plateau), rising again at 33-35 (output preparation). Quantization sensitivity follows the *same* early-layer pattern but not the late-layer uptick — the correlation is driven by the embedding layers.

T-7 discovered the "linearization paradox": layer 16 is locally very linear (low gap) but globally irreplaceable (37% CE recovery). Quantization tells a different story: layer 16 is *not* particularly sensitive (2-bit Δ = +0.39). This makes sense — quantization is a local perturbation (unlike linearization which replaces the entire computation), so local linearity correctly predicts local robustness.

T-9 showed Q/K projections are low-rank (0.25-0.38) while V/MLP are high-rank (0.50-0.68). Our per-matrix analysis confirms Q/K are the *least* sensitive to quantization, consistent with their redundancy. But T-9's layer-level rank does not predict layer-level sensitivity, because the dominant factor is *position in the network* (early vs late), not *weight structure*.

## Usage

```bash
poetry run python experiments/t11_quantization/run.py
```

Prerequisites: `bitsandbytes`, `torchao`, `llmcompressor` (`poetry run pip install bitsandbytes torchao llmcompressor`). WikiText-2 is downloaded automatically via the `datasets` library.

Runtime: ~12 minutes on a single B200 GPU (~3 min for phases 1/1b/3/4, ~10 min for GPTQ calibration in phase 2). Without GPTQ methods: ~3 minutes.
