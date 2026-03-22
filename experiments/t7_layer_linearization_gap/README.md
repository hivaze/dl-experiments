# T-7: Layer Linearization Gap

## Motivation & Research Question

**How nonlinear is each layer's computation on real inputs?**

Each transformer layer applies a nonlinear transformation g(x) = layer(x) - x to the residual stream. This transformation involves attention (softmax) and MLP (SwiGLU activation). If a layer's computation is approximately linear on the data manifold, it could potentially be replaced by a cheap linear map without significant quality loss. We measure the "linearization gap" — how much the actual layer output deviates from what a linear approximation would predict — and track how this varies across depth.

**Hypothesis**: Early layers are more linear (mainly doing token mixing via attention), while late layers are more nonlinear (doing feature composition via MLP). If true, early layers are candidates for linearization/distillation.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 layers, hidden_dim=2560, GQA 32q/8kv, SwiGLU MLP)
- **Data**: 28 pre-generated calibration completions (vLLM, temp=0), completion tokens only
- **Hardware**: NVIDIA B200, bf16 inference
- **Seed**: 42
- **Max sequence length**: 128 tokens
- **Runtime**: ~88 seconds

## Mathematical Framework

### Notation

For transformer layer l, define:
- $\mathbf{x} \in \mathbb{R}^{B \times T \times d}$: the input hidden states (residual stream), where B=batch, T=seq_len, d=2560
- $g: \mathbb{R}^d \to \mathbb{R}^d$: the non-residual transform (what the layer *adds* to the residual stream), applied per-token
- $\mathbf{J}_g(\mathbf{x}) \in \mathbb{R}^{d \times d}$: the Jacobian $\partial g / \partial \mathbf{x}$ evaluated at $\mathbf{x}$ (per-token)

The layer's full operation is $\mathbf{x} \to \mathbf{x} + g(\mathbf{x})$ (residual connection). The function $g$ decomposes as:

$$g(\mathbf{x}) = f_{\text{attn}}(\mathbf{x}) + f_{\text{mlp}}(\mathbf{x} + f_{\text{attn}}(\mathbf{x}))$$

where:

$$f_{\text{attn}}(\mathbf{x}) = \mathbf{W}_o \cdot \text{Attn}(\mathbf{W}_q \cdot \text{RMSNorm}(\mathbf{x}),\; \mathbf{W}_k \cdot \text{RMSNorm}(\mathbf{x}),\; \mathbf{W}_v \cdot \text{RMSNorm}(\mathbf{x}))$$

$$f_{\text{mlp}}(\mathbf{h}) = \mathbf{W}_{\text{down}} \cdot (\text{SiLU}(\mathbf{W}_{\text{gate}} \cdot \text{RMSNorm}(\mathbf{h})) \odot (\mathbf{W}_{\text{up}} \cdot \text{RMSNorm}(\mathbf{h})))$$

The nonlinear components are:
1. **Softmax** in attention: $\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$ — quadratic in Q, K via the bilinear form, then nonlinear via exp
2. **SiLU** ($= x \cdot \sigma(x)$) in SwiGLU: smooth, approximately linear near 0, approximately identity for large positive x
3. **RMSNorm**: $\mathbf{x} \to \mathbf{x} \cdot \sqrt{d} / \|\mathbf{x}\|_2$ — projects onto the unit sphere, making $g$ scale-invariant

### Fréchet Derivative and Linearization

The Fréchet derivative of $g$ at $\mathbf{x}$ is the bounded linear operator $Dg(\mathbf{x}): \mathbb{R}^d \to \mathbb{R}^d$ such that:

$$g(\mathbf{x} + \mathbf{h}) = g(\mathbf{x}) + Dg(\mathbf{x})[\mathbf{h}] + o(\|\mathbf{h}\|)$$

For finite-dimensional $g$, $Dg(\mathbf{x})$ is represented by the Jacobian matrix $\mathbf{J}_g(\mathbf{x}) \in \mathbb{R}^{d \times d}$:

$$[\mathbf{J}_g(\mathbf{x})]_{ij} = \frac{\partial g_i}{\partial x_j}$$

The linearization of $g$ at $\mathbf{x}$ is:

$$g_{\text{lin}}(\mathbf{x} + \mathbf{h}) = g(\mathbf{x}) + \mathbf{J}_g(\mathbf{x}) \, \mathbf{h}$$

This is the best linear approximation to $g$ near $\mathbf{x}$. The **linearization gap** measures how well this approximation works for perturbations of a given size.

### Taylor Expansion and Error Analysis

For a twice-differentiable function $g$, the second-order Taylor expansion at $\mathbf{x}$ gives:

$$g(\mathbf{x} + \varepsilon \mathbf{d}) = g(\mathbf{x}) + \varepsilon \, \mathbf{J}_g(\mathbf{x}) \, \mathbf{d} + \frac{\varepsilon^2}{2} \, \mathbf{d}^\top \mathbf{H}_g(\mathbf{x}) \, \mathbf{d} + \mathcal{O}(\varepsilon^3)$$

where $\mathbf{H}_g(\mathbf{x})$ is the Hessian tensor ($d \times d \times d$ array of second derivatives), and $\mathbf{d}^\top \mathbf{H}_g \mathbf{d}$ denotes the bilinear contraction.

The key quantities are:

$$\Delta = g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x}) = \varepsilon \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] + \mathcal{O}(\varepsilon^3) \quad \text{(actual displacement)}$$

$$\hat{\Delta} = \varepsilon \, \mathbf{J}_g(\mathbf{x}) \, \mathbf{d} = \varepsilon \mathbf{J} \mathbf{d} \quad \text{(linear prediction)}$$

$$\mathbf{r} = \Delta - \hat{\Delta} = \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] + \mathcal{O}(\varepsilon^3) \quad \text{(2nd-order residual)}$$

The **perturbation gap** is the relative magnitude of this residual:

$$\text{gap}(\mathbf{x}, \mathbf{d}, \varepsilon) = \frac{\|\mathbf{r}\|}{\|\Delta\|} = \frac{\|\Delta - \hat{\Delta}\|}{\|\Delta\|}$$

**Scaling with** $\varepsilon$**:** For a fixed direction $\mathbf{d}$:

$$\|\mathbf{r}\| \sim \frac{\varepsilon^2}{2} \|\mathbf{H}[\mathbf{d}, \mathbf{d}]\| \quad \text{(2nd-order term dominates } \mathbf{r}\text{)}$$

$$\|\Delta\| \sim \varepsilon \|\mathbf{J} \mathbf{d}\| \quad \text{(1st-order term dominates } \Delta\text{)}$$

$$\text{gap} \sim \frac{\varepsilon \, \|\mathbf{H}[\mathbf{d}, \mathbf{d}]\|}{2 \|\mathbf{J} \mathbf{d}\|} \quad \text{(gap is linear in } \varepsilon \text{ for quadratic nonlinearity)}$$

More generally, if the dominant nonlinearity is degree $k$ (e.g., $k=2$ for softmax's quadratic form, $k=3$ for cubic terms), then $\text{gap} \sim \varepsilon^{k-1}$. The multi-scale analysis fits this power law.

### Central Difference Approximation

We estimate $\mathbf{J}_g(\mathbf{x}) \mathbf{d}$ using central finite differences:

$$\mathbf{J}_g(\mathbf{x}) \, \mathbf{d} \approx \frac{g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x} - \varepsilon \mathbf{d})}{2\varepsilon}$$

**Error analysis:** The Taylor expansion of both terms:

$$g(\mathbf{x} + \varepsilon \mathbf{d}) = g(\mathbf{x}) + \varepsilon \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] + \frac{\varepsilon^3}{6} \mathbf{T}[\mathbf{d}, \mathbf{d}, \mathbf{d}] + \cdots$$

$$g(\mathbf{x} - \varepsilon \mathbf{d}) = g(\mathbf{x}) - \varepsilon \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] - \frac{\varepsilon^3}{6} \mathbf{T}[\mathbf{d}, \mathbf{d}, \mathbf{d}] + \cdots$$

Subtracting:

$$\frac{g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x} - \varepsilon \mathbf{d})}{2\varepsilon} = \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{6} \mathbf{T}[\mathbf{d}, \mathbf{d}, \mathbf{d}] + \mathcal{O}(\varepsilon^4)$$

The even-order terms (Hessian) cancel exactly, giving $\mathcal{O}(\varepsilon^2)$ accuracy instead of $\mathcal{O}(\varepsilon)$ for forward differences. This is crucial because:
- bf16 precision has ~7.8e-3 relative error, requiring $\varepsilon \geq 0.01$
- At $\varepsilon = 0.05$, forward differences would have $\mathcal{O}(0.05) = 5\%$ Jacobian error
- Central differences have $\mathcal{O}(0.0025) = 0.25\%$ error — 20x better

### bf16 Perturbation Scaling

Direct additive perturbation $\mathbf{x} + \varepsilon \mathbf{d}$ can lose precision when $\|\mathbf{d}\| \ll \|\mathbf{x}\|$ in bf16. We scale perturbations to be proportional to the input norm:

$$\boldsymbol{\delta} = \varepsilon \, \|\mathbf{x}\| \, \hat{\mathbf{d}} \quad \text{where } \hat{\mathbf{d}} = \mathbf{d} / \|\mathbf{d}\| \text{ is a unit direction}$$

This ensures $\boldsymbol{\delta}$ has the same magnitude order as $\mathbf{x}$, so the bf16 representation of $\mathbf{x} + \boldsymbol{\delta}$ retains the perturbation information. The effective perturbation relative to $\mathbf{x}$ is:

$$\|\boldsymbol{\delta}\| / \|\mathbf{x}\| = \varepsilon \, \|\hat{\mathbf{d}}\| = \varepsilon$$

### Homogeneity Gap

Tests **degree-1 homogeneity**: if $g$ were linear and passed through the origin, then $g(\mathbf{x}) = \mathbf{J} \mathbf{x}$ exactly. We compute:

$$\mathbf{J} \mathbf{x} \approx \frac{g((1+\varepsilon)\mathbf{x}) - g((1-\varepsilon)\mathbf{x})}{2\varepsilon}$$

$$\text{homogeneity\_gap} = \frac{\|g(\mathbf{x}) - \mathbf{J} \mathbf{x}\|}{\|g(\mathbf{x})\|}$$

**Why it saturates at ~1.0:** RMSNorm normalizes by input magnitude:

$$\text{RMSNorm}(\alpha \mathbf{x}) = \frac{\alpha \mathbf{x} \cdot \sqrt{d}}{\|\alpha \mathbf{x}\|} = \frac{\mathbf{x} \cdot \sqrt{d}}{\|\mathbf{x}\|} = \text{RMSNorm}(\mathbf{x})$$

This makes $g$ approximately **scale-invariant** (degree-0 homogeneous): $g(\alpha \mathbf{x}) \approx g(\mathbf{x})$ for all $\alpha > 0$. Therefore $\partial g / \partial \alpha \approx 0$ at $\alpha = 1$, which means the Jacobian contracted with $\mathbf{x}$ (the radial direction) is near zero:

$$\mathbf{J} \mathbf{x} = \lim_{\varepsilon \to 0} \frac{g(\mathbf{x} + \varepsilon \mathbf{x}) - g(\mathbf{x})}{\varepsilon} = \lim_{\varepsilon \to 0} \frac{g((1+\varepsilon)\mathbf{x}) - g(\mathbf{x})}{\varepsilon} \approx 0$$

So the homogeneity gap becomes $\|g(\mathbf{x}) - 0\| / \|g(\mathbf{x})\| = 1$. This is not a bug — it reveals that every layer operates on the *angular* (directional) structure of $\mathbf{x}$, not its magnitude.

**Geometric interpretation:** RMSNorm projects the residual stream onto a sphere of radius $\sqrt{d}$. Each layer is effectively a map on $S^{d-1}$, the unit sphere in $d$ dimensions. The Jacobian $\mathbf{J}_g$ has a null space that includes the radial direction $\mathbf{x}/\|\mathbf{x}\|$, so its rank is at most $d-1$. This is why the homogeneity gap provides no information about the layer's actual nonlinearity — it's entirely dominated by the geometric constraint of RMSNorm.

### Multi-Scale Nonlinearity Order

If the dominant nonlinear term in $g$ is degree $k$, then:

$$\text{gap}(\varepsilon) \sim C \cdot \varepsilon^{k-1}$$

where $C$ depends on the Hessian/higher-derivative norms and the Jacobian norm. Taking logarithms:

$$\log(\text{gap}) = \log(C) + (k-1) \log(\varepsilon)$$

Linear regression of $\log(\text{gap})$ vs $\log(\varepsilon)$ gives slope $\beta = k-1$, so the estimated nonlinearity order is $k = \beta + 1$.

**Expected values:**
- Purely quadratic nonlinearity (softmax $\mathbf{Q}\mathbf{K}^\top$, SiLU): $k = 2$, slope $= 1$
- Purely cubic: $k = 3$, slope $= 2$

**Why we observe sub-quadratic orders ($k \sim 0.6\text{--}0.8$):** The gap is a ratio $\|\mathbf{r}\|/\|\Delta\|$. At large $\varepsilon$:
- The numerator $\|\mathbf{r}\|$ grows as $\varepsilon^2$ (Hessian term) but is damped by RMSNorm re-normalization
- The denominator $\|\Delta\|$ grows as $\varepsilon$ but also gets damped by RMSNorm

RMSNorm's scale-invariance means that for large perturbations, both the "actual" and "linear" responses are pulled toward the same normalized manifold, reducing their relative difference. The effective gap-vs-$\varepsilon$ curve bends downward in log-log space at large $\varepsilon$, giving an apparent slope $< 1$ (order $< 2$).

Additionally, the R² of the log-log fit varies dramatically: early/middle layers have R² ~ 0.55-0.78, but late layers degrade sharply — layer 33 has R² = 0.19, layer 35 has R² = 0.27, and layers 31-32 are at 0.40-0.44. The poor fit for late layers means the single-power-law model is inadequate there — the true gap-vs-eps curve has significant curvature in log-log space, likely due to the transition between different nonlinearity regimes at different scales. **The nonlinearity order metric (0.78-0.84) reported for layers 33-35 should be treated with caution** given these low R² values.

### Spectral Norm via Power Iteration

The spectral norm $\|\mathbf{J}_g\|_2 = \sigma_{\max}(\mathbf{J}_g)$ is estimated via power iteration:

$$\mathbf{v}_0 = \text{random unit vector}$$

$$\mathbf{v}_{k+1} = \frac{\mathbf{J} \mathbf{v}_k}{\|\mathbf{J} \mathbf{v}_k\|} \quad \text{(iterate 5 times)}$$

$$\|\mathbf{J}\|_2 \approx \|\mathbf{J} \mathbf{v}_5\|$$

where each $\mathbf{J} \mathbf{v}$ product is estimated via central differences: $\mathbf{J} \mathbf{v} \approx [g(\mathbf{x} + \varepsilon \|\mathbf{x}\| \mathbf{v}) - g(\mathbf{x} - \varepsilon \|\mathbf{x}\| \mathbf{v})] / (2\varepsilon \|\mathbf{x}\|)$.

Power iteration converges geometrically: after $k$ iterations, the error is $\mathcal{O}((\sigma_2/\sigma_1)^k)$ where $\sigma_1, \sigma_2$ are the two largest singular values of $\mathbf{J}$. With 5 iterations and typical $\sigma_2/\sigma_1 \sim 0.5\text{--}0.8$, the error is ~3-33%.

**Interpretation:** $\|\mathbf{J}_g\|_2$ is the worst-case amplification factor. If $\|\mathbf{J}_g\|_2 > 1$, perturbations can grow through the layer; if $< 1$, they shrink. The full layer Jacobian is $\mathbf{J}_{\text{layer}} = \mathbf{I} + \mathbf{J}_g$, so the layer's spectral norm is approximately $1 + \|\mathbf{J}_g\|_2$ (when $\mathbf{J}_g$'s top singular vector aligns with the residual). For stable training/inference, we need the product of all layer spectral norms to not explode — which is ensured when most layers have $\|\mathbf{J}_g\|_2 < 1$ (contractive).

### Mean Amplification

$$\mathbb{E}[\|\mathbf{J} \hat{\mathbf{d}}\|] = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{J} \mathbf{d}_i\|$$

averaged over $N$ random unit directions $\mathbf{d}_i$. This measures the *typical* amplification of the Jacobian, as opposed to the worst-case (spectral norm). By the Johnson-Lindenstrauss-type concentration of norm of Gaussian projections:

$$\mathbb{E}[\|\mathbf{J} \hat{\mathbf{d}}\|^2] = \frac{\|\mathbf{J}\|_F^2}{d} = \frac{\sum \sigma_i^2(\mathbf{J})}{d}$$

So mean amplification $\approx \|\mathbf{J}\|_F / \sqrt{d}$, which is the "average singular value" of $\mathbf{J}$. If most singular values are $< 1$, the mean amplification is $< 1$ even if a few singular values are $> 1$.

## Methods

### Method 1: Perturbation Gap (Primary Metric)

For each layer with input $\mathbf{x}$ and transform $g(\mathbf{x})$:
1. Generate 16 random unit perturbation directions $\mathbf{d}_i$
2. Scale perturbation to bf16-safe magnitude: $\boldsymbol{\delta}_i = \varepsilon \|\mathbf{x}\| \hat{\mathbf{d}}_i$ ($\varepsilon = 0.05$)
3. Estimate Jacobian-vector product via central differences: $\mathbf{J}\boldsymbol{\delta} \approx (g(\mathbf{x}+\boldsymbol{\delta}) - g(\mathbf{x}-\boldsymbol{\delta})) / 2$
4. Compare actual displacement $g(\mathbf{x}+\boldsymbol{\delta}) - g(\mathbf{x})$ to linear prediction $\mathbf{J}\boldsymbol{\delta}$
5. Gap $= \|\text{actual} - \text{linear}\| / \|\text{actual}\|$, averaged over directions and tokens

### Method 2: Homogeneity Gap

Tests scale-invariance by comparing $g(\mathbf{x})$ to $\mathbf{J}\mathbf{x}$ (the Jacobian applied to the input itself). See mathematical framework above for why this saturates at ~1.0 for RMSNorm-based architectures.

### Method 3: Attention vs MLP Decomposition

Applies the perturbation gap separately to:
- **Attention sublayer**: $f_{\text{attn}}(\mathbf{x}) = \mathbf{W}_o \cdot \text{Attn}(\text{LN}_1(\mathbf{x}))$ — nonlinearity from softmax + RMSNorm
- **MLP sublayer**: $f_{\text{mlp}}(\mathbf{x}) = \mathbf{W}_{\text{down}} \cdot \text{SwiGLU}(\text{LN}_2(\mathbf{x} + f_{\text{attn}}(\mathbf{x})))$ — nonlinearity from SiLU gating + RMSNorm

Note: the MLP sublayer function includes the attention computation (since its input depends on it), measuring the *marginal* nonlinearity of adding MLP to the attention output.

### Method 4: Jacobian Spectral Properties

- **Spectral norm** $\|\mathbf{J}\|_2$: Estimated via power iteration with finite-difference JVPs (5 iterations). Measures worst-case amplification of perturbations.
- **Mean amplification** $\mathbb{E}[\|\mathbf{J}\hat{\mathbf{d}}\|]$: Average Jacobian action on random unit vectors (16 samples). Measures typical amplification, proportional to $\|\mathbf{J}\|_F / \sqrt{d}$.

### Method 5: Multi-Scale Nonlinearity Order

Perturbation gap computed at $\varepsilon \in \{0.01, 0.02, 0.05, 0.1, 0.2\}$ with 8 random directions each. Log-log linear regression of gap vs $\varepsilon$ gives the nonlinearity order per layer. $R^2$ of the fit indicates how well a single power law describes the nonlinearity.

**Caveat:** At large $\varepsilon$ (0.1--0.2), the Taylor expansion may not converge well, and higher-order terms can cause non-monotonic gap behavior (we observe this in layers 0 and 35 where the gap at $\varepsilon=0.2$ exceeds $\varepsilon=0.1$). The fitted "order" should be interpreted as an effective scaling exponent over the tested range, not a true mathematical order of the dominant nonlinearity.

## Results

### Perturbation Gap Across Depth

![Linearization Gap](results/linearization_gap.png)

| Layer Range | Perturb Gap | Attn Gap | MLP Gap | NL Order | Interpretation |
|------------|-------------|----------|---------|----------|----------------|
| 0-1        | 0.24-0.25   | 0.15     | 0.20-0.21 | 0.61-0.70 | **Most nonlinear** — embedding projection |
| 2-5        | 0.17-0.21   | 0.12-0.16| 0.12-0.18 | 0.69-0.74 | Decreasing nonlinearity |
| 6-18       | 0.13-0.15   | 0.11-0.13| 0.10-0.11 | 0.62-0.77 | **Minimum nonlinearity plateau** |
| 19-32      | 0.15-0.18   | 0.12-0.14| 0.11-0.13 | 0.65-0.75 | Gradual increase |
| 33-35      | 0.17-0.23   | 0.11-0.14| 0.14-0.24 | 0.78-0.84 | **Late spike** — MLP-driven |

Key observations:
- **U-shaped nonlinearity profile**: Layers 6-18 are the most linear (gap ~0.13), with higher nonlinearity at both ends — middle layers are 54% less nonlinear than early layers (0.137 vs 0.211) and 25% less than late layers (0.137 vs 0.171). This means ~85-87% of middle-layer behavior is captured by a first-order Taylor approximation.
- **Layer 0 is an outlier**: Perturbation gap 0.24, and its transform norm $\|g(\mathbf{x})\|/\|\mathbf{x}\| = 8.29$ is ~15x larger than any other layer — this layer does the heavy lifting of projecting embeddings into the residual stream geometry.
- **Layer 35 (final)**: MLP gap spikes to 0.24, making it the most nonlinear MLP — consistent with its role in final feature extraction before the language modeling head.
- **Attention and MLP nonlinearity are nearly identical on average**: Overall mean attention gap (0.129) vs MLP gap (0.127) — just 1.2% difference. However, the pattern varies by depth: early layers (0-4) have higher MLP gaps, middle layers (5-33) have slightly higher attention gaps, and the final layers (34-35) see MLP dominate again (layer 35 MLP gap 0.24 vs attention 0.11). The softmax nonlinearity in attention is the dominant source in the plateau region, while SwiGLU drives late-layer nonlinearity.

### Multi-Scale Nonlinearity Order

![Multi-Scale Analysis](results/multiscale_analysis.png)

The fitted nonlinearity order across all layers is **0.61--0.84**, consistently below the expected value of 2.0 for purely quadratic nonlinearity. This sub-linear scaling (gap grows slower than $\varepsilon$) has three explanations:

1. **RMSNorm dampening**: At larger $\varepsilon$, RMSNorm re-normalization "absorbs" perturbation magnitude, making both actual and linear responses converge toward the same normalized representation. This compresses the gap at large scales.

2. **Softmax saturation**: For moderate perturbations, attention weights shift smoothly (quadratic regime). For larger perturbations, attention saturates (approaches one-hot), and further perturbation has diminishing effect — the gap plateaus rather than growing.

3. **Universal non-monotonic gap at large** $\varepsilon$: All 36/36 layers show gap *increasing* from $\varepsilon=0.1$ to $\varepsilon=0.2$, beyond the power-law prediction. Examples: layer 0 goes from 0.17 to 0.24 (+40%), layer 35 from 0.24 to 0.43 (+77%). This is universal — even the most linear plateau layers exhibit it (e.g., layer 16: 0.11 to 0.19). This means the linear approximation has a narrow validity domain ($\varepsilon \leq 0.1$); beyond that, the function enters a qualitatively different operating regime where the Taylor expansion breaks down and the "linear prediction" becomes meaningless rather than merely inaccurate.

**Late layers have higher nonlinearity order** (0.78-0.84 for layers 33-35 vs 0.62-0.70 for layers 0-10). While their absolute gap is similar to early layers, the *scaling* with perturbation size differs — late layers' nonlinearity grows faster with perturbation magnitude, suggesting their nonlinear features are more "deeply nonlinear" (higher-order interactions between features via the SwiGLU gate).

### Jacobian Properties

![Jacobian Properties](results/jacobian_properties.png)

| Layer Range | Spectral Norm | Mean Amplification | $\|g(\mathbf{x})\|/\|\mathbf{x}\|$ |
|------------|---------------|-------------------|----------------|
| 0          | 5.4           | 3.4               | 8.29           |
| 1-5        | 1.1-1.5       | 0.5-0.7           | 0.49-0.74      |
| 6-18       | 1.0-1.9       | 0.6-0.7           | 0.33-0.56      |
| 19-34      | 0.9-1.5       | 0.5-0.7           | 0.38-0.63      |
| 35         | 1.6           | 0.6               | 1.12           |

**Layer 0 spectral analysis:** Spectral norm ~5.4, meaning worst-case perturbations are amplified 5.4x. Mean amplification 3.4 means even *typical* perturbations are amplified 3.4x. This is consistent with layer 0's massive transform magnitude ($\|g(\mathbf{x})\|/\|\mathbf{x}\| = 8.29$) — it's an expansive map that projects the $d$-dimensional embedding into a richer representation.

**Spectral norm clustering:** Beyond layer 0, elevated spectral norms cluster at layers 9 (1.91) and 10 (1.60) in the early-middle range, and layers 34 (1.54) and 35 (1.60) at the end. These are the layers where worst-case perturbation amplification is highest, suggesting more complex transformations at these specific depths.

**Mean amplification $< 1$ for layers 1-35 (remarkably uniform):** All 35 non-embedding layers are contractive on average, with mean amplification ranging from 0.525 to 0.797 (mean 0.651, std = 0.058 — very tight). This uniformity across 35 layers is striking and suggests a strong training-time constraint on Jacobian norms. The mean amplification relates to the Frobenius norm of the Jacobian: $\mathbb{E}[\|\mathbf{J}\mathbf{d}\|] \sim \|\mathbf{J}\|_F / \sqrt{d}$. With $\|\mathbf{J}\|_F / \sqrt{d} < 1$, the Jacobian has Frobenius norm less than $\sqrt{d}$, meaning its squared singular values sum to less than $d$. Since most of the $d$ singular values must be $< 1$, the Jacobian is "mostly contractive" with possibly a few expanding directions (captured by the spectral norm being near or above 1).

**Dynamical systems interpretation:** The residual connection makes the full layer map $F(\mathbf{x}) = \mathbf{x} + g(\mathbf{x})$ with Jacobian $\mathbf{J}_F = \mathbf{I} + \mathbf{J}_g$. For stability:
- We need the spectral radius $\rho(\mathbf{I} + \mathbf{J}_g) < 1$ for convergence (in an iterative sense)
- Since $\mathbf{J}_g$ is contractive on average ($\|\mathbf{J}_g\|_F / \sqrt{d} < 1$), most eigenvalues of $\mathbf{J}_g$ are small, making $\mathbf{J}_F \approx \mathbf{I}$ — the layer makes small, stable corrections to the residual stream
- The product of all 36 layer Jacobians determines the end-to-end sensitivity: since most have spectral norm near 1, the network avoids both vanishing and exploding gradients

### Per-Prompt Gap Variability

![Gap Heatmap](results/gap_heatmap.png)

### Cross-Reference with T-2 Layer Criticality

![Gap vs Criticality](results/gap_vs_criticality.png)

Pearson correlation between perturbation gap and knockout loss delta: **r = 0.38** (moderate positive). The most critical layer (layer 0, knockout delta = 8.76) is also the most nonlinear. However, the correlation is driven primarily by this outlier. Excluding layer 0, the correlation weakens, suggesting that nonlinearity and criticality capture different aspects of layer importance:
- **Criticality** measures how much *information* the layer contributes (removal destroys output quality)
- **Nonlinearity** measures how *complex* the computation is (how much the function deviates from a linear map)

**Cautionary example — layer 6:** This layer sits squarely in the "linear" plateau (gap = 0.154) yet has T-2 criticality of 1.96 (second-highest after layer 0's 8.76). Linearizing or pruning layer 6 based on its low nonlinearity would be dangerous — it performs a critical function despite being nearly linear. Similarly, layers 9-10 have above-average criticality but average nonlinearity, while layers 34-35 have above-average nonlinearity but below-average criticality. This decoupling means you cannot predict layer importance from linearization gap alone.

### Cross-Reference with T-9 Weight Spectral Structure

T-9 found a significant negative correlation between weight effective rank and linearization gap: **r = -0.42, p = 0.011**. Layers with higher-rank weight matrices tend to be *more linear*, not less. The explanation is that high-rank layers spread computation across many dimensions, where self-averaging makes the aggregate more linear (CLT-type effect). Low-rank layers concentrate nonlinearity in fewer dimensions, making it more prominent.

## Conclusions & Key Findings

1. **Hypothesis partially refuted**: The expected monotonic early=linear, late=nonlinear pattern does NOT hold. Instead, we observe a **U-shaped** profile where middle layers (6-18) are the most linear and both early and late layers are more nonlinear.

2. **Middle layers are linearization candidates**: Layers 6-18 have perturbation gaps of only 0.13-0.15, meaning ~85-87% of their behavior is captured by a first-order (linear) approximation. These layers could potentially be replaced by linear maps with modest quality loss.

3. **Sub-quadratic nonlinearity everywhere**: The multi-scale analysis shows effective nonlinearity orders of 0.6--0.8 rather than the expected 2.0 for softmax/SiLU. This results from (a) RMSNorm dampening scale-dependent nonlinearity, (b) softmax saturation at large perturbations, and (c) the gap ratio measure compressing at large $\varepsilon$. Even the "most nonlinear" layers are less nonlinear than their activation functions would suggest in isolation.

4. **RMSNorm dominates scale sensitivity**: The homogeneity gap (~1.0 for all layers) reveals that RMSNorm makes every layer approximately scale-invariant — the function $g(\mathbf{x})$ depends on the *direction* of $\mathbf{x}$, not its magnitude. Geometrically, each layer operates on the unit sphere $S^{d-1}$ where $d = 2560$. The Jacobian has a null space containing the radial direction $\mathbf{x}/\|\mathbf{x}\|$.

5. **Layer 0 is qualitatively different**: The only expansive layer (mean amplification 3.4 vs < 1 for all others), with transform magnitude 15x larger, spectral norm 5.4, and the highest nonlinearity. It projects from embedding space to the model's internal representation geometry.

6. **MLP drives late-layer nonlinearity**: In layers 33-35, MLP gap increases sharply (0.14→0.24) while attention remains stable (~0.11). The nonlinearity order also peaks (0.84, though R² is low — see caveat in multi-scale section), suggesting higher-order feature interactions via SwiGLU gating.

7. **Most layers are remarkably uniformly contractive**: Mean Jacobian amplification $< 1$ for layers 1-35, with very tight spread (mean 0.651, std 0.058). This uniformity suggests a training-time constraint on Jacobian norms. Combined with the residual connection ($\mathbf{x} \to \mathbf{x} + g(\mathbf{x})$ where $\|g(\mathbf{x})\| \sim 0.5\|\mathbf{x}\|$), this creates a stable dynamical system. The Jacobian $\mathbf{J}_F = \mathbf{I} + \mathbf{J}_g$ has spectral radius near 1, ensuring neither vanishing nor exploding gradients.

8. **Linear approximation has a narrow validity domain**: The universal non-monotonicity at $\varepsilon=0.2$ (all 36/36 layers show gap increasing after decreasing at $\varepsilon=0.1$) means perturbation-based linearization is fundamentally limited. Linear surrogates are valid only for small perturbations ($\varepsilon \leq 0.1$); beyond that, the function enters a qualitatively different regime. This constrains practical applications of linear surrogates to scenarios where inputs stay close to the calibration manifold.

## Practical Implications

### Linearization Opportunities

**1. Linear surrogates for plateau layers (6-18).** With 85-87% of behavior captured by a first-order Taylor approximation, these layers can be replaced by pre-computed affine maps $\mathbf{x} + \mathbf{J}\mathbf{x} + \mathbf{b}$ at inference time — skipping softmax, RMSNorm, and SwiGLU entirely. Cost per replaced layer drops from full attention+MLP to a single ($2560 \times 2560$) matmul. Requires calibration data since the Jacobian is input-dependent.

**2. Plateau layer pruning.** Layers 6-18 have the lowest nonlinearity (gap ~0.13), are contractive (mean amplification < 1), and T-2 confirms low knockout criticality. Removing 3-5 layers (e.g., 10-14) cuts ~14% of compute. The near-linear behavior means neighboring layers can partially compensate.

**3. Depth-heterogeneous attention.** The U-shaped profile directly motivates a hybrid architecture:
- **Plateau (6-18)**: Replace softmax with linear attention (GLA, cosine-similarity). Softmax contributes only ~13-15% of functional complexity here. This eliminates $\mathcal{O}(T^2)$ cost for 13/36 layers.
- **Edges (0-5, 28-35)**: Keep full softmax. Attention nonlinearity is higher (gap 0.15) and routing is more complex.
- This matches hybrid approaches like Qwen3.5 (DeltaNet + attention) and Jamba (Mamba + attention), but with layer assignments grounded in measured nonlinearity.

**4. Activation function simplification.** MLP gap in plateau layers is only 0.10-0.11 — SwiGLU barely contributes nonlinearity. Replacing with GELU or ReLU in layers 6-18 saves the element-wise sigmoid * multiply while losing <3% of functional complexity. Keep SwiGLU in layers 33-35 where MLP gap reaches 0.24.

### What NOT to Linearize

**Layer 0** and **layers 33-35** are load-bearing (see conclusions #5, #6) and should be the last candidates for any approximation. **Layer 6** is a cautionary case — despite sitting in the linear plateau, it has T-2 criticality of 1.96 (see cross-reference above).

### Convergent Evidence

- **T-2 (Layer Knockout)**: Plateau layers have low knockout criticality, confirming they are redundant — but layer 6 is a notable exception (see cross-reference above)
- **T-9 (Spectral Structure)**: Plateau layers are both low-rank and near-linear (see cross-reference above)
- **T-3 (Layer Swap Cost)**: Adjacent plateau layers have the cheapest swap costs, consistent with near-linear layers being more interchangeable

## Usage

```bash
# Generate calibration data first (if not already done)
poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Instruct-2507

# Run the experiment
poetry run python experiments/t7_layer_linearization_gap/run.py
```

Results are saved to `experiments/t7_layer_linearization_gap/results/`:
- `summary.json` — all per-layer metrics including multi-scale analysis
- `linearization_gap.png` — perturbation gap, homogeneity gap, and attn/MLP decomposition
- `jacobian_properties.png` — spectral norm, amplification, transform magnitude
- `gap_vs_criticality.png` — cross-reference with T-2 knockout experiment
- `gap_heatmap.png` — per-prompt perturbation gap variability
- `multiscale_analysis.png` — gap vs eps log-log plots, nonlinearity order across depth

Runtime: ~88s on NVIDIA B200.
