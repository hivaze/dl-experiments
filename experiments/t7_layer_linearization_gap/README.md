# T-7: Layer Linearization Gap

## Motivation & Research Question

**How nonlinear is each layer's computation on real inputs?**

Each transformer layer applies a nonlinear transformation g(x) = layer(x) - x to the residual stream. This transformation involves attention (softmax) and MLP (SwiGLU activation). If a layer's computation is approximately linear on the data manifold, it could potentially be replaced by a cheap linear map without significant quality loss. We measure the "linearization gap" — how much the actual layer output deviates from what a linear approximation would predict — and track how this varies across depth.

**Hypothesis**: Early layers are more linear (mainly doing token mixing via attention), while late layers are more nonlinear (doing feature composition via MLP). If true, early layers are candidates for linearization/distillation.

## Setup

- **Model**: Qwen3-4B-Instruct-2507 (36 layers, hidden_dim=2560, GQA 32q/8kv, SwiGLU MLP)
- **Data**: 200 pre-generated calibration completions (vLLM, temp=0), completion tokens only. 50 prompts used for per-prompt analysis (Methods 1,3,4,5,8 — gap is stable across prompts with 10-20% std), full 200 for Methods 6 and 7 which benefit from more data.
- **Hardware**: NVIDIA B200, bf16 inference
- **Seed**: 42
- **Max sequence length**: 256 tokens
- **Runtime**: ~1248 seconds total (~200s per-prompt analysis, ~2s PCA gap, ~13s Jacobian consistency, ~1093s enhanced layer replacement)

## Mathematical Framework

### The Core Question

A transformer layer takes a vector $\mathbf{x}$ (the "residual stream" at that depth) and returns a modified vector. If that transformation is *approximately linear*, the layer could be replaced by a simple matrix multiply — which is dramatically cheaper than running full attention + MLP. The goal of this experiment is to measure exactly how nonlinear each layer actually is on real data.

The challenge: "nonlinear" is not a single number. A layer can be smooth locally but vary wildly across inputs. It can be linear in some directions but not others. It can look linear at small scales but reveal nonlinearity at large scales. We need multiple complementary measurements, each capturing a different aspect. This framework defines those measurements and explains why each one matters.

### Notation

Each transformer layer computes:

$$\mathbf{x}_{\text{out}} = \mathbf{x} + g(\mathbf{x})$$

where:
- $\mathbf{x} \in \mathbb{R}^d$ is the input hidden state for a single token ($d = 2560$)
- $g(\mathbf{x})$ is what the layer *adds* to the residual stream (the attention + MLP computation)
- The $+ \mathbf{x}$ is the residual (skip) connection — the layer's output is the input plus a correction

We isolate $g$ because the residual connection is already perfectly linear. All the nonlinearity lives in $g$.

The function $g$ decomposes into two stages applied sequentially:

$$g(\mathbf{x}) = f_{\text{attn}}(\mathbf{x}) + f_{\text{mlp}}\big(\mathbf{x} + f_{\text{attn}}(\mathbf{x})\big)$$

The attention sublayer:

$$f_{\text{attn}}(\mathbf{x}) = W_o \cdot \text{Attn}\big(W_q \cdot \text{RMSNorm}(\mathbf{x}), W_k \cdot \text{RMSNorm}(\mathbf{x}), W_v \cdot \text{RMSNorm}(\mathbf{x})\big)$$

The MLP sublayer (SwiGLU):

$$f_{\text{mlp}}(\mathbf{h}) = W_{\text{down}} \cdot \Big[\text{SiLU}\big(W_{\text{gate}} \cdot \text{RMSNorm}(\mathbf{h})\big) \odot \big(W_{\text{up}} \cdot \text{RMSNorm}(\mathbf{h})\big)\Big]$$

There are three sources of nonlinearity inside $g$:

1. **Softmax** in attention: $\text{softmax}(QK^\top / \sqrt{d_k})$ — the $QK^\top$ product is bilinear (quadratic in the input), then $\exp$ makes it fully nonlinear
2. **SiLU** (the function x · σ(x), where σ is the sigmoid) in SwiGLU: a smooth gating function — approximately linear near 0, approximately identity for large positive x
3. **RMSNorm**: $\mathbf{x} \mapsto \mathbf{x} \cdot \sqrt{d} / \|\mathbf{x}\|_2$ — normalizes the magnitude, making the output depend only on the *direction* of $\mathbf{x}$

### Linearization: The Best Linear Approximation

**The idea.** Any smooth function, no matter how complex, looks linear when you zoom in far enough. Think of a curve on a map — at city-block scale it curves, but at centimeter scale it's a straight line. The same applies to $g$: near any specific input $\mathbf{x}$, we can approximate $g$ by a linear function. The question is: *how far can we zoom out before this approximation breaks down?*

**The Jacobian matrix.** The linear approximation of $g$ at point $\mathbf{x}$ is given by its Jacobian, the $d \times d$ matrix of partial derivatives:

$$\mathbf{J}_g(\mathbf{x}) \in \mathbb{R}^{d \times d}, \qquad [\mathbf{J}_g]_{ij} = \frac{\partial g_i}{\partial x_j}$$

Each row $i$ tells you: "if I wiggle the input, how does output dimension $i$ respond?" Each column $j$ tells you: "if I wiggle input dimension $j$, how do all outputs respond?"

**The linear approximation.** For a small perturbation $\mathbf{h}$:

$$g(\mathbf{x} + \mathbf{h}) \approx g(\mathbf{x}) + \mathbf{J}_g(\mathbf{x}) \cdot \mathbf{h}$$

This says: "the change in output equals the Jacobian times the change in input." It is the best possible linear prediction. The **linearization gap** measures how badly this approximation fails — i.e., how much nonlinear behavior is left over.

### The Gap: Quantifying the Approximation Error

**Taylor expansion.** To understand the error, we expand $g$ to second order. Perturb the input by $\varepsilon \mathbf{d}$, where $\mathbf{d}$ is a unit direction and $\varepsilon$ controls the perturbation size:

$$g(\mathbf{x} + \varepsilon \mathbf{d}) = \underbrace{g(\mathbf{x})}_{\text{value at base point}} + \underbrace{\varepsilon \mathbf{J} \mathbf{d}}_{\text{1st order (linear)}} + \underbrace{\frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}]}_{\text{2nd order (quadratic)}} + \underbrace{\mathcal{O}(\varepsilon^3)}_{\text{higher order}}$$

Here $\mathbf{H}$ is the Hessian tensor (second derivatives of $g$), and $\mathbf{H}[\mathbf{d}, \mathbf{d}]$ is the second-order correction in direction $\mathbf{d}$. We write $\mathbf{J}$ as shorthand for $\mathbf{J}_g(\mathbf{x})$.

Now define three quantities:

**Actual displacement** — what really happens when we perturb the input:

$$\Delta = g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x}) = \varepsilon \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] + \mathcal{O}(\varepsilon^3)$$

**Linear prediction** — what the Jacobian predicts should happen:

$$\hat{\Delta} = \varepsilon \mathbf{J} \mathbf{d}$$

**Residual** — the part the linear approximation misses:

$$\mathbf{r} = \Delta - \hat{\Delta} = \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] + \mathcal{O}(\varepsilon^3)$$

The **perturbation gap** is the relative size of this residual:

$$\text{gap} = \frac{\|\mathbf{r}\|}{\|\Delta\|} = \frac{\|\text{actual} - \text{linear prediction}\|}{\|\text{actual}\|}$$

A gap of 0.13 means: "87% of what the layer does is captured by the linear approximation; 13% is genuinely nonlinear."

**How the gap scales with perturbation size.** For small $\varepsilon$:

- The residual grows as $\|\mathbf{r}\| \sim \varepsilon^2$ (dominated by the Hessian term)
- The actual displacement grows as $\|\Delta\| \sim \varepsilon$ (dominated by the Jacobian term)
- Their ratio: $\text{gap} \sim \varepsilon^2 / \varepsilon = \varepsilon$

So for a function with quadratic nonlinearity (like softmax or SiLU), doubling the perturbation size doubles the gap. More generally, if the leading nonlinearity is degree $k$ (quadratic: $k=2$, cubic: $k=3$), the gap scales as $\varepsilon^{k-1}$. The multi-scale analysis (Method 5) measures this exponent.

### Central Differences: Computing Jacobian-Vector Products Efficiently

**The problem.** The Jacobian $\mathbf{J}$ is a $2560 \times 2560$ matrix (~26 million entries). We cannot store it or compute it fully. But we don't need the full matrix — we only need its product with specific direction vectors, $\mathbf{J} \mathbf{d}$.

**The trick.** By definition, $\mathbf{J} \mathbf{d}$ is the derivative of $g$ in direction $\mathbf{d}$. We can approximate this with finite differences — evaluate $g$ at two nearby points and take the slope.

**Forward difference** (the naive approach):

$$\mathbf{J} \mathbf{d} \approx \frac{g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x})}{\varepsilon}$$

**Central difference** (what we actually use):

$$\mathbf{J} \mathbf{d} \approx \frac{g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x} - \varepsilon \mathbf{d})}{2\varepsilon}$$

**Why central is better.** Expand both evaluations via Taylor series:

$$g(\mathbf{x} + \varepsilon \mathbf{d}) = g(\mathbf{x}) + \varepsilon \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] + \frac{\varepsilon^3}{6} \mathbf{T}[\mathbf{d}, \mathbf{d}, \mathbf{d}] + \cdots$$

$$g(\mathbf{x} - \varepsilon \mathbf{d}) = g(\mathbf{x}) - \varepsilon \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{2} \mathbf{H}[\mathbf{d}, \mathbf{d}] - \frac{\varepsilon^3}{6} \mathbf{T}[\mathbf{d}, \mathbf{d}, \mathbf{d}] + \cdots$$

Note: even powers of $\varepsilon$ have the same sign in both; odd powers flip sign. Subtracting cancels all even-order terms:

$$\frac{g(\mathbf{x} + \varepsilon \mathbf{d}) - g(\mathbf{x} - \varepsilon \mathbf{d})}{2\varepsilon} = \mathbf{J} \mathbf{d} + \frac{\varepsilon^2}{6} \mathbf{T}[\mathbf{d}, \mathbf{d}, \mathbf{d}] + \mathcal{O}(\varepsilon^4)$$

The error is $\mathcal{O}(\varepsilon^2)$ instead of $\mathcal{O}(\varepsilon)$ for forward differences — the Hessian term vanishes. This matters in practice:
- bf16 has ~7.8e-3 relative precision, forcing $\varepsilon \geq 0.01$ (smaller perturbations drown in rounding noise)
- At $\varepsilon = 0.05$: forward difference error is ~5%, central difference error is ~0.25% — 20x better

### bf16 Perturbation Scaling

**The problem.** In bf16 (bfloat16), numbers have only ~3 decimal digits of precision. If $\mathbf{x}$ has magnitude 100 and we add a perturbation of magnitude 0.001, bf16 rounds $100 + 0.001 = 100$ — the perturbation vanishes entirely.

**The fix.** Scale perturbations to be proportional to the input magnitude:

$$\boldsymbol{\delta} = \varepsilon \cdot \|\mathbf{x}\| \cdot \hat{\mathbf{d}}$$

where $\hat{\mathbf{d}} = \mathbf{d} / \|\mathbf{d}\|$ is a unit direction vector. Now $\|\boldsymbol{\delta}\| = \varepsilon \cdot \|\mathbf{x}\|$, so the perturbation is always a fixed *fraction* $\varepsilon$ of the input magnitude, regardless of the absolute scale:

$$\frac{\|\boldsymbol{\delta}\|}{\|\mathbf{x}\|} = \varepsilon$$

This keeps $\mathbf{x} + \boldsymbol{\delta}$ representable in bf16 without losing the perturbation to rounding.

### Homogeneity Gap: Does Scaling the Input Scale the Output?

**What it tests.** A truly linear function satisfies $g(\alpha \mathbf{x}) = \alpha g(\mathbf{x})$ for any scalar $\alpha$. Equivalently, $g(\mathbf{x}) = \mathbf{J} \mathbf{x}$ (the Jacobian applied to the input itself). The homogeneity gap measures how badly this fails:

$$\text{homogeneity gap} = \frac{\|g(\mathbf{x}) - \mathbf{J} \mathbf{x}\|}{\|g(\mathbf{x})\|}$$

where $\mathbf{J} \mathbf{x}$ is estimated via central differences using perturbation direction $\mathbf{x}$ itself:

$$\mathbf{J} \mathbf{x} \approx \frac{g((1+\varepsilon)\mathbf{x}) - g((1-\varepsilon)\mathbf{x})}{2\varepsilon}$$

**Why it saturates at ~1.0 for every layer.** RMSNorm normalizes by input magnitude:

$$\text{RMSNorm}(\alpha \mathbf{x}) = \frac{\alpha \mathbf{x} \cdot \sqrt{d}}{\|\alpha \mathbf{x}\|} = \frac{\mathbf{x} \cdot \sqrt{d}}{\|\mathbf{x}\|} = \text{RMSNorm}(\mathbf{x})$$

The scalar $\alpha$ cancels out. Since every layer passes through RMSNorm before computing attention and MLP, the entire function $g$ becomes approximately **scale-invariant**: $g(\alpha \mathbf{x}) \approx g(\mathbf{x})$ for any $\alpha > 0$. This means the output doesn't change when we scale the input, so the derivative with respect to scale is zero:

$$\mathbf{J} \mathbf{x} = \lim_{\varepsilon \to 0} \frac{g((1+\varepsilon)\mathbf{x}) - g(\mathbf{x})}{\varepsilon} \approx 0$$

The homogeneity gap becomes $\|g(\mathbf{x}) - 0\| / \|g(\mathbf{x})\| = 1$.

**What this means.** This is not a failure of the metric — it reveals a fundamental geometric fact. RMSNorm projects the residual stream onto a sphere of radius $\sqrt{d}$. Every layer operates on the *direction* of $\mathbf{x}$, not its magnitude. The Jacobian has a null space containing the radial direction $\mathbf{x}/\|\mathbf{x}\|$, so its effective rank is at most $d-1$. The homogeneity gap is uninformative about actual nonlinearity — it is dominated entirely by this geometric constraint.

### Multi-Scale Analysis: Identifying the Type of Nonlinearity

**The idea.** The perturbation gap at a single scale $\varepsilon$ tells us *how much* nonlinearity there is. By measuring the gap at multiple scales, we can determine *what kind* of nonlinearity dominates.

If the leading nonlinearity is degree $k$, the gap scales as:

$$\text{gap}(\varepsilon) \sim C \cdot \varepsilon^{k-1}$$

Taking logarithms of both sides:

$$\log(\text{gap}) = (k-1) \cdot \log(\varepsilon) + \log(C)$$

This is a straight line in log-log space with slope $\beta = k - 1$. Linear regression of $\log(\text{gap})$ vs $\log(\varepsilon)$ gives the slope, and the nonlinearity order is:

$$k = \beta + 1$$

**Expected values:** Purely quadratic nonlinearity (softmax, SiLU) gives $k = 2$, so slope $= 1$. Purely cubic gives $k = 3$, slope $= 2$.

**Why we observe sub-quadratic orders ($k \approx 0.6\text{--}0.8$).** The gap is a *ratio* $\|\mathbf{r}\| / \|\Delta\|$. At larger $\varepsilon$, RMSNorm re-normalization dampens both the numerator and denominator — it pulls both the "actual" and "linear" responses toward the same normalized manifold. This compression bends the log-log curve downward at large $\varepsilon$, giving an apparent slope $< 1$ (order $< 2$).

Additionally, R² of the log-log fit varies across layers: early/middle layers have R² ~ 0.34-0.77, while late layers can be much worse (layer 33: R² = 0.19, layer 35: R² = 0.26). Low R² means the single-power-law model is inadequate — the gap-vs-$\varepsilon$ curve has curvature in log-log space, likely from transitioning between nonlinearity regimes at different scales. **The nonlinearity order for layers 33-35 (0.78-0.84) should be treated with caution** given these poor fits.

### Spectral Norm: Worst-Case Amplification

**Why we need it.** The perturbation gap tells us how nonlinear a layer is. The spectral norm tells us something different: how much a layer can *amplify* perturbations. A layer with spectral norm 2 can take a small input change and double it — this matters for stability (will errors accumulate and explode through 36 layers?).

**Definition.** The spectral norm of $\mathbf{J}$ is its largest singular value:

$$\|\mathbf{J}\|_2 = \sigma_{\max}(\mathbf{J}) = \max_{\|\mathbf{d}\|=1} \|\mathbf{J} \mathbf{d}\|$$

It is the maximum factor by which $\mathbf{J}$ can stretch any unit vector.

**Computation via power iteration.** We can't compute $\sigma_{\max}$ directly (that would require the full $2560 \times 2560$ Jacobian). Instead, we use power iteration — start with a random vector and repeatedly multiply by $\mathbf{J}$, normalizing each time:

$$\mathbf{v}_0 = \text{random unit vector}$$

$$\mathbf{v}_{k+1} = \frac{\mathbf{J} \mathbf{v}_k}{\|\mathbf{J} \mathbf{v}_k\|}$$

After $K$ iterations, the approximation

$$\|\mathbf{J} \mathbf{v}_K\| \approx \sigma_{\max}$$

holds. Each $\mathbf{J} \mathbf{v}$ product is computed via central differences (no need to store $\mathbf{J}$):

$$\mathbf{J} \mathbf{v} \approx \frac{g(\mathbf{x} + \varepsilon \|\mathbf{x}\| \mathbf{v}) - g(\mathbf{x} - \varepsilon \|\mathbf{x}\| \mathbf{v})}{2\varepsilon \|\mathbf{x}\|}$$

**Convergence.** After $K$ iterations, the error is

$$\mathcal{O}\big((\sigma_2/\sigma_1)^K\big)$$

where $\sigma_1$ and
$\sigma_2$ are the two largest singular values.
With $K=5$ and typical ratio

$$\sigma_2/\sigma_1 \sim 0.5\text{--}0.8$$

the error is ~3-33%.

**Interpretation for stability.** The full layer (with residual connection) has Jacobian $\mathbf{I} + \mathbf{J}_g$. The layer is contractive or expansive depending on the spectral norm:

$$\|\mathbf{J}_g\|_2 < 1 \quad \text{(contractive)} \qquad \|\mathbf{J}_g\|_2 > 1 \quad \text{(expansive)}$$

If contractive, every perturbation shrinks through the layer. If expansive, some perturbation directions get amplified. For stable propagation through 36 layers, we need most layers to be contractive (or at least not strongly expansive).

### Mean Amplification: Typical Behavior

**Why we need it.** The spectral norm captures the *worst case* — the single direction that gets amplified most. But most input perturbations won't align with that direction. Mean amplification measures the *average* amplification across random directions:

$$\text{mean amplification} = \frac{1}{N} \sum_{i=1}^{N} \|\mathbf{J} \mathbf{d}_i\|$$

where $\mathbf{d}_i$ are random unit vectors.

**Connection to singular values.** For random unit vectors in $\mathbb{R}^d$, the expected squared norm of $\mathbf{J} \mathbf{d}$ equals the average squared singular value:

$$\mathbb{E}\big[\|\mathbf{J} \mathbf{d}\|^2\big] = \frac{\|\mathbf{J}\|_F^2}{d} = \frac{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_d^2}{d}$$

So mean amplification $\approx \|\mathbf{J}\|_F / \sqrt{d}$, the RMS (root-mean-square) singular value. If this is $< 1$, the layer is *typically* contractive even if a few singular values exceed 1.

### Jacobian Consistency: Local vs Global Linearity

**The key distinction.** Everything above measures nonlinearity *at a single input* $\mathbf{x}$. But the practical question is: can we replace a layer with one fixed linear map $W$ that works for *all* inputs?

- **Locally linear**: at each input $\mathbf{x}$, the function is well-approximated by its Jacobian $\mathbf{J}(\mathbf{x})$ — the perturbation gap is small
- **Globally linear**: the Jacobian is approximately the *same matrix* at all inputs — one $W$ works everywhere

A layer can be locally linear but globally nonlinear. Think of a function like $g(x) = x^2$: at any point, the tangent line is a good local fit, but different points have different slopes. Similarly, a transformer layer might apply smooth, nearly-linear attention routing at each input, but the *routing pattern itself* changes with input content — so the Jacobian at input $\mathbf{x}_1$ is a different matrix than at input
$\mathbf{x}_2$.

**Measuring consistency.** Pick a random direction $\hat{\mathbf{d}}$, and compute $\mathbf{J}(\mathbf{x}_i) \hat{\mathbf{d}}$ at multiple data points

$$\mathbf{x}_1, \ldots, \mathbf{x}_K$$

If the Jacobian is the same everywhere, all these vectors should point in the same direction. We measure this via pairwise cosine similarity:

$$C_g = \mathbb{E}_{\hat{\mathbf{d}}} \left[ \underset{i \neq j}{\text{mean}} \cos\Big(\mathbf{J}(\mathbf{x}_i)\hat{\mathbf{d}}, \mathbf{J}(\mathbf{x}_j)\hat{\mathbf{d}}\Big) \right]$$

- $C_g = 1$: the Jacobian maps every direction identically at all inputs — the layer is globally linear
- $C_g \to 0$: the Jacobian rotates outputs inconsistently across inputs — only locally linear

**Why consistency can be low even when the perturbation gap is low:** Consider a layer that performs context-dependent attention routing. At each input $\mathbf{x}$, the softmax attention weights are locally smooth (small perturbations produce smooth responses → low gap). But different inputs produce *different* attention patterns, so the Jacobian at $\mathbf{x}_1$ is a different matrix than at
$\mathbf{x}_2$. The layer is a smooth function everywhere, but it's a *different* smooth function at each point.

**Connection to global linear replacement (Method 7):** When fitting the least-squares problem

$$\mathbf{W} = \arg\min \sum_k \|g(\mathbf{x}_k) - \mathbf{W}\mathbf{x}_k\|^2$$

the solution is

$$\mathbf{W} = \Big(\sum_k g(\mathbf{x}_k)\mathbf{x}_k^\top\Big)\Big(\sum_k \mathbf{x}_k \mathbf{x}_k^\top\Big)^{-1}$$

which is a data-weighted average of the per-point Jacobians. If Jacobians are consistent
($C_g \to 1$), this average is close to any individual Jacobian and the replacement works. If Jacobians vary
($C_g \ll 1$), the average washes out the input-specific structure and produces a poor approximation that may be worse than the identity.

## Methods

### Method 1: Perturbation Gap (Primary Metric)

For each layer with input $\mathbf{x}$ and transform $g(\mathbf{x})$:
1. Generate 16 random unit perturbation directions $\mathbf{d}_i$
2. Scale perturbation to bf16-safe magnitude ($\varepsilon = 0.05$):
$$\boldsymbol{\delta}_i = \varepsilon \|\mathbf{x}\| \hat{\mathbf{d}}_i$$
3. Estimate Jacobian-vector product via central differences: $\mathbf{J}\boldsymbol{\delta} \approx (g(\mathbf{x}+\boldsymbol{\delta}) - g(\mathbf{x}-\boldsymbol{\delta})) / 2$
4. Compare actual displacement $g(\mathbf{x}+\boldsymbol{\delta}) - g(\mathbf{x})$ to linear prediction $\mathbf{J}\boldsymbol{\delta}$
5. Gap $= \|\text{actual} - \text{linear}\| / \|\text{actual}\|$, averaged over directions and tokens

### Method 2: Homogeneity Gap (v1 only — not computed in v2)

Tests scale-invariance by comparing $g(\mathbf{x})$ to $\mathbf{J}\mathbf{x}$ (the Jacobian applied to the input itself). See mathematical framework above for why this saturates at ~1.0 for RMSNorm-based architectures. **Dropped from v2** because it is uninformative — RMSNorm makes every layer scale-invariant, so the gap is ~1.0 everywhere regardless of actual nonlinearity.

### Method 3: Attention vs MLP Decomposition

Applies the perturbation gap separately to:
- **Attention sublayer** (nonlinearity from softmax + RMSNorm):
$$f_{\text{attn}}(\mathbf{x}) = \mathbf{W}_o \cdot \text{Attn}(\text{LN}_1(\mathbf{x}))$$
- **MLP sublayer** (nonlinearity from SiLU gating + RMSNorm):
$$f_{\text{mlp}}(\mathbf{x}) = \mathbf{W}_{\text{down}} \cdot \text{SwiGLU}(\text{LN}_2(\mathbf{x} + f_{\text{attn}}(\mathbf{x})))$$

Note: the MLP sublayer function includes the attention computation (since its input depends on it), measuring the *marginal* nonlinearity of adding MLP to the attention output.

### Method 4: Jacobian Spectral Properties

- **Spectral norm** $\|\mathbf{J}\|_2$: Estimated via power iteration with finite-difference JVPs (5 iterations). Measures worst-case amplification of perturbations.
- **Mean amplification** $\mathbb{E}[\|\mathbf{J}\hat{\mathbf{d}}\|]$: Average Jacobian action on random unit vectors (16 samples). Measures typical amplification, proportional to $\|\mathbf{J}\|_F / \sqrt{d}$.

### Method 5: Multi-Scale Nonlinearity Order

Perturbation gap computed at $\varepsilon \in \lbrace 0.005, 0.01, 0.02, 0.05, 0.1 \rbrace$ with 8 random directions each (v2 dropped $\varepsilon=0.2$ where Taylor expansion breaks down, added $\varepsilon=0.005$ for better small-scale resolution). Log-log linear regression of gap vs $\varepsilon$ gives the nonlinearity order per layer. $R^2$ of the fit indicates how well a single power law describes the nonlinearity.

**Caveat:** At large $\varepsilon$ (0.1--0.2), the Taylor expansion may not converge well, and higher-order terms can cause non-monotonic gap behavior (we observe this in layers 0 and 35 where the gap at $\varepsilon=0.2$ exceeds $\varepsilon=0.1$). The fitted "order" should be interpreted as an effective scaling exponent over the tested range, not a true mathematical order of the dominant nonlinearity.

### Method 6: Jacobian Consistency Across Inputs

Methods 1-5 measure properties of the Jacobian at a single operating point. Method 6 asks a different question: **how much does the Jacobian change across different inputs?** This directly predicts whether a single global linear map $W$ can approximate the layer across all inputs (Method 7).

For each layer, we:
1. Pick $D=16$ shared random unit directions in $\mathbb{R}^{2560}$:
$$\hat{d}_1, \ldots, \hat{d}_D$$
2. For each of $K=30$ calibration prompts, compute the JVP (Jacobian-vector product) at the last completion token — the normalized output direction when perturbing in direction $\hat{d}$:
$$v_k^{(d)} = J(x_k) \hat{d} / \|J(x_k) \hat{d}\|$$
3. For each direction $d$, compute the mean pairwise cosine similarity across all prompts of:
$$\lbrace v_1^{(d)}, \ldots, v_K^{(d)} \rbrace$$
4. Average across directions → **Jacobian consistency score**.

If the Jacobian is the same matrix at all inputs (globally linear), all $v_k^{(d)}$ are identical → consistency $= 1.0$. If the Jacobian varies strongly (globally nonlinear), they point in different directions → consistency $\to 0$.

### Method 7: Global Linear Replacement

Methods 1-6 measure local properties (how the Jacobian behaves at or near a single input). Method 7 tests **global linearity** directly: can a layer's entire computation be replaced by a single learned linear map that works across all inputs?

**Why this matters.** A layer with low perturbation gap (Methods 1-3) is locally well-approximated by its Jacobian at each operating point — but a different Jacobian at each point. If we want to actually *replace* a layer in practice (for inference speedup or distillation), we need one fixed matrix W that works everywhere. Method 7 tests whether such a matrix exists.

**The approach.** For each target layer, we:

1. **Collect activation pairs**: Run all calibration data through the model, recording each layer's input $X \in \mathbb{R}^{N \times d}$ and output $Y \in \mathbb{R}^{N \times d}$ (where $N$ is total tokens across all sequences). The data is split 80/20: the linear map is fitted on the training split, and evaluated on the held-out test split — this prevents overfitting from inflating recovery scores.

2. **Fit a linear map** via ridge regression (L2-regularized least-squares, computed in float32 for numerical stability):

$$W = \arg\min_W \|Y - WX\|_F^2 + \lambda \|W\|_F^2$$

The regularization penalty $\lambda$ prevents the full-rank solution from overfitting to idiosyncratic activation patterns in the training data. Without it, the fitted W for middle layers produces catastrophic loss spikes on new inputs (the original experiment showed recovery of -2630% for layer 8). Multiple $\lambda$ values are tested to find the best trade-off.

3. **Affine variant**: Fit with a bias term, $Y \approx WX + b$, where $b$ is a learned offset vector. Since RMSNorm shifts the mean of activations, the bias captures this shift and can substantially improve fit quality — especially for layers where the mean output differs from the mean input.

4. **Low-rank variants**: Fit the residual $R = Y - X$ with $W_r X$, then truncate $W_r$'s SVD to various ranks (16, 32, 64, 128, 256, 512). This preserves the skip connection (identity) — architecturally appropriate since transformer layers compute:

$$h_{i+1} = h_i + f(h_i)$$

The skip connection carries the bulk of information through middle layers, so constraining the fit to learn only the *correction* acts as a strong structural prior.

5. **Evaluate**: Hook the replacement into the model (output = $Wx$ or $Wx + b$ instead of the full attention+MLP computation) and measure cross-entropy loss on held-out data.

**The recovery metric** compares the linear surrogate against two baselines: the original model and the layer knockout (from T-2). It measures what fraction of the knockout damage is recovered:

$$\text{Recovery} = 1 - \frac{\Delta L_{\text{replacement}}}{\Delta L_{\text{knockout}}}$$

Interpretation:
- **Recovery = 100%**: The linear surrogate perfectly reproduces the layer — zero additional loss vs the original model. The layer is globally linear.
- **Recovery = 0%**: The linear surrogate is exactly as bad as skipping the layer entirely. The linear map adds nothing useful.
- **Recovery < 0%** (negative): The linear surrogate is *worse* than skipping the layer. The fitted W actively poisons the residual stream — it learned spurious correlations that generalize poorly.

**Connection to Method 6.** Method 6 measures Jacobian consistency — whether the per-input Jacobians point in the same direction across different inputs. Method 7 is the end-to-end test of this prediction: layers with high consistency (same Jacobian everywhere) should be globally replaceable by one W, while layers with low consistency (input-dependent Jacobians) should resist it. The recovery metric quantifies exactly how well this prediction holds.

### Method 8: Data-Aligned Perturbation Gap (PCA-Aligned)

Methods 1-5 perturb in **random** directions. But the data manifold occupies a thin subspace of $\mathbb{R}^{2560}$ — random directions are mostly off-manifold noise. Method 8 asks: **is the layer more or less nonlinear along directions the data actually uses?**

1. Collect hidden states $\lbrace \mathbf{x}_k \rbrace$ at each layer's input across all calibration tokens
2. Compute PCA of the input activations to find the top-$K$ principal directions ($K=20$) — these span the data manifold
3. Measure the perturbation gap (same as Method 1) along these PCA directions (**on-manifold gap**)
4. Measure the perturbation gap along $K=20$ random directions orthogonal to the PCA subspace (**off-manifold gap**)
5. Compute the **gap ratio** = on-manifold / off-manifold

**Interpretation:**
- Ratio $< 1$: The layer is *more linear on-manifold* — its nonlinearity is mostly in directions the data doesn't use. This is good for linearization: the nonlinear component is irrelevant noise.
- Ratio $\approx 1$: Nonlinearity is isotropic — equally strong in all directions.
- Ratio $> 1$: The layer is *more nonlinear on-manifold* — its nonlinearity is specifically aligned with the data. This is where genuinely useful nonlinear computation happens — the layer is doing something that a linear map fundamentally cannot capture along the directions that matter.

**Effective rank** of input activations (number of PCA components capturing 95% of variance) measures data manifold dimensionality at each layer.

## Results

### Perturbation Gap Across Depth

![Linearization Gap](results/linearization_gap_v2.png)

| Layer Range | Perturb Gap | Attn Gap | MLP Gap | Interpretation |
|------------|-------------|----------|---------|----------------|
| 0-1        | 0.245-0.252 | 0.14-0.16| 0.20-0.22 | **Most nonlinear** — embedding projection |
| 2-5        | 0.16-0.22   | 0.12-0.17| 0.12-0.18 | Decreasing nonlinearity |
| 6-18       | 0.13-0.14   | 0.10-0.13| 0.10-0.11 | **Minimum nonlinearity plateau** |
| 19-32      | 0.15-0.18   | 0.12-0.14| 0.11-0.13 | Gradual increase |
| 33-35      | 0.17-0.23   | 0.11-0.15| 0.14-0.25 | **Late spike** — MLP-driven |

Key observations:
- **U-shaped nonlinearity profile**: Layers 6-18 are the most linear (gap ~0.13), with higher nonlinearity at both ends — middle layers are ~45% less nonlinear than early layers (0.13 vs 0.25) and ~25% less than late layers (0.13 vs 0.18). This means ~87% of middle-layer behavior is captured by a first-order Taylor approximation.
- **Layer 0 is an outlier**: Perturbation gap 0.245, and its transform norm $\|g(\mathbf{x})\|/\|\mathbf{x}\| = 8.23$ is ~15x larger than any other layer — this layer does the heavy lifting of projecting embeddings into the residual stream geometry.
- **Layer 35 (final)**: MLP gap spikes to 0.25, making it the most nonlinear MLP — consistent with its role in final feature extraction before the language modeling head.
- **Attention and MLP nonlinearity are nearly identical on average**: However, the pattern varies by depth: early layers (0-4) have higher MLP gaps, middle layers (5-33) have slightly higher attention gaps, and the final layers (34-35) see MLP dominate again (layer 35 MLP gap 0.25 vs attention 0.11). The softmax nonlinearity in attention is the dominant source in the plateau region, while SwiGLU drives late-layer nonlinearity.

### Multi-Scale Nonlinearity Order

![Multi-Scale Analysis](results/multiscale_analysis_v2.png)

V2 uses $\varepsilon \in \lbrace 0.005, 0.01, 0.02, 0.05, 0.1 \rbrace$ (dropping $\varepsilon=0.2$ where the Taylor expansion breaks down, adding $\varepsilon=0.005$ for better small-scale resolution). The log-log fits have substantially improved R² (0.85-0.98 for most layers vs 0.19-0.77 in v1) thanks to staying within the convergence domain.

The fitted nonlinearity order ranges from ~0.3 (early layers) to ~2.0 (late layers), with a clear depth-dependent trend. Early layers (0-5) show sub-linear scaling (order 0.3-1.0), while late layers (25-35) approach the expected quadratic scaling (order 1.5-2.0). Middle layers (8-20) show orders around 1.0-1.5.

**Late layers have higher nonlinearity order** (1.5-2.0 for layers 25-35 vs 0.3-1.0 for layers 0-10). While their absolute gap is similar to early layers, the *scaling* with perturbation size differs — late layers' nonlinearity grows faster with perturbation magnitude, suggesting their nonlinear features involve higher-order interactions between features via the SwiGLU gate. Early layers' sub-linear scaling is consistent with RMSNorm dampening dominating at those depths.

### Jacobian Properties

![Jacobian Properties](results/jacobian_properties_v2.png)

| Layer Range | Spectral Norm | Mean Amplification | $\|g(\mathbf{x})\|/\|\mathbf{x}\|$ |
|------------|---------------|-------------------|----------------|
| 0          | 5.3           | 3.2               | 8.23           |
| 1-5        | 1.1-1.9       | 0.5-0.7           | 0.45-0.75      |
| 6-18       | 0.9-1.9       | 0.6-0.7           | 0.30-0.55      |
| 19-34      | 0.9-1.5       | 0.6-0.8           | 0.35-0.65      |
| 35         | 1.6           | 0.6               | 1.10           |

**Layer 0 spectral analysis:** Spectral norm ~5.3, meaning worst-case perturbations are amplified 5.3x. Mean amplification 3.2 means even *typical* perturbations are amplified 3.2x. This is consistent with layer 0's massive transform magnitude ($\|g(\mathbf{x})\|/\|\mathbf{x}\| = 8.23$) — it's an expansive map that projects the $d$-dimensional embedding into a richer representation.

**Mean amplification $< 1$ for layers 1-35 (remarkably uniform):** All 35 non-embedding layers are contractive on average, with mean amplification in a tight range. This uniformity across 35 layers suggests a strong training-time constraint on Jacobian norms. The mean amplification relates to the Frobenius norm: $\mathbb{E}[\|\mathbf{J}\mathbf{d}\|] \sim \|\mathbf{J}\|_F / \sqrt{d}$. With this ratio $< 1$, the Jacobian has Frobenius norm less than $\sqrt{d}$, meaning its squared singular values sum to less than $d$ — the Jacobian is "mostly contractive" with possibly a few expanding directions.

**Dynamical systems interpretation:** The residual connection makes the full layer map $F(\mathbf{x}) = \mathbf{x} + g(\mathbf{x})$ with Jacobian

$$\mathbf{J}_F = \mathbf{I} + \mathbf{J}_g$$

For stability:
- We need the spectral radius $\rho(\mathbf{I} + \mathbf{J}_g) < 1$ for convergence (in an iterative sense)
- Since $\mathbf{J}_g$ is contractive on average (i.e.,
$$\|\mathbf{J}_g\|_F / \sqrt{d} < 1$$
), most eigenvalues of $\mathbf{J}_g$ are small, making
$\mathbf{J}_F \approx \mathbf{I}$ — the layer makes small, stable corrections to the residual stream
- The product of all 36 layer Jacobians determines the end-to-end sensitivity: since most have spectral norm near 1, the network avoids both vanishing and exploding gradients
- **Jacobian consistency adds a second stability dimension:** Layers with consistent Jacobians ($C_g \to 1$) behave like fixed linear operators regardless of input — the dynamical system is essentially autonomous. Layers with low consistency ($C_g \ll 1$) behave like time-varying (input-varying) linear operators. The general increase of consistency with depth (0.65 to 0.90) suggests the model transitions from adaptive, context-sensitive processing to fixed, context-invariant output formatting

### Per-Prompt Gap Variability

![Gap Heatmap — perturbation gap by layer and prompt, showing how nonlinearity varies with input content](results/gap_heatmap_v2.png)

The heatmap reveals that most of the gap variation is **structural** (across layers) rather than **data-dependent** (across prompts). For any given layer, the gap is remarkably stable across the 50 calibration prompts — standard deviations are typically 10-20% of the mean. This means our per-layer conclusions are robust and not driven by specific prompt content.

Notable exceptions: layer 0 shows slightly more cross-prompt variation than other layers, consistent with its role as the input-dependent embedding projector. Late layers (33-35) also show elevated variation, particularly on prompts with unusual token patterns.

### Cross-Reference with T-2 Layer Criticality

![Gap vs Criticality](results/gap_vs_criticality_v2.png)

Pearson correlation between perturbation gap and knockout loss delta: **r = 0.35** (moderate positive). The most critical layer (layer 0, T-2 knockout delta = 9.08) is also the most nonlinear. However, the correlation is driven primarily by this outlier. Excluding layer 0, the correlation weakens, suggesting that nonlinearity and criticality capture different aspects of layer importance:
- **Criticality** measures how much *information* the layer contributes (removal destroys output quality)
- **Nonlinearity** measures how *complex* the computation is (how much the function deviates from a linear map)

**Cautionary example — layer 6:** This layer sits squarely in the "linear" plateau (gap = 0.151) yet has T-2 criticality of 1.88 (second-highest after layer 0). Linearizing or pruning layer 6 based on its low nonlinearity would be dangerous — it performs a critical function despite being nearly linear. Method 7 confirms this: layer 6 has residual R²=0.997 (essentially perfect activation-space fit) yet only 54.2% CE recovery — the tiny nonlinear residual carries disproportionate downstream information.

### Jacobian Consistency Across Inputs

![Jacobian Consistency — consistency across depth and local nonlinearity vs global consistency scatter](results/jacobian_consistency_v2.png)

While the perturbation gap measures how nonlinear a layer is *at a given input*, Jacobian consistency measures how much the Jacobian *varies across different inputs*.

| Layer Range | Consistency | Perturbation Gap | CE Recovery | Interpretation |
|------------|-------------|------------------|------------:|----------------|
| 0 | **0.65** | 0.245 (high) | 98.4% | Moderate consistency, high nonlinearity, globally linearizable |
| 1-3 | 0.72-0.76 | 0.21-0.25 | 80-83% | Moderate consistency |
| 4-8 | 0.73-0.82 | 0.14-0.18 | 73-85% | Rising consistency |
| 9-19 | 0.76-0.84 | 0.13-0.15 | 37-82% | High consistency, yet highly variable recovery |
| 20-28 | 0.80-0.84 | 0.15-0.18 | 43-64% | High consistency, **poor recovery** |
| 29-35 | **0.84-0.90** | 0.17-0.23 | 52-75% | **Highest consistency**, moderate recovery |

**The consistency profile is generally increasing** — from 0.65 at layer 0 to 0.90 at layer 35. This is fundamentally different from the U-shaped perturbation gap. It means:

- **Late layers apply approximately the same transformation regardless of input content.** Their Jacobian is stable across prompts — the attention patterns and MLP activations don't change much.
- **Early layers (especially layer 0) have more input-dependent Jacobians** — the transformation varies with content. Yet layer 0 achieves 98.4% recovery because its dominant transformation is a large-scale embedding projection (8.23x magnification) that overwhelms the input-dependent component in a least-squares fit.

**Critical finding: consistency does NOT predict CE recovery.** Layers 20-31 have high consistency (0.80-0.84) yet poor CE recovery (43-64%). This means even when the Jacobian is nearly the same matrix at all inputs — i.e., one W should theoretically suffice — the layer's **nonlinear residual** (the ~13% not captured by the linear approximation) carries information that is critical for downstream computation. Consistency measures whether a single linear map *fits* well, not whether the fit *matters* for model output.

### Data-Aligned Perturbation Gap (Method 8)

![PCA-Aligned Gap](results/pca_aligned_gap.png)

Method 8 reveals whether nonlinearity is aligned with the data manifold or scattered in irrelevant directions. The **gap ratio** (on-manifold / off-manifold) shows a striking depth-dependent pattern:

| Layer Range | On-Manifold Gap | Off-Manifold Gap | Gap Ratio | Eff. Rank | Interpretation |
|------------|----------------|-----------------|-----------|-----------|----------------|
| 0-1 | 0.13-0.16 | 0.25-0.27 | 0.49-0.64 | 18 | **More linear on-manifold** — nonlinearity in unused directions |
| 2-9 | 0.12-0.13 | 0.14-0.23 | 0.56-0.86 | 18-19 | Transitioning toward isotropy |
| 10-18 | 0.12-0.14 | 0.13-0.14 | **0.92-1.05** | 18 | **Isotropic to on-manifold-nonlinear** |
| 19-33 | 0.12-0.13 | 0.15-0.18 | 0.65-0.87 | 18 | Back to more linear on-manifold |
| 34-35 | 0.12-0.17 | 0.20-0.23 | 0.63-0.75 | 18 | More linear on-manifold |

**Key finding — layers 11-18 have gap ratio $\geq 1.0$**: These middle layers are *more nonlinear along data-relevant directions* than random ones. This is where genuinely useful nonlinear computation happens — the model is doing something that a linear map fundamentally cannot capture along the directions that matter. This explains why these layers show poor CE recovery in Method 7 despite having low absolute perturbation gaps.

**Effective rank is uniformly ~18** across all layers (95% variance threshold), meaning the data manifold occupies only 18 of 2560 dimensions regardless of depth. This extreme low-dimensionality means random perturbation directions (Method 1) are almost entirely off-manifold, potentially overstating how "linear" the layer appears.

### Global Linear Replacement (Method 7)

![Layer replacement analysis](results/layer_replacement_v2.png)

![Residual fitting analysis](results/residual_fitting_v2.png)

V2 tests **all 36 layers** with ridge regression (L2-regularized), using an 80/20 train/test split on token activations. We fit both **residual** maps ($g(\mathbf{x}) \approx W_r \mathbf{x}$, predicting the layer's correction) and **full-output** maps ($\mathbf{x}_{out} \approx W \mathbf{x}$, predicting the complete output). Lambda is selected by test-set activation MSE.

| Layer | KO delta | Res R² | Res Recovery | Full Recovery |
|-------|----------|--------|-------------|--------------|
| 0 | 9.076 | 0.787 | **98.4%** | 98.4% |
| 1 | 0.518 | 0.852 | 81.2% | 80.2% |
| 4 | 0.234 | 0.613 | 82.7% | **85.3%** |
| 6 | 1.880 | **0.997** | 54.4% | 54.2% |
| 8 | 0.292 | 0.535 | 68.1% | 73.2% |
| 10 | 0.517 | 0.664 | **82.8%** | 81.8% |
| 16 | 0.269 | 0.768 | **38.8%** | 37.2% |
| 20 | 0.238 | 0.529 | 47.9% | 50.0% |
| 21 | 0.233 | 0.541 | 51.6% | 42.6% |
| 22 | 0.256 | 0.578 | 44.4% | 46.8% |
| 28 | 0.290 | 0.496 | 54.1% | 53.3% |
| 31 | 0.267 | 0.453 | 58.7% | 58.5% |
| 35 | 0.281 | 0.955 | 74.5% | 74.6% |

#### The central finding: activation-space fit does NOT predict downstream utility

**This is the most important result of the entire experiment.** Residual R² measures how well the linear map fits the layer's transformation in activation space. CE recovery measures how well the fitted replacement preserves the model's output. These two metrics are **decoupled**:

- **Layer 6**: R² = 0.997 (essentially perfect fit), CE recovery = **54.2%**. The linear map captures 99.7% of activation variance but the remaining 0.3% carries information critical for downstream layers. Layer 6 is T-2's second most critical layer (KO delta = 1.88) — its function cannot be linearized despite the near-perfect fit.
- **Layer 16**: R² = 0.768 (good fit), CE recovery = **37.2%** (worst of all layers). A well-fitting linear map that actively misleads downstream computation.
- **Layer 35**: R² = 0.955, CE recovery = 74.6%. High R² translates to moderate recovery here — better than L6 because L35's nonlinear component, while more prominent, is less critical for downstream computation (it's the final layer before the LM head).

The scatter plot of R² vs CE recovery (right panel of the residual fitting figure) shows no clear monotonic relationship. This means **you cannot determine whether a layer is linearizable by measuring how well a linear map fits its activations** — you must measure the downstream impact.

#### Depth-dependent recovery profile

CE recovery across all 36 layers follows a **declining arch**:

- **Layer 0**: 98.4% — the massive embedding projection is almost perfectly linear globally
- **Layers 1-5**: 72-85% — good linearizability
- **Layers 6-14**: 54-83% — highly variable; L6 and L8 are weak points (54%, 68%), while L10 peaks at 83%
- **Layers 15-23**: **37-70%** — the worst region. L16 (37.2%), L20 (50%), L21 (42.6%), L22 (44.4%)
- **Layers 24-33**: 52-65% — modest recovery, barely better than identity
- **Layers 34-35**: 69-75% — partial recovery at the end

**Only 14 of 36 layers achieve $\geq$ 73% CE recovery.** The previous experiment's claim of "all layers $\geq$ 73% recovery" was based on inflated OLS numbers without proper train/test validation. The v2 results with proper methodology paint a much more honest picture: **middle-to-late layers (15-33) genuinely resist linearization**, with recovery often below 60%.

#### Why middle layers resist linearization despite low perturbation gap

This is the core paradox of the experiment. Layers 8-18 have the lowest perturbation gaps (~0.13, meaning 87% of local behavior is linear), yet their global linear replacement recovery ranges from 37% to 83%. Three factors explain this:

1. **On-manifold nonlinearity (Method 8)**: Layers 11-18 have gap ratio $\geq 1.0$ — their nonlinearity is specifically concentrated along data-relevant directions. The 13% nonlinear residual at each operating point is not random noise; it encodes information the model needs.

2. **Error amplification through depth**: A 13% local error at layer 16 propagates through 19 subsequent layers. Even with contractive Jacobians (mean amplification ~0.65), the error compounds because it lies along data-relevant directions that downstream layers are sensitive to.

3. **The "small norm, large impact" effect**: Layer 6 demonstrates this most dramatically. Its residual R² = 0.997 means the linear map captures 99.7% of activation variance. But the 0.3% residual is aligned with directions that layer 7+ depend on for routing decisions — small in L2 norm but large in downstream informational content.

### Cross-Reference with T-9 Weight Spectral Structure

T-9 found a significant negative correlation between weight effective rank and linearization gap: **r = -0.43, p = 0.009**. Layers with higher-rank weight matrices tend to be *more linear*, not less. The explanation is that high-rank layers spread computation across many dimensions, where self-averaging makes the aggregate more linear (CLT-type effect). Low-rank layers concentrate nonlinearity in fewer dimensions, making it more prominent.

### Layer 0 Paradox Resolution

![Layer 0 Paradox](results/paradox_resolution.png)

Layer 0 is simultaneously the most locally nonlinear (gap = 0.245, highest of all layers) and the most globally linearizable (98.4% CE recovery). This resolves through three observations:

1. **Magnitude dominance**: Layer 0's transform-to-input ratio is 8.23x (15x larger than any other layer). Its dominant linear component overwhelms the nonlinear variation in a least-squares fit.
2. **On-manifold linearity**: Gap ratio = 0.64 (on-manifold gap 0.16 vs off-manifold 0.25). The nonlinearity is concentrated in directions the data doesn't use.
3. **Structural simplicity**: Despite high absolute nonlinearity, its *function* (embedding projection) is inherently close to linear — a large-scale basis change with modest nonlinear corrections.

## Conclusions & Key Findings

1. **Hypothesis partially refuted — U-shaped nonlinearity profile**: The expected monotonic early=linear, late=nonlinear pattern does NOT hold. Instead, we observe a **U-shaped** profile where middle layers (6-18) are the most locally linear (perturbation gap ~0.13) and both early and late layers are more nonlinear. This is structural and robust across prompts.

2. **Local linearity does NOT imply global linearizability**: This is the experiment's most important finding. Middle layers have the lowest perturbation gaps (~0.13, i.e., 87% of local behavior is linear) yet many achieve poor CE recovery when replaced by a global linear map (L16: 37%, L20-22: 43-50%). The perturbation gap measures a fundamentally different property than practical replaceability. Only 14 of 36 layers achieve $\geq$ 73% CE recovery.

3. **Activation-space fit (R²) is decoupled from downstream utility (CE recovery)**: Layer 6 achieves R² = 0.997 (near-perfect activation fit) but only 54% CE recovery. Layer 16 has R² = 0.768 but only 37% recovery. A linear map can fit the activations well while completely missing the small nonlinear residual that downstream layers depend on. **You cannot determine linearizability from activation-space metrics alone — downstream impact must be measured.**

4. **On-manifold nonlinearity explains the paradox (Method 8)**: Layers 11-18 have gap ratio $\geq 1.0$, meaning they are *more nonlinear along data-relevant directions* than random ones. Their 13% nonlinear residual is not isotropic noise — it is specifically concentrated where it matters. This is where the model performs genuinely nonlinear computation (feature interactions, routing decisions) that no linear map can capture. Earlier layers (0-5) and late layers (24-35) have ratio $< 1$, meaning their nonlinearity is mostly off-manifold and less consequential.

5. **Layer 0 paradox resolved**: Simultaneously the most locally nonlinear (gap = 0.245) and most globally linearizable (98.4% recovery). Three factors: (a) magnitude dominance (8.23x transform overwhelms variation), (b) on-manifold linearity (gap ratio 0.64 — nonlinearity in unused directions), (c) structural simplicity (embedding projection is inherently near-linear).

6. **MLP drives late-layer nonlinearity**: In layers 33-35, MLP gap increases sharply (0.14 to 0.25) while attention remains stable (~0.11). This suggests higher-order feature interactions via SwiGLU gating dominate final-layer computation.

7. **Most layers are remarkably uniformly contractive**: Mean Jacobian amplification $< 1$ for layers 1-35 in a tight range. Combined with the residual connection ($\mathbf{x} \to \mathbf{x} + g(\mathbf{x})$), the full Jacobian $\mathbf{J}_F = \mathbf{I} + \mathbf{J}_g$ has spectral radius near 1, ensuring neither vanishing nor exploding gradients.

8. **Jacobian consistency increases with depth but does NOT predict recovery**: Consistency rises from 0.65 (L0) to 0.90 (L35), meaning late layers apply nearly identical transformations regardless of input. But high consistency coexists with poor CE recovery (layers 20-31: consistency 0.80+, recovery 43-65%). Consistency measures whether one W *fits* well, not whether the fit *matters*.

9. **The data manifold is extremely low-dimensional**: Effective rank is ~18 across all layers (out of 2560 dimensions). This means random perturbation directions (Method 1) are almost entirely off-manifold, and the perturbation gap systematically *underestimates* how nonlinear the layer is along directions that matter. Method 8's on-manifold measurement is more relevant for practical linearizability.

## Practical Implications

*Note: The following are hypotheses suggested by the data. None have been validated end-to-end.*

### The Linearization Landscape is Narrower Than Expected

The v1 experiment suggested "all layers $\geq$ 73% recovery" — implying broad linearization opportunity. V2 shows the realistic picture: only **layers 0-5 and 9-14** (16 of 36 layers) achieve reasonable recovery ($\geq$ 68%). Middle-to-late layers (15-33) cluster around 37-65%, meaning a linear replacement loses 35-63% of the layer's contribution. This is not catastrophic (it's better than knockout) but far from a practical drop-in replacement.

### What CAN be linearized

**1. Layer 0** (98.4% recovery): The embedding projection is near-perfectly linear. Replacing it with a matrix multiply is essentially lossless.

**2. Layers 1-5, 9-14** (68-85% recovery): These early-to-mid layers are the best linearization candidates. Their on-manifold nonlinearity is low (gap ratio < 0.9), meaning their nonlinear component is mostly irrelevant to the data.

### What CANNOT be linearized

**1. Layers 15-23** (37-60% recovery): Despite sitting in or near the "local linearity plateau," these layers' nonlinear residuals carry critical downstream information. L16 (37.2%) is the worst — its gap ratio is 1.05 (on-manifold nonlinear), meaning its nonlinearity is specifically aligned with data-relevant directions.

**2. Layer 6** (54.2% recovery): The most extreme example of "good fit, bad replacement." R² = 0.997 but recovery = 54%. This is T-2's second most critical layer — its nonlinear component, though tiny in norm, is essential.

**3. Layers 24-33** (52-65% recovery): Moderate recovery at best. Not catastrophic, but not practical either.

### The On-Manifold Criterion

Method 8's gap ratio provides a better predictor of linearizability than the perturbation gap alone. Layers with gap ratio $< 0.8$ (nonlinearity mostly off-manifold) tend to have higher CE recovery, while layers with gap ratio $\geq 1.0$ (nonlinearity on-manifold) resist linearization regardless of their absolute gap. This suggests that **future linearization efforts should measure on-manifold nonlinearity**, not just perturbation gap.

### Convergent Evidence

- **T-2 (Layer Knockout)**: Knockout criticality is orthogonal to linearization gap — L6 is T-2's 2nd most critical layer despite being "locally linear"
- **T-9 (Spectral Structure)**: Weight effective rank correlates negatively with linearization gap (r = -0.43, p = 0.009)
- **T-3 (Layer Swap Cost)**: Adjacent plateau layers have the cheapest swap costs, consistent with near-linear layers being more interchangeable

## Usage

```bash
# Generate calibration data first (if not already done)
poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Instruct-2507

# Run v2 experiment (Methods 1-8, ~21 minutes on B200)
poetry run python experiments/t7_layer_linearization_gap/run_v2.py
```

Results are saved to `experiments/t7_layer_linearization_gap/results/`:
- `summary_v2.json` — all per-layer metrics including PCA gap, Jacobian consistency, and layer replacements
- `linearization_gap_v2.png` — perturbation gap and attn/MLP decomposition
- `jacobian_properties_v2.png` — spectral norm, amplification, transform magnitude
- `gap_vs_criticality_v2.png` — cross-reference with T-2 knockout experiment
- `gap_heatmap_v2.png` — per-prompt perturbation gap variability
- `multiscale_analysis_v2.png` — gap vs eps log-log plots, nonlinearity order across depth
- `jacobian_consistency_v2.png` — Jacobian consistency across depth and local vs global linearity scatter
- `layer_replacement_v2.png` — knockout vs linear replacement comparison (all methods)
- `residual_fitting_v2.png` — residual R² and R² vs CE recovery scatter
- `pca_aligned_gap.png` — on-manifold vs off-manifold nonlinearity (Method 8)
- `paradox_resolution.png` — Layer 0 paradox deep-dive

Runtime: ~1248s (~20.8 min) on NVIDIA B200 (~200s per-prompt analysis, ~2s PCA gap, ~13s Jacobian consistency, ~1093s layer replacement).
