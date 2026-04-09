# Transformers Don't Compute - They Compress, Explode, and Self-Destruct

*A layer-by-layer dissection of what actually happens inside a 4B-parameter transformer, and why it matters for building better AI systems.*

---

Everyone has the same mental model of a transformer: tokens go in, layers refine them step by step, predictions come out. Each layer does a bit of attention, a bit of processing, and after 36 of these gentle refinements you get your answer. Clean, symmetric, intuitive.

It's also missing most of the picture.

We took Qwen3-4B, a 4-billion parameter transformer, and dissected it. SVD of every weight matrix, Jacobians at every layer, tracking thousands of tokens through the network, stress-testing with quantization down to 2 bits. Four experiments, 252 weight matrices, one goal: find out what's actually going on inside.

What we found looks nothing like the textbook.

---

## The Model

[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) is a 4-billion parameter decoder-only transformer. 36 layers, hidden dimension of 2560, Grouped Query Attention, SwiGLU MLP, RoPE embeddings. Every layer is architecturally identical. Same structure, same dimensions, same parameter count (~101M each).

Nothing in the blueprint distinguishes layer 0 from layer 35. The learned weights tell a very different story.

---

## Part I: The Six Lives of a Hidden State

The standard picture says representations are gradually refined across layers. The data says the residual stream goes through six distinct geometric phases, including two near-singular bottlenecks where information gets squeezed down to a handful of dimensions.

We used Singular Value Decomposition (SVD) and a metric called Participation Ratio (PR) to measure how many dimensions the model actually uses at each layer. PR tells you the effective number of active dimensions, if all information sits on one axis, PR = 1. If it's spread across k axes equally, PR = k. No arbitrary cutoffs needed.

The hidden dimension is 2560. Here's how many dimensions the model *actually uses*:

| Phase | Layers | Effective Dimensions (PR) | What Happens |
|-------|--------|--------------------------|--------------|
| Expansion | 0 | 73 → 125 | Layer 0 *erases* the input embedding and doubles dimensionality |
| First Compression | 1–5 | 6–43 | Representations collapse to as few as 6 effective dimensions |
| Distributed Processing | 6–15 | 147–205 | Recovery to high dimensionality - the real work happens here |
| Second Compression | 16–24 | 2.3–17 | The Keyhole - PR drops to 2.3. One axis explains 67% of all variance |
| Output Preparation | 25–34 | 24–127 | Norms explode from 139 to 571; all tokens point the same direction |
| Dispersal | 35 | 160 | Layer 35 *actively opposes* everything before it, breaking the pattern |

This is not gradual refinement. It's destroy → compress → expand → compress → cannon → fire backwards.

Out of 2560 available dimensions, the model squeezes everything down to 2.3 at layer 16. Whatever survives this bottleneck is all the remaining 19 layers have to work with. If you prune, fine-tune, or compress the wrong layers, you corrupt what passes through this pinhole and the whole model degrades.

[![The Dimensionality Bottleneck](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig3_bottleneck.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig3_bottleneck.png)
*Figure 1. Left: Participation ratio across depth, with two bottleneck regions highlighted in red. At layer 16, PR = 2.3, nearly one-dimensional. Right: each layer's signed contribution to the final representation.*

### The embedding is erased by layer 5

The cosine similarity between each layer's output and the original embedding drops to essentially zero by layer 5. All directional information in the later residual stream comes from the transformer's own updates, not from the embedding table. The embedding provides initial material, but its geometric identity is completely overwritten.

### Layer 35: The dispersal mechanism

The final layer does something no other layer does. Every layer from 18 to 34 reinforces the residual, their updates point in roughly the same direction as the accumulated representation. Then layer 35 arrives and actively opposes it with a cosine alignment of −0.73.

The result: norms drop from 571 to 388, and mean cosine similarity between tokens plummets from 0.63 to 0.09. Before this layer, tokens are packed in a tight directional cone and are nearly indistinguishable. After it, they're spread out and the output head can finally tell them apart.

Skip this layer and tokens stay at 0.63 cosine similarity - the model cannot discriminate between them. Layer 35 is not refining. It's creating the separation the output head needs to function. Any architecture change that shares, skips, or compresses the final layer needs an alternative dispersal mechanism, or predictions become mush.

[![The Dispersal Mechanism](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig6_self_destruct.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig6_self_destruct.png)
*Figure 2. Layer 35 is unique. Left: mean cosine similarity between tokens drops from 0.63 to 0.09. Center: norms drop despite the largest update in the model. Right: update-residual alignment is strongly negative - the only layer that pushes against everything before it.*

### Where predictions actually come from

Projecting intermediate hidden states through the output head (logit lens) shows that top-1 accuracy is near zero through layer 21, then ramps from 11% to 100% between layers 22 and 35. The first half of the network builds scaffolding. The actual prediction is assembled in the last third and reshaped by layer 35's dispersal.

[![Logit Lens + Bottleneck](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig9_logit_lens_overlay.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig9_logit_lens_overlay.png)
*Figure 3. Logit lens accuracy (solid cyan) vs participation ratio (dashed blue). Predictions form entirely in the post-bottleneck layers (22–35).*

---

## Part II: The Linearization Paradox

This is where it gets uncomfortable.

We tested whether each layer could be replaced by a learned linear map. For each of the 36 layers, we collected input-output pairs, fit a ridge-regression replacement, and measured two things: how well it fits the activations (R²) and how well it actually works when plugged into the model (CE recovery).

The assumption was that good fit means good replacement. It's spectacularly wrong.

| Layer | R² (Activation Fit) | CE Recovery (Actual Quality) | What's Going On |
|-------|---------------------|------------------------------|-----------------|
| Layer 6 | 0.997 (99.7% fit) | 54% | The 0.3% the linear map misses = nearly half the downstream loss |
| Layer 16 | 0.768 | 37% (worst) | The map actively misleads downstream computation |
| Layer 0 | 0.787 | 98% | Lower fit, but near-perfect replacement - dominant embedding projection overwhelms the nonlinear variation |

Only 15 of 36 layers achieve ≥73% CE recovery. The middle-to-late layers (16–33) resist linearization despite appearing locally smooth.

R² = 0.997 sounds like a perfect fit. But the 0.3% it misses is where nearly half the model's knowledge lives. You cannot determine whether a layer is replaceable from activation-space metrics alone. If you're pruning or distilling because "this layer looks linear, so it's safe to approximate," you're probably breaking things that won't surface until end-to-end evaluation. Always measure end-to-end.

[![The Linearization Paradox](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig2_paradox.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig2_paradox.png)
*Figure 4. Each dot is one layer. X-axis: how well a linear map fits its activations. Y-axis: how well that map actually works when plugged into the model. The expected diagonal relationship doesn't hold.*

### Nonlinearity follows a U-shape

We expected early layers to be linear (simple token mixing) and late layers to be nonlinear (complex composition). Wrong. Nonlinearity is highest at both ends (layers 0–1 and 33–35) and lowest in the middle (layers 8–18, ~87% linear). The most nonlinear single component is layer 35's MLP, the dispersal mechanism requires hard nonlinear computation.

### Why "locally linear" doesn't mean "globally replaceable"

The hidden state manifold has an effective dimensionality of only ~18 (out of 2560). When we measure nonlinearity along actual data directions instead of random ones, layers 11 and 15–18 are more nonlinear along data-relevant directions than along random ones. The nonlinear computation is concentrated precisely where the data lives.

Error amplification through depth makes it worse: a 13% error at layer 16 propagates through 19 subsequent layers, along directions that downstream layers are tuned to be sensitive to. Small in norm. Large in informational content.

[![PCA-Aligned Gap](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig_pca_gap.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig_pca_gap.png)
*Figure 5. Left: on-manifold vs off-manifold nonlinearity. Center: their ratio - values above 1.0 mean the model is more nonlinear along data-relevant directions. Right: the data manifold is only ~18-dimensional everywhere.*

---

## Part III: Routing Is Cheap, Thinking Is Expensive

The third experiment looks at the layers' weight matrices themselves. Every layer has 7 weight matrices. We computed SVD on all 252 and measured effective rank - how much of their capacity each matrix actually uses.

There's a clean hierarchy:

| Matrix | Capacity Used | Role |
|--------|--------------|------|
| Q (query) | 25% | "Where should I look?" - low-dimensional address lookup |
| K (key) | 38% | "What's at each position?" - slightly richer addressing |
| V (value) | 61% | "What should I extract?" - high-dimensional content |
| O (output) | 42% | Reassembling multi-head outputs |
| MLP | 50–68% | The heavyweight processing - uses the most capacity |

Q/K routing is low-rank in every single layer (36/36). "Where to attend" is a much simpler computation than "what to extract."

Q/K matrices only use 25–38% of their theoretical capacity, which means massive built-in redundancy for compression. They tolerate aggressive quantization. MLP gate projections, on the other hand, use much more of their capacity and are ~58x more sensitive to compression at 3-bit. If you're quantizing a model, the attention routing matrices are safe to squeeze. The MLP gate is where things break.

[![Routing vs Content](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig4_routing_vs_content.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig4_routing_vs_content.png)
*Figure 6. Left: effective rank ratio across depth for three functional groups. Right: average capacity usage by matrix type.*

### The counterintuitive correlation: higher rank = more linear

Naive expectation: higher rank means more complex, means more nonlinear. The data says the opposite. r = −0.43, p = 0.009. Higher-rank layers are significantly more linear. A plausible explanation: a high-rank layer spreads computation across many dimensions. Each applies a nonlinearity, but many small nonlinear contributions tend to average out. Low-rank layers concentrate computation in fewer dimensions where each nonlinear channel dominates.

This correlation is entirely MLP-driven (up_proj r = −0.74). Attention matrices show r ≈ 0. The SwiGLU MLP is where the rank-linearity relationship lives; softmax attention behaves independently of weight rank.

[![Higher Rank = More Linear](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig5_rank_vs_linearity.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig5_rank_vs_linearity.png)
*Figure 7. Each layer plotted by mean weight effective rank vs perturbation gap. Negative correlation: higher-rank layers are more linear, not less.*

---

## Part IV: Everything Converges

The four experiments weren't designed to tell a unified story, but they do.

### The bottleneck layers are non-linearizable

Layers 16–24 have the lowest dimensionality (PR = 2.3–17), sit in the "linear plateau" of the perturbation gap (~87% linear), and have moderate weight rank. They look like prime candidates for linearization. They achieve the worst CE recovery (37–65%).

Why? Squeezing through a near-one-dimensional bottleneck means the tiny nonlinear residual encodes all the distinguishing information. When representations are compressed to 2–3 effective dimensions, two different inputs land on nearly the same axis. The small nonlinear correction the linear map misses is exactly what separates them.

### Quantization confirms the phase structure independently

Instead of replacing layers with linear maps, we quantized each layer's weights to 2-bit precision. The pattern matches perfectly: layers 0–3 are catastrophically sensitive (quantizing layer 2 alone increases perplexity by 3,828), while the entire mid-depth range (layers 8–20) absorbs 2-bit quantization with less than 1 PPL impact.

The linearity gap correlates with quantization sensitivity at ρ = 0.71 (p < 0.0001). Layers that are more nonlinear are also harder to quantize.

The model is not 36 interchangeable layers. It's three functional modules with distinct geometric roles: an early processing block, a mid-depth distributed processing zone, and a late output preparation zone, with L0 and L35 doing something entirely their own at each end. Quantization, LoRA, pruning, distillation - they all need to respect this structure.

[![Linearity Gap Predicts Quantization Sensitivity](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig10_quant_sensitivity.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig10_quant_sensitivity.png)
*Figure 8. Linearity gap (red) and 2-bit quantization sensitivity (blue bars, log scale) track each other across depth. Early layers are catastrophically sensitive. Mid-layers absorb even 2-bit quantization.*

[![Cross-Experiment Alignment](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig8_cross_experiment.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig8_cross_experiment.png)
*Figure 9. Four metrics across all 36 layers with shared phase annotations. Phase boundaries identified independently in each experiment align.*

[![Update Correlation Matrix](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig_update_correlation.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig_update_correlation.png)
*Figure 10. Cosine similarity between mean layer updates. Clear block structure. Layer 35 is anti-correlated with everything.*

---

## Part V: Where Geometry Meets Interpretability

Our experiments measure transformer internals from the outside - geometry, spectral structure, stress-testing. A parallel line of research from Anthropic and others approaches the same layers from the inside, tracing individual features and circuits. The two perspectives line up consistently.

### The bottleneck as a superposition chokepoint

The residual stream represents concepts as sparse linear directions, many more features than dimensions, encoded as nearly-orthogonal directions (superposition). Research on abliteration has shown that safety-trained refusal in Llama 3 is mediated by a single direction - orthogonalize it away and the behavior disappears. One direction, one behavior.

At the bottleneck, variance concentrates onto 2–3 dominant axes. The representations still live in 2,560 dimensions - other directions still carry information, but the signal-to-noise ratio on minor axes drops dramatically. If features are individual directions and you can lose an entire behavioral mode by corrupting one of them, then the bottleneck is where superposition is most fragile.

This explains the linearization paradox: a linear map captures dominant variance axes perfectly but smears the low-variance directions where feature distinctions live. It also explains why the bottleneck works best frozen during LoRA: adding new adaptation directions into a space at maximum feature density risks overwriting the precise angular relationships superposition depends on.

### The linearization gap from both sides

Anthropic's circuit tracing work linearizes the transformer by freezing attention patterns and normalization, replacing MLPs with sparse linear features. Their result: ~50% next-token prediction match. Our per-layer replacements tell the same story, roughly half of model behavior is captured by linear approximation. The other half is where the interesting computation lives.

Why does linearization partially work? Because most inter-feature interactions in the residual stream are linear - features combine via addition. The nonlinearity is concentrated in MLP activations (SwiGLU), attention patterns (softmax), and normalization (RMSNorm).

The geometric measurements and the mechanistic interpretability research independently converge on the same picture. The residual stream carries meaning as precise directional relationships. The bottleneck is where those relationships are most fragile. Understanding this geometry tells you exactly where compression, fine-tuning, and architectural modifications will succeed or fail.

---

## Part VI: Practical Implications

The geometry tells you where to intervene. The interpretability tells you why. Here's what it means for anyone building, deploying, or optimizing models.

### 1. Don't trust activation-space metrics for pruning

R² between original and replacement activations tells you almost nothing about downstream impact. Layer 6's 0.3% residual corresponds to nearly half the downstream loss degradation. The only reliable metric is end-to-end evaluation. This applies to LoRA, distillation, quantization, and pruning equally.

A layer that "looks replaceable" by activation metrics can be hiding half the model's knowledge in the noise floor. Always measure end-to-end.

### 2. The dispersal layer is load-bearing

Any architecture change that shares, skips, or compresses the final layer needs an alternative dispersal mechanism. Without it, the output head receives representations where tokens are nearly indistinguishable (0.63 cosine similarity). This extends to early-exit architectures: exiting before the last layer skips the dispersal entirely.

### 3. LoRA should follow the spectral structure

Standard practice: same LoRA rank everywhere. The spectral data says this wastes parameters.

Q/K in early/mid layers (0–16) need rank 8–16 at most. Already at 23–36% effective rank, adding parameters here buys nothing. Gate/MLP in late layers (17–35) need rank 32–64. Gate_proj jumps from 0.41 to 0.58 effective rank in late layers - uniform rank-16 under-fits.

Testing confirmed this: a phase-aware LoRA allocation achieved the highest out-of-distribution accuracy (80.5% vs 78.0% for uniform rank-16). Adapting only the last 11 layers (10M params) matched all-layer LoRA (33M params) on both benchmarks.

Stop applying the same LoRA rank to every layer. Spend your parameter budget where the model has capacity to use it - late-layer MLP matrices. And if you're fine-tuning a model that's already competent at your target task, drop the learning rate to 5e-6 (instead of the default 2e-4) before concluding LoRA doesn't help.

[![Non-Uniform LoRA Guide](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig7_lora_guide.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig7_lora_guide.png)
*Figure 11. Effective rank ratio by matrix type, split by depth region. Gate projection shows the largest plateau-to-late gap, motivating depth-adaptive rank allocation.*

### 4. The bottleneck works best frozen

Skipping the bottleneck layers during LoRA (adapting layers 0–14 + 25–35 only) achieved the highest in-distribution accuracy (93.0% GSM8K) while perfectly preserving out-of-distribution capability. The bottleneck functions as a frozen information channel, features packed at maximum density, no room for new LoRA directions without disrupting existing ones.

For efficient fine-tuning: skip the bottleneck layers (16–24). Concentrate LoRA on the distributed processing zone (6–15) and output preparation zone (25–35).

### 5. Quantize everything - except the gate and the endpoints

The SwiGLU gate projection is the quantization Achilles' heel - ~58x more sensitive than attention routing matrices at 3-bit. Q/K matrices are nearly immune thanks to built-in redundancy from low effective rank.

The practical recipe: 4-bit quantization is essentially lossless for single layers (< 0.1 PPL impact). The cliff appears between 3-bit and 2-bit, concentrated in the first 5 layers and gate projections. Giving the final layer one extra bit reduces quality penalty by 27%.

Sophisticated mixed-precision schemes guided by spectral metrics fail catastrophically compared to simple heuristics. They confuse "low rank" with "safe to compress." Early layers have the lowest spectral rank and the highest quantization sensitivity.

Use NF4 quantization (PPL within 0.1 of full precision). If you need fine control, protect early layers (0–3) and layer 35 with extra bits. Don't bother with per-matrix mixed precision, it adds complexity without benefit.

[![Per-Matrix Quantization Sensitivity](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig11_quant_matrix.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig11_quant_matrix.png)
*Figure 12. Left: Mean PPL impact of 3-bit quantization per matrix type. gate_proj is ~58x more sensitive than attention routing matrices. Right: gate_proj sensitivity by layer.*

[![Mixed-Precision Recipes](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig12_mixed_precision.png)](https://github.com/hivaze/dl-experiments/raw/master/blog/images/fig12_mixed_precision.png)
*Figure 13. All recipes target ~4-bit average. The spectral-informed recipe fails catastrophically because it assigns 2-bit to early layers. Giving layer 35 one extra bit beats every sophisticated strategy.*

---

## Conclusion

The standard mental model, 36 identical layers, each refining a little - needs an update. What actually happens:

1. Destroy the input embedding and expand into a working space
2. Compress into an initial low-dimensional passage
3. Expand and do distributed, high-dimensional processing
4. Compress again through the bottleneck - variance collapses to 2–3 axes
5. Build an anisotropic cannon (norms growing superlinearly, all tokens pointing the same way)
6. Fire backwards - actively oppose everything the previous 17 layers built, creating the separation the output head needs

The architecture that emerges from training isn't a smooth pipeline. It's a sequence of radical geometric transformations separated by low-dimensional bottlenecks. And somewhere in the 0.3% of variance that your linear approximation misses, the model is hiding almost half of what it knows.

Transformers are not 36 identical refinement steps. They're three functional modules separated by information bottlenecks, bookended by a destructive first layer and a dispersal last layer. Every practical decision, quantization, LoRA, pruning, distillation, early-exit, needs to respect this phase structure or risk corrupting the narrow channels where actual computation happens.

---

*All experiments conducted on a single NVIDIA B200 GPU on Qwen3-4B-Instruct-2507. Reproduction scripts and full results at [github.com/hivaze/dl-experiments](https://github.com/hivaze/dl-experiments).*

*Research by [hivaze](https://github.com/hivaze). Accessible rewrite by White Circle.*
