# X / Twitter Thread

---

**Tweet 1**

Someone (us 👀) took Qwen3-4B apart. SVD of all 252 weight matrices, Jacobians at every layer, thousands of tokens tracked through the network.

Halfway through, the 2560-dimensional hidden state collapses to 2.3 effective dimensions. One axis carries 67% of all variance. Everything the model knows gets squeezed through a near-singular pinhole.

The textbook says "gradual refinement across layers." The data says something else entirely.

---

**Tweet 2**

The model goes through six geometric phases:

Layer 0 erases the input embedding (cosine similarity with input drops to 0.11) and doubles dimensionality. Layers 1-5 crush it down to 6 effective dimensions. Layers 6-15 expand back to 200, this is where the real distributed processing happens.

Then layers 16-24 squeeze everything through that 2.3-dimension bottleneck again. A single axis explains two-thirds of all token variation.

Whatever survives this pinhole is all the remaining 19 layers get to work with.

---

**Tweet 3**

Here's the result that broke my assumptions.

They fit a linear replacement to layer 6. R² = 0.997. The map captures 99.7% of activation variance.

Plugged it back into the full model: 54% CE recovery. The 0.3% the linear map missed was responsible for nearly half of downstream loss degradation.

Knock layer 6 out entirely and loss increases 21.7x. It's the second most critical layer in the model. The (5, 6) pair knockout: 63.7x loss increase, far exceeding the sum of individual effects.

You cannot tell if a layer is safely replaceable from activation metrics.

---

**Tweet 4**

The weight spectral structure tells you what each matrix is actually doing:

Q projections use 25% of capacity (vs 62% Marchenko-Pastur baseline for a random matrix of the same shape). K: 38%. V: 61%. MLP gate/up/down: 50-68%.

Training compresses routing far more aggressively than content processing. Q is at 40% of random-matrix capacity. The "where to attend" computation has been squeezed into a much smaller subspace than random initialization would occupy.

gate_proj at 3-bit: ~58x more sensitive than Q or K. SiLU + multiplicative interaction means small weight errors compound nonlinearly.

---

**Tweet 5**

Layer 35 does something no other layer does.

Layers 18-34 all push in the same direction as the accumulated residual. Positive cosine alignment up to +0.44. Norms grow superlinearly from 139 to 571. Every token converges into a tight directional cone — 0.63 mean cosine similarity. They're nearly indistinguishable.

Then layer 35 arrives. Cosine alignment with the residual: -0.73. It fires backwards.

After: norms drop from 571 to 388, cosine similarity drops from 0.63 to 0.09. The cone shatters. Now the LM head can tell tokens apart.

Remove this layer and prediction fails.

---

**Tweet 6**

This connects to Anthropic's superposition work. The residual stream encodes features as nearly-orthogonal directions — far more features than dimensions. Abliteration showed that refusal in Llama 3 lives on a single direction. One rank-1 spectral modification kills it.

At the bottleneck, PR = 2.3 but hundreds of features still need to remain distinguishable. Signal-to-noise on minor axes collapses. This is where superposition is most fragile.

That's why the linear map fails at 0.997 R²: it captures the dominant axes perfectly but smears the low-variance directions where individual feature distinctions live. Including safety features.

---

**Tweet 7**

Engineering takeaways worth pinning:

4-bit quantization is lossless per-layer (<0.1 PPL). The cliff is 3-bit to 2-bit. Layer 2 alone at 2-bit: +3,828 perplexity. A spectral-informed mixed-precision scheme that seemed smart caused a 13,000x PPL increase — it put 2-bit on early layers because they looked "simple." They're simple AND critical. Errors at layer 2 propagate through 34 subsequent layers.

LoRA: skip layers 16-24 (the bottleneck). Features are packed at maximum density there, new adaptation directions corrupt existing ones. Phase-aware allocation hit 80.5% OOD vs 78.0% uniform. Adapting just L25-35 (10M params) matched full 33M-param LoRA.

---

**Tweet 8**

What actually happens inside a transformer:

Destroy the embedding. Compress to 6 dimensions. Expand and process. Compress again to 2.3 dimensions. Build an anisotropic cannon. Fire backwards.

Not a smooth pipeline. Not gradual refinement. A sequence of radical geometric transformations separated by information bottlenecks where the model is most vulnerable.

We spend a lot of time at @WhiteCircle thinking about what's load-bearing inside these models. When you're deploying LLMs into finance, healthcare, security, understanding where the fragile channels are isn't optional. It's the whole game.

Research by @hivaze. Reproduction scripts: github.com/hivaze/dl-experiments

---
