# X / Twitter Thread

---

**Tweet 1 (Hook)**

Someone dissected a 4B-parameter transformer layer by layer and what they found is wild.

Your 2560-dimensional hidden state? Collapses to 2.3 effective dimensions halfway through. One axis = 67% of all variance.

The textbook picture of "gradual refinement" is wrong. Thread 🧵

---

**Tweet 2**

They tried replacing a layer with its best linear approximation.

R² = 0.997. Sounds perfect right?

Plugged it back in → only recovered 54% of model quality.

The 0.3% the linear map missed? Nearly half the model's knowledge lives there.

TLDR: you cannot tell if a layer is replaceable by looking at activation metrics. Full stop.

---

**Tweet 3**

Attention routing (Q/K) matrices only use 25-38% of their capacity. MLP gate projections are ~58x more sensitive to compression.

"Where to attend" = simple computation.
"What to extract" = where the real work happens.

TLDR: Q/K tolerates aggressive quantization. The MLP gate is where things break.

---

**Tweet 4**

The final layer actively OPPOSES everything the previous 17 layers built. Cosine alignment of −0.73.

Before it: tokens at 0.63 cosine similarity (indistinguishable).
After it: 0.09 (separable).

Skip this layer and the model literally can't tell its outputs apart.

---

**Tweet 5**

Practical takeaways:

→ 4-bit quantization is essentially lossless per layer
→ The cliff is between 3-bit and 2-bit, concentrated in first 5 layers + gate projections
→ LoRA: skip the bottleneck layers (16-24), concentrate on layers 6-15 + 25-35
→ Adapting just the last 11 layers (10M params) matched all-layer LoRA (33M params)

---

**Tweet 6**

TLDR of TLDRs: Transformers are not 36 identical refinement steps. They're 3 functional modules separated by information bottlenecks + a destructive first layer + a dispersal last layer.

Quantization, LoRA, pruning, distillation — all need to respect this structure.

This is the kind of depth that drives how we think about AI safeguards and model deployment @ White Circle.

Full research + reproduction scripts: github.com/hivaze/dl-experiments
Accessible rewrite on our blog.
