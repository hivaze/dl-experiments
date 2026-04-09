# LinkedIn Post

"The 0.3% of variance your linear approximation misses? That's where half the model's knowledge lives."
— hivaze, dissecting Qwen3-4B

Alex and the Research Team have just dropped their latest blog post and it's one of those posts where you read it and go - yeah, this is what we're building towards at White Circle. Understanding what's actually happening inside these models. Not the textbook version. The real version.

The paper is a full dissection of a 4B-parameter transformer. Four experiments. 252 weight matrices. What they found:

1. Halfway through the network, your 2560-dimensional hidden state collapses to 2.3 effective dimensions. A single axis explains two-thirds of all variance. The model is passing everything through a near-singular pinhole, and whatever survives is all the remaining layers get to work with.

2. They fitted a linear replacement to one layer and got R² = 0.997. Sounds perfect. Plugged it back in → recovered only 54% of model quality. The 0.3% the linear map missed was responsible for nearly half the downstream loss. 

3. Attention routing matrices (Q/K) only use 25–38% of their capacity. MLP gate projections are ~58x more sensitive to compression. "Where to attend" is a simple computation. "What to extract" is where the real work happens.

4. The final layer actively opposes everything the previous 17 layers built, cosine alignment of −0.73 with the accumulated representation. Without this dispersal, tokens are indistinguishable (0.63 cosine similarity). Skip it and the model can't tell outputs apart.

TLDR — why does this matter? Transformers are not 36 identical refinement steps. They're three functional modules separated by information bottlenecks, bookended by a destructive first layer and a dispersal last layer. Every practical decision — quantization, LoRA, pruning, distillation, needs to respect this phase structure.

This is the kind of depth that drives how we think about AI safeguards and model deployment at White Circle. If you're compressing, fine-tuning, or deploying models without understanding what's actually load-bearing inside them, you're flying blind.

Full accessible rewrite on our blog. Original research and reproduction scripts by hivaze: github.com/hivaze/dl-experiments

S/O White Circle Research Team for this one.
