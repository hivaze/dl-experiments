# Experiment Ideas

## Layer-Level Analysis & Model Internals

### T-1. Logit Lens / Residual Stream Decoding ✅ [completed](experiments/t1_logit_lens/)
Project the residual stream after each layer through the final LM head to see how token predictions evolve across depth.
- At which layer does the correct next-token first appear in top-k?
- Are there layers that *hurt* predictions (correct token drops in rank)?
- Do different token types (function words vs content words vs reasoning tokens) "crystallize" at different depths?
- Connects to shuffle finding that layers 0-16 are functionally similar — would show whether they're doing similar work or different work with similar weight norms.

### T-2. Layer Knockout / Criticality Mapping ✅ [completed](experiments/t2_layer_knockout/)
Skip each layer one at a time (and pairs) and measure loss degradation.
- Are there "critical" layers whose removal is catastrophic vs "redundant" ones with minimal impact?
- Is criticality correlated with weight norm?
- Do critical layers cluster or distribute uniformly?
- Can you remove N layers while keeping loss below some threshold (structured pruning)?
- Compare: is the "causal bottleneck" (layer whose restoration recovers a corrupted factual recall) the same as the "most critical" layer? Run activation patching on factual/linguistic/arithmetic prompts to check.

### T-3. Layer Swap Cost Matrix ✅ [completed](experiments/t3_layer_swap_cost/)
Directly extends the shuffle experiment. For every pair of layers (i, j), swap them and measure loss degradation.
- Produces a 36×36 "swap cost" matrix — a functional distance between layers.
- Which pairs are nearly interchangeable (low swap cost)? These form **functional equivalence classes**.
- Is the swap cost matrix symmetric? (swapping i↔j might hurt differently than j↔i if the model relies on ordering.)
- Cluster layers by swap cost — do the clusters align with the weight-norm plateau (layers 0-16) from the shuffle experiment?
- Practical: identifies which layers can be reordered for pipeline parallelism without quality loss.

### T-4. Residual Stream Geometry Across Depth ✅ [completed](experiments/t4_residual_stream_geometry/)
Track how the geometry of the hidden-state manifold changes layer by layer.
- **Effective dimensionality** (participation ratio of singular values)
- **Isotropy** (uniformly distributed or clustered in a low-dim cone?)
- **Token clustering**: do semantically similar tokens converge as depth increases?
- **Anisotropy collapse**: do representations become increasingly anisotropic?
- Would explain why activation flow fingerprints work so well in the shuffle experiment.

### T-7. Layer Linearization Gap ✅ [completed](experiments/t7_layer_linearization_gap/)
How nonlinear is each layer's computation on real inputs?
- For each layer, compute the Jacobian J = ∂output/∂input on calibration data.
- Compare `J @ input` (linear approximation) vs actual layer output.
- The gap measures how much the attention softmax and SwiGLU activation actually matter at each depth.
- Hypothesis: early layers are more linear (they mainly do token mixing), late layers are more nonlinear (they do feature composition). If true, early layers could be replaced by cheap linear maps.
- Also: does the linearization gap correlate with layer criticality from T-2?

### T-9. Weight Matrix Spectral Structure ✅ [completed](experiments/t9_weight_spectral_structure/)
Compute SVD of each weight matrix (Q, K, V, O, gate, up, down) per layer. How much capacity does each matrix actually use?
- Do singular values follow power laws? Does the exponent change across depth?
- **Effective rank** (participation ratio of singular values) vs actual rank — reveals compressibility per matrix.
- Layers with low effective rank should be LoRA-friendly; layers with high effective rank resist low-rank adaptation. Validate by actually running LoRA at different ranks per layer.
- Cross-reference with linearization gap (T-7): high-rank layers should also be more nonlinear (using more of their capacity for complex transformations).
- Compare Q/K matrices vs V/O matrices — are attention routing weights lower-rank than value processing weights? Would confirm that "where to attend" is simpler than "what to extract."
- Track how spectral structure differs between the weight-norm plateau layers (0-16) and the later layers from the shuffle experiment.

### T-17. Contrastive Completion Trajectories ✅ [completed](experiments/t17_contrastive_trajectories/)
For each prompt, force-decode semantically related completions (synonyms, antonyms, style variants, unrelated) and track how their hidden-state representations diverge or converge layer-by-layer.
- **Data**: hand-crafted contrastive pairs (`data/text_completions/contrastive_pairs.json`) — 111 groups (v3) across 4 relationship types with prefix control (immediate vs shared divergence). V3 balances sample sizes: 48 immediate-antonym pairs matching 48 immediate-synonym pairs for fair comparison.
- **Contrastive logit lens**: at each layer, project residual stream through LM head and compare logit distributions (KL divergence, rank of alternative tokens) between paired completions.
- **Representational similarity**: cosine similarity, normalized L2, and linear CKA of hidden states between completions at each layer. Do synonyms converge? Do antonyms diverge? At what depth?
- **Pivot token tracking**: for antonym/synonym pairs that differ at a single "pivot" token, track that specific token's representation across layers to find the commitment point.
- **Divergence onset**: identify the first layer where antonym representations separate beyond a statistical threshold (synonym mean − 2σ).
- Cross-reference with T-1 (logit lens phases), T-4 (geometry), T-7 (linearization gap). Hypothesis: divergence onset aligns with T-1's "prediction formation" phase and T-7's high-nonlinearity layers.

---

## Architecture Surgery & Ablation

### T-5. Cross-Model Layer Transplant
Take individual layers from one model and insert them into another (same architecture, different checkpoint).
- How compatible are layers across models?
- Is there a "universal layer format" at certain positions?
- Can you create a chimera model that partially works?

### T-6. Layer Doubling / Iteration
Run each layer through itself N times instead of once. How does the model behave?
- Which layers are "idempotent" (running twice ≈ running once) vs "explosive" (activations diverge)?
- Idempotent layers are likely doing projection/cleanup; explosive layers are doing nontrivial computation.
- Is there a sweet spot where doubling a specific layer actually *improves* output? (Would suggest the layer is "undercooked" at 1 pass.)
- Plot activation norm trajectory as you iterate each layer 1→10 times — expect fixed points, limit cycles, or divergence. The type of attractor reveals the layer's dynamical character.

### T-8. Thinking vs Answer Token Routing
Qwen3 supports `/think` mode with extended reasoning. Do thinking tokens and answer tokens use the model differently?
- Generate responses with thinking enabled. Collect hidden states for `<think>...</think>` tokens vs answer tokens.
- Measure per-layer activation delta `‖h_{k+1} - h_k‖ / ‖h_k‖` for thinking tokens vs answer tokens — which layers "work harder" during reasoning?
- Logit lens (T-1) split by phase: do answer tokens crystallize earlier than thinking tokens? (Thinking tokens may remain uncertain longer since they're exploring.)
- Layer knockout (T-2) split by phase: are different layers critical for coherent thinking vs correct final answers?
- Attention pattern comparison: do thinking tokens attend more broadly (exploring context) while answer tokens attend narrowly (retrieving)?

### T-15. Normalization Layer Analysis & Replacement
Comprehensive study of normalization methods: forward/backward behavior, computational trade-offs, and feasibility of swapping norm layers in a pretrained model without (or with minimal) fine-tuning.

**Normalization methods to investigate:**

| Method | Normalizes Over | Learnable Params | Batch-Dependent | Used In |
|--------|----------------|-----------------|-----------------|---------|
| **LayerNorm** | Feature dims per sample | Scale (γ) + Shift (β) | No | GPT-2, GPT-J, GPT-NeoX, BLOOM, Falcon-40B |
| **RMSNorm** | Feature dims per sample (RMS only, no mean centering) | Scale (γ) only | No | LLaMA 1/2/3, Mistral, Mixtral, Qwen3, OLMo-2, Phi |
| **BatchNorm** | Per channel across batch + spatial | Scale (γ) + Shift (β) | Yes | CNNs (ResNet, EfficientNet); not used in LLMs |
| **GroupNorm** | Per group per sample | Scale (γ) + Shift (β) | No | Small-batch vision (Stable Diffusion U-Net) |
| **InstanceNorm** | Per channel per sample (spatial only) | Optional γ, β | No | Style transfer |
| **LocalResponseNorm** | Adjacent channels | None | No | Legacy CNNs (AlexNet); obsolete |

Special variant: **Qwen3.5-2B uses `Qwen3_5RMSNormGated`** — a gated RMS norm that applies `hidden_states * F.silu(gate)` inside the Gated DeltaNet (linear attention) layers, alongside standard `Qwen3_5RMSNorm` for full-attention layers. Weight init: `torch.zeros(dim)` with `(1.0 + self.weight)` forward — different from LLaMA's `torch.ones(dim)` init.

**Forward & backward pass analysis:**
- Benchmark forward/backward latency for each norm type on B200 at hidden sizes (2048, 2560, 4096) and sequence lengths (512, 2K, 8K).
- RMSNorm skips mean computation — measure actual wall-clock savings vs LayerNorm (literature claims 7-64%, hardware-dependent).
- Backward pass complexity: BatchNorm has the most expensive backward (gradient depends on batch statistics); RMSNorm has simpler backward than LayerNorm (no variance term); measure gradient computation time.
- Memory footprint: RMSNorm < LayerNorm (no bias param, no running stats) < BatchNorm (running mean/var + affine).
- Numerical stability: compare activation distributions after each norm type on identical inputs — measure kurtosis, outlier rates.

**Norm placement analysis (pre-norm vs post-norm):**
- Modern LLMs universally use **pre-norm** (normalize before attention/MLP). Why: post-norm causes gradient vanishing at depth, pre-norm keeps residual stream "clean."
- Measure gradient norms at layer 0 vs layer 35 under both placements on Qwen3-4B — quantify the gradient flow difference.
- **QK-norm**: Qwen3, OLMo-2, and Phi apply additional RMSNorm to Q and K projections inside attention. Ablate QK-norm: does removing it cause attention logit explosion at long context?

**Norm replacement in pretrained Qwen3-4B:**
- **RMSNorm → LayerNorm**: add bias term (initialize to zero), keep scale from original weights. Measure perplexity delta zero-shot — hypothesis: near-zero degradation since RMSNorm ≈ LayerNorm when mean is small.
- **RMSNorm → GroupNorm**: reshape hidden dim into groups, transfer scale weights. Sweep group counts (1, 8, 32, 64). GroupNorm with 1 group = LayerNorm.
- **RMSNorm → BatchNorm1d**: requires batch dimension for statistics — fundamentally changes inference behavior (single-sample inference uses running stats). Measure train/eval mode discrepancy.
- **RMSNorm → Identity (no norm)**: remove all norm layers, measure how quickly activations explode. Establishes the "value" of normalization.
- **LayerNorm → RMSNorm** (on GPT-2-small for comparison): drop bias, keep scale. Does the pretrained model survive?
- For each replacement: measure perplexity on held-out text, GSM8K accuracy (math is sensitive to numerical precision), and activation statistics (mean, variance, max) at each layer.

**Minimal fine-tuning recovery:**
- After each replacement, fine-tune only the norm parameters (scale, bias) for 100/500/1000 steps on a small calibration set. How much quality recovers?
- Compare: fine-tuning norm params only vs fine-tuning norm + adjacent linear layers vs full fine-tuning. Find the minimum intervention for full recovery.
- Hypothesis: RMSNorm↔LayerNorm swaps recover easily; swaps involving BatchNorm or GroupNorm require more fine-tuning due to fundamentally different normalization axes.

**Gated normalization deep dive (Qwen3.5):**
- Profile `Qwen3_5RMSNormGated` vs standard `Qwen3_5RMSNorm` — how much overhead does the SiLU gate add?
- Ablate the gate: replace gated norm with standard RMSNorm in Gated DeltaNet layers. Does quality degrade? The gate may be essential for the linear attention mechanism's stability.
- Can we retrofit gated normalization into standard transformer layers? Test on Qwen3-4B: add SiLU gating to existing RMSNorm layers.

**Key questions:**
- Is RMSNorm → LayerNorm a "free" swap in pretrained models (zero quality loss), confirming that mean centering is redundant when residual stream means are near-zero?
- Which norm replacement causes the most severe degradation — and does it correlate with how different the normalization axes are?
- Can normalization layer replacement serve as a cheap way to adapt models to different hardware (e.g., swapping to a norm type with better kernel support on a target accelerator)?
- Does the gated normalization in Qwen3.5 explain any of its quality advantages over standard Qwen3, or is it purely an efficiency mechanism for linear attention?
- For pruning/distillation: if some layers have near-identity norm behavior (scale ≈ 1, bias ≈ 0), can those norm layers be fused away entirely?

### T-16. Activation Function Survey & Ablation
Comprehensive study of activation functions in transformer MLPs: forward/backward characteristics, computational trade-offs, and feasibility of swapping activations in a pretrained Qwen3.5-4B (`Qwen/Qwen3.5-4B`) without (or with minimal) fine-tuning.

**Activation functions to investigate:**

| Method | Formula | Smooth | Monotonic | Used In |
|--------|---------|--------|-----------|---------|
| **ReLU** | max(0, x) | No | Yes | Classic transformers, early GPT |
| **GeLU** | x · Φ(x) | Yes | No | GPT-2, BERT, OPT, BLOOM |
| **SiLU / Swish** | x · σ(x) | Yes | No | LLaMA, Mistral, Qwen3 (inside SwiGLU) |
| **SwiGLU** | SiLU(W_gate · x) ⊙ (W_up · x) | Yes | N/A (gated) | LLaMA 1/2/3, Mistral, Mixtral, Qwen3, Qwen3.5, Phi, Gemma |
| **GeGLU** | GeLU(W_gate · x) ⊙ (W_up · x) | Yes | N/A (gated) | PaLM, Gemini |
| **ReGLU** | ReLU(W_gate · x) ⊙ (W_up · x) | No | N/A (gated) | T5 v1.1, Flan-T5 |
| **Mish** | x · tanh(softplus(x)) | Yes | No | YOLOv4, some vision models; rarely used in LLMs |
| **Squared ReLU** | max(0, x)² | No | Yes (x≥0) | Primer (Google), some sparse attention work |
| **SoLU** | x · softmax(x) (per-feature) | Yes | No | Anthropic interpretability research; promotes sparsity |
| **ReLU²+LN** | LayerNorm(max(0, x)²) | Piecewise | Yes (pre-LN) | Anthropic interpretability follow-up |
| **Identity (no activation)** | x (gate branch zeroed or bypassed) | Yes | Yes | Ablation baseline — measures activation's total contribution |

**Forward & backward pass analysis:**
- Benchmark forward/backward latency for each activation on B200 at hidden sizes (2560, 4096, 6144) and sequence lengths (512, 2K, 8K).
- SwiGLU requires 3 weight matrices (gate, up, down) vs 2 for non-gated (up, down) — measure actual wall-clock and memory overhead of the extra projection.
- Backward pass: GeLU/SiLU require storing activations for the non-monotonic gradient; ReLU only needs the sign bit. Measure activation memory difference across a full forward-backward pass.
- Gradient flow: compute gradient norms at layer 0 vs final layer for each activation — which activations maintain the healthiest gradient magnitude across depth?
- Numerical stability: measure activation output distributions (mean, variance, kurtosis, outlier rate, sparsity) after each activation type on identical inputs. Squared ReLU and SoLU should be sparser; does sparsity help or hurt downstream layers?

**Gated vs non-gated comparison:**
- SwiGLU vs plain SiLU: isolate the effect of gating by comparing `SiLU(W_gate · x) ⊙ (W_up · x)` vs `SiLU(W_up · x)` (single projection + activation). Does gating improve expressivity enough to justify the 50% parameter increase in the MLP?
- GeGLU vs GeLU, ReGLU vs ReLU: same gated-vs-ungated comparison for other activation families. Is the benefit of gating consistent across activation types, or does it interact with smoothness/monotonicity?
- Gate activation statistics: what does the gate branch actually learn? Measure sparsity, variance, and correlation with the value branch across layers and depth. Is the gate doing input-dependent feature selection or acting as a near-constant scale?

**Activation replacement in pretrained Qwen3.5-4B:**
- **SwiGLU → GeGLU**: replace SiLU with GeLU in the gate branch, keep all weights. Measure perplexity delta zero-shot — hypothesis: near-zero degradation since SiLU ≈ GeLU for most input ranges.
- **SwiGLU → ReGLU**: replace SiLU with ReLU in the gate branch. Introduces non-smoothness — does the model tolerate the hard zero threshold?
- **SwiGLU → plain SiLU**: remove the gating mechanism entirely, merge gate+up projections or drop one. Measures how much the model depends on multiplicative gating.
- **SwiGLU → GeLU (non-gated)**: full replacement, dropping gating and changing activation. Maximum architectural disruption within reasonable bounds.
- **SwiGLU → Squared ReLU**: test the sparsity hypothesis — does a sparser activation improve interpretability metrics (e.g., feature monosemanticity) while degrading task performance?
- **SwiGLU → SoLU**: Anthropic's interpretability-motivated activation — does it produce more interpretable features at the cost of quality?
- **SwiGLU → Identity**: remove all nonlinearity from MLP blocks, keeping linear projections. Establishes the total "value" of nonlinear activations — how much of the model's capability lives in the activation function vs the linear transforms?
- For each replacement: measure perplexity on held-out text, GSM8K accuracy (math is sensitive), and activation statistics (mean, variance, sparsity, max) at each layer.

**Per-layer activation heterogeneity:**
- Replace the activation function in only one layer at a time (sweep all layers). Produce a "layer sensitivity to activation change" map analogous to T-2's criticality map.
- Are early layers more tolerant of activation swaps than late layers (or vice versa)?
- Cross-reference with T-7 (linearization gap): layers with high nonlinearity gap should be more sensitive to activation changes.
- Cross-reference with T-9 (spectral structure): do layers with higher effective rank rely more on activation nonlinearity?
- Test mixed-activation models: e.g., SwiGLU for layers 0–18, ReGLU for layers 19–35. Can different layers tolerate different activations?

**Minimal fine-tuning recovery:**
- After each replacement, fine-tune only the MLP weights (gate_proj, up_proj, down_proj) for 100/500/1000 steps on a small calibration set. How much quality recovers?
- Compare: fine-tuning MLP only vs MLP + adjacent norm layers vs full fine-tuning. Find the minimum intervention for full recovery.
- Hypothesis: swaps within the GLU family (SwiGLU↔GeGLU↔ReGLU) recover easily; swaps that remove gating or change to fundamentally different activations (SoLU, Squared ReLU) require more fine-tuning.
- LoRA recovery: apply LoRA (rank 16, 64) to MLP layers after activation swap. Can low-rank adaptation compensate for activation mismatch?

**Sparsity & interpretability analysis:**
- For each activation, measure neuron-level sparsity: fraction of near-zero activations across calibration data. ReLU/Squared ReLU/SoLU should be sparser than SiLU/GeLU.
- Does sparsity correlate with feature monosemanticity? Run a simple feature attribution analysis: for the top-k most active neurons under each activation, measure how consistently each neuron fires for the same semantic concept across different inputs.
- Practical: sparse activations enable hardware-accelerated sparse computation. Measure actual speedup (if any) from activation sparsity on B200 using torch sparse ops.

**Key questions:**
- Is SwiGLU → GeGLU a "free" swap in pretrained models, confirming that the specific activation (SiLU vs GeLU) matters less than the gating mechanism?
- How much of Qwen3.5-4B's capability comes from the gating mechanism vs the activation nonlinearity vs the linear projections? (Identity ablation quantifies this.)
- Do sparsity-promoting activations (Squared ReLU, SoLU) produce more interpretable internal representations at an acceptable quality cost?
- Which activation replacement causes the most severe degradation — and does it correlate with how different the activation's gradient profile is from SiLU?
- Can per-layer activation heterogeneity (different activations at different depths) outperform a uniform activation choice? If so, what's the optimal assignment?
- For model surgery/adaptation: is activation replacement a viable cheap intervention (like norm replacement in T-15), or does the model's learned weight structure encode strong assumptions about the specific activation function?

---

## Inference & Systems Experiments

### T-10a. Attention Architecture Survey
Comparative study of attention mechanism **designs** — how they compute attention, what they store, and how they trade off quality vs efficiency. Focus on architectural properties, not kernel-level optimization.

**Architectures to compare:**

| Architecture | Type | State | Complexity | KV Memory | Used In |
|-------------|------|-------|-----------|-----------|---------|
| **MHA** (Multi-Head Attention) | Full quadratic | KV cache per head | O(n²d) | O(n·h·d) | GPT-2, GPT-3, OPT |
| **GQA** (Grouped Query Attention) | Full quadratic, shared KV | KV cache per group | O(n²d) | O(n·g·d) where g < h | LLaMA 2/3, Mistral, Qwen3 |
| **MLA** (Multi-head Latent Attention) | Full quadratic, compressed KV | Low-rank KV via `kv_lora_rank` | O(n²d) | O(n·r) where r ≪ h·d | DeepSeek-V2/V3, DeepSeek-R1 |
| **MLHA** (Multi-head Linearized Attention) | Linear approximation | Recurrent state | O(n·d²) | O(d²) fixed | Research; linearized softmax |
| **Gated DeltaNet** | Gated delta rule, recurrent | Recurrent matrix state | O(n·d²) chunk / O(n·d) recurrent | O(d²) fixed | Qwen3.5 (linear layers) |
| **Lightning Attention** | IO-aware linear | Cumsum-based causal | O(n·d²) | O(d²) fixed | TransNormerLLM |
| **Mamba2 / SSD** | Structured state space | Selective scan state | O(n·d·s) | O(d·s) fixed | Mamba-2, Jamba, Zamba |
| **RWKV-7** | Linear RNN | Token-shift + decay state | O(n·d) | O(d) fixed | RWKV-7 (Eagle/Finch) |
| **Based** | Linear + sliding window | Taylor expansion + local window | O(n·d²) + O(n·w) | O(d² + w·d) | Based (Stanford) |
| **NSA / DSA** (Native Sparse Attention) | Trainable sparse, hierarchical | Compressed blocks + top-k fine-grained + sliding window | O(n·k·d) where k ≪ n | O(n·d) but sparse | DeepSeek-V3.2; ACL 2025 Best Paper |
| **Differential Attention** | Noise-canceling quadratic | Difference of two softmax maps from split Q/K heads | O(n²d) | O(n·g·d) (GQA-compatible) | Microsoft DIFF Transformer; ICLR 2025 |
| **MoBA** (Mixture of Block Attention) | MoE-routed sparse | Parameter-free top-k gating routes queries to KV blocks | O(n·k·b·d) where k blocks selected | O(n·g·d) | Moonshot AI / Kimi production |
| **KDA** (Kimi Delta Attention) | Channel-wise gated delta rule | Gated DeltaNet with per-channel gating in recurrent state | O(n·d²) chunk / O(n·d) recurrent | O(d²) fixed | Kimi Linear (48B/3B active); first linear attn to beat full attn fairly |
| **Sliding Window / Global hybrid** | Interleaved local + global | Alternating sliding-window layers with full-context layers | O(n·w·d) local + O(n²d) global | O(w·g·d) local + O(n·g·d) global | Gemma 3 (5:1), Cohere Command A (3:1+NoPE), Llama 4 (iRoPE) |
| **Falcon-H1 parallel hybrid** | Parallel attention + Mamba2 | Attention and Mamba2 run in parallel, outputs concatenated | O(n²d) + O(n·d·s) parallel | Combined KV + SSM state | Falcon-H1 (0.5B–34B); configurable attn/Mamba head ratio |

**Architectural property analysis:**
- **Expressivity**: implement each architecture as a drop-in attention replacement in a small model (~500M). Train from scratch on the same data (e.g., SlimPajama subset) for fixed compute budget. Compare validation perplexity — pure architectural quality comparison, no confounding from kernel speed or pretrained weights.
- **Recall capacity**: synthetic retrieval tasks (associative recall, induction heads, needle-in-haystack) at increasing sequence lengths. Full quadratic attention should be perfect; linear/recurrent methods will degrade — find the degradation curve for each.
- **State capacity**: for recurrent/linear methods, measure how much information the fixed-size state can retain. Feed sequences of increasing length, probe the state for information from early tokens. Which architectures forget gracefully vs catastrophically?
- **Hybrid analysis**: Qwen3.5-2B uses 3 Gated DeltaNet + 1 full attention per cycle. Profile which layer type handles which information: do full-attention layers act as "refresh points" that re-ground the recurrent state? Ablate: what happens if you remove all full-attention layers? All DeltaNet layers?

**KV memory scaling:**
- For each architecture, compute theoretical and measured KV/state memory at context lengths 2K, 8K, 32K, 128K, 512K.
- Full attention (MHA/GQA): memory grows linearly with context. At what context length does KV cache exceed model weights in memory?
- MLA: measure actual compression ratio vs GQA at different `kv_lora_rank` values. Does the low-rank approximation lose critical information for specific task types?
- Linear/recurrent (DeltaNet, Mamba2, RWKV-7): fixed-size state — but is the state large enough? Measure state size vs hidden dim and compare to the KV cache it replaces.

**Quality comparison on downstream tasks:**
- Perplexity: WikiText-103, C4, The Pile (different domains test generalization).
- Retrieval: RULER benchmark (multi-hop retrieval, variable-tracking) at 4K, 16K, 64K context.
- Math/reasoning: GSM8K, MATH — attention mechanism may affect multi-step reasoning where the model needs to retrieve intermediate results from earlier in the chain.
- Code: HumanEval, MBPP — long-range dependencies in code (function definitions, imports) stress attention differently than prose.
- Compare models of matched size: Qwen3-1.7B (pure GQA), Qwen3.5-2B (hybrid GQA + DeltaNet), a Mamba2-based model (~2B) if available.

**Hybrid interleaving patterns (cross-model comparison):**
- Qwen3.5-2B: 3 DeltaNet + 1 full attention (sequential)
- Jamba 1.5: 7 Mamba + 1 attention (sequential) + MoE every 2 blocks
- Falcon-H1: attention + Mamba2 in parallel per block (not sequential)
- Gemma 3: 5 sliding-window + 1 global attention (same mechanism, different window)
- Llama 4: 3 RoPE local + 1 NoPE global (same mechanism, different positional encoding)
- Cohere Command A: 3 sliding-window RoPE + 1 global NoPE (parallel block design)
- Compare: does the interleaving pattern matter more than the component architectures? Test by holding the ratio constant and varying the pattern (e.g., AAAB vs AABA vs ABAA for 3:1).

**Sparse attention analysis (NSA/DSA, MoBA, Differential):**
- NSA/DSA: implement the three-path routing (compressed blocks + top-k fine-grained + sliding window). Measure how block selection adapts to different input types — does it learn to select relevant blocks for retrieval tasks?
- MoBA: measure the quality-sparsity tradeoff as you vary the number of selected blocks. At what sparsity level does quality degrade? Compare MoBA's learned routing vs NSA's hierarchical selection.
- Differential Attention: implement the dual-softmax differencing. Measure attention entropy and sparsity vs standard attention — does differencing actually produce cleaner attention patterns? Test on hallucination-prone prompts where noise cancellation should help.
- All three are compatible with full-attention architectures (GQA/MHA) — they modify *which* tokens to attend to, not the attention mechanism itself. Can they be composed with MLA or KV compression?

**Architectural ablations on Qwen3.5-2B:**
- Replace all DeltaNet layers with GQA (make it a pure transformer). Quality change?
- Replace all GQA layers with DeltaNet (make it a pure recurrent model). Quality change?
- Vary the hybrid ratio: 1:1, 2:1, 4:1, 7:1 (DeltaNet:GQA). Is 3:1 optimal, or was it arbitrary?
- Swap DeltaNet for Mamba2 or RWKV-7 in the linear layer slots. Is the specific linear architecture critical, or is it just "any recurrent layer" that works?
- Test Falcon-H1-style parallel hybrid: run DeltaNet and GQA in parallel (concatenate outputs) instead of alternating. Does parallel beat sequential?
- Upgrade DeltaNet to KDA (channel-wise gating): does per-channel gating meaningfully improve over the existing scalar gating?

**Key questions:**
- Is the hybrid (linear + quadratic) architecture strictly better than pure approaches, or is it a compromise?
- At what context length do recurrent/linear methods start losing quality vs full attention? Is there a clean crossover point?
- Does MLA's low-rank KV compression lose information that GQA retains, or is GQA's per-head KV redundant?
- For Qwen3.5-2B's hybrid architecture: do the full-attention layers serve as "memory checkpoints" that prevent recurrent state degradation?
- Is there an optimal hybrid ratio (linear:quadratic layers) that maximizes quality per FLOP?
- Can you predict which architecture will work best for a given task type (retrieval-heavy vs generation-heavy vs reasoning) based on its state capacity?
- Do sparse attention methods (NSA, MoBA, Differential) offer a better quality-efficiency tradeoff than linear/recurrent methods for long context? They're still O(n) in selected tokens but retain exact attention over the selected set.
- Sequential vs parallel hybrid: is Falcon-H1's parallel design (attention ‖ Mamba2) fundamentally better than Qwen3.5's sequential design (attention → DeltaNet), or just different?

### T-10b. Attention Kernel Benchmark
Pure performance benchmarking of attention **kernel implementations** — different ways to compute the same (or similar) attention operation on GPU. Focuses on throughput, latency, memory, and numerical accuracy, not architectural design.

**Kernels to benchmark:**

*For full quadratic attention (GQA/MHA):*

| Kernel | Library | Key Technique | Precision | Notes |
|--------|---------|--------------|-----------|-------|
| `eager` | PyTorch | Naive matmul + softmax | bf16/fp16 | Reference baseline, no fusion |
| `sdpa` | PyTorch ≥2.0 | Auto-dispatch to best backend | bf16/fp16 | `torch.nn.functional.scaled_dot_product_attention` |
| `flash_attention_2` | flash-attn | Tiling + recomputation, IO-aware | bf16/fp16 | Tri Dao; fused kernel, no materialized attention matrix |
| `flash_attention_3` | flash-attn 3 | Asynchronous softmax, FP8 path | bf16/fp16/fp8 | B200/H100 specific; warp specialization |
| `FlashInfer` | flashinfer | PagedAttention + RaggedTensor | bf16/fp16/fp8 | Optimized for serving (variable-length batches, paged KV) |
| `flex_attention` | PyTorch ≥2.5 | JIT-compiled Triton via `score_mod` | bf16/fp16 | Supports custom attention patterns (causal, sliding window, etc.) |
| `xformers` | xformers | Memory-efficient attention | bf16/fp16 | Meta; block-sparse support |
| `cutlass` | NVIDIA CUTLASS | Template-based GEMM fusion | bf16/fp16/fp8 | Low-level, maximum control |
| `cuDNN` | cuDNN ≥9 | Graph-based fused attention | bf16/fp16/fp8 | NVIDIA's official fused multi-head attention |
| `flash_attention_4` | flash-attn 4 | CuTe-DSL, dual-exp pipeline | bf16/fp16/fp8 | **Blackwell-native (B200)**; 3.6x over FA2 at 32K, 71% HW util; in vLLM v0.17+ |
| `triton_attn` | vLLM (TRITON_ATTN) | Cross-platform Triton kernel | bf16/fp16 | ~800 lines Triton; 100.7% of FA3 on H100; works on NVIDIA/AMD/Intel |
| `SageAttention3` | thu-ml | Quantized attention (FP4/INT4/INT8) | fp4/int4/int8 | 2–5x over FA2; not in vLLM yet; standalone benchmark candidate |
| `trtllm_attn` | TensorRT-LLM via FlashInfer | FP8-optimized decode | fp8 | Blackwell SM100-specific; auto-preferred for decode in vLLM |

*For MLA (Multi-head Latent Attention):*

| Kernel | Library | Key Technique | Notes |
|--------|---------|--------------|-------|
| `FlashMLA` | deepseek-ai/FlashMLA | Fused MLA with paged KV | 660 TFLOPS on H800; primary MLA kernel on Hopper |
| `FlashMLA Sparse` | deepseek-ai | Sparse MLA for DeepSeek-V3.2 | Supports NSA/DSA sparse patterns |
| `CUTLASS MLA` | vLLM | CUTLASS-based MLA | Default on B200; no FP8 KV support |
| `FlashInfer MLA` | flashinfer | General MLA | Preferred for FP8 KV cache |
| `Triton MLA` | vLLM | Cross-platform MLA fallback | Portable across GPU vendors |

*For linear/recurrent attention (DeltaNet, Mamba2):*

| Kernel | Library | Architecture | Key Technique |
|--------|---------|-------------|--------------|
| `fused_recurrent_gated_delta_rule` | FLA | Gated DeltaNet | Fused recurrent scan, single-token optimal |
| `chunk_gated_delta_rule` | FLA | Gated DeltaNet | Chunked parallel, prefill-optimal |
| `mamba_ssm` (selective_scan) | mamba-ssm | Mamba2/SSD | Hardware-aware chunked selective scan |
| `causal_conv1d` | causal-conv1d | Mamba2 pre-processing | Fused causal 1D convolution |
| `triton_linear_attention` | FLA | Generic linear attn | Triton-based, portable |

**Microbenchmark methodology:**
- Isolate each kernel from the full model. Feed synthetic inputs of controlled shape: batch × heads × seq_len × head_dim.
- Sweep sequence lengths: 128, 512, 2K, 8K, 32K, 128K.
- Sweep batch sizes: 1, 4, 16, 64.
- Sweep head configurations: GQA (32q/8kv), MHA (32q/32kv), heavy GQA (32q/2kv).
- Measure: forward latency (μs), backward latency (μs), peak GPU memory (MB), throughput (TFLOP/s vs theoretical peak), numerical accuracy (max abs diff + mean abs diff vs fp64 eager reference).
- Each measurement: 100 warmup iterations, 1000 timed iterations, report mean + p50 + p99.

**Prefill vs decode profiling:**
- **Prefill** (compute-bound): process full prompt in one pass. Large batch of queries against growing KV. Kernels that fuse well and maximize FLOP utilization win.
- **Decode** (memory-bandwidth-bound): single new token, attend to full KV cache. Kernels with efficient KV read patterns win. FlashInfer's paged attention should excel here.
- Measure separately: prefill latency at seq_len 512/2K/8K/32K, decode latency at KV cache size 512/2K/8K/32K.
- For FLA kernels: compare `chunk_*` (prefill-optimal) vs `fused_recurrent_*` (decode-optimal) on the same DeltaNet layer. Measure the crossover point where switching kernels pays off.

**Kernel dispatch & overhead analysis:**
- Measure kernel launch overhead in isolation: how many microseconds does each kernel's launch + teardown cost, independent of compute?
- For short sequences (128-512 tokens), launch overhead can dominate. Find the minimum sequence length where each kernel's compute exceeds its overhead.
- `flex_attention` JIT compilation: measure first-call compilation time, warm-call dispatch time, and how compilation cost scales with `score_mod` complexity.
- SDPA auto-dispatch: trace which backend SDPA selects at each (batch, seq_len, head_dim) combo. Is its selection optimal, or does manual kernel selection beat it?
- Multi-kernel pipelines: for hybrid models (Qwen3.5-2B), measure the overhead of switching between FA2 (for full-attn layers) and FLA (for DeltaNet layers) within a single forward pass.

**B200-specific optimizations:**
- **FlashAttention-4 is the primary target**: FA4 is purpose-built for Blackwell — uses CuTe-DSL, software-emulated exp via FMA to work around B200's asymmetric scaling (tensor core throughput jumped 2.25x but softmax units didn't). Must be the baseline for all B200 benchmarks.
- FP8 attention: FA3, FA4, FlashInfer, and TRTLLM all support FP8 paths. Measure throughput gain vs bf16 and quantify numerical degradation. Is FP8 attention viable for quality-sensitive tasks?
- **SageAttention3 FP4**: if FP8 attention works, does FP4 (SageAttention3) push further? Measure quality cliff between FP8 → FP4 attention quantization.
- Tensor Core utilization: profile each kernel's SM occupancy and Tensor Core usage via `ncu` (NVIDIA Nsight Compute). FA4 claims 71% utilization — verify, and compare others.
- Memory hierarchy: profile L2 cache hit rates for each kernel. IO-aware kernels (FA2, FA3, FA4) should have better cache behavior — verify on B200's specific L2 size.
- NVLink: for 2-GPU setups with tensor parallelism, measure all-reduce overhead between attention heads. Does kernel choice affect communication patterns?
- **Cross-platform comparison**: run Triton Attention (TRITON_ATTN) on B200 alongside FA4 — how close does portable Triton get to vendor-optimized CUDA?

**Numerical accuracy deep dive:**
- Compare each kernel's output to fp64 eager reference. Report max abs error, mean abs error, and relative error distribution.
- Do errors accumulate across layers? Run full model inference (36 layers) with each kernel and compare final logits. Small per-kernel errors may compound.
- FP8 vs bf16 vs fp16: measure accuracy-performance tradeoff at each precision. Is bf16 always better than fp16 on B200, or does fp16 have a throughput advantage?
- Attention sink tokens (first few positions): do any kernels handle the numerically extreme softmax values at sink positions differently?

**End-to-end model benchmarks (Qwen3.5-2B):**
- Plug each kernel into the full model and measure end-to-end: prefill latency, decode throughput (tokens/sec), peak memory.
- Optimal kernel routing: test all combinations of {FA2, FA3, FA4, FlashInfer, triton_attn, flex_attention} for full-attn layers × {chunk_gated_delta_rule, fused_recurrent} for DeltaNet layers. Find the fastest combo at each sequence length.
- MLA kernel comparison (if testing DeepSeek models): FlashMLA vs CUTLASS MLA vs FlashInfer MLA vs Triton MLA. Which is fastest on B200? Does FP8 KV cache work with all of them?
- Compare against vLLM's default kernel selection. Is vLLM already optimal, or can manual routing improve throughput?

**Key questions:**
- Which kernel is fastest for each (architecture, sequence length, batch size, phase) combination? Produce a lookup table.
- **FA4 vs FA3 vs FA2 on B200**: is FA4 strictly dominant, or are there regimes (short sequences, small batches) where older versions win due to lower launch overhead?
- Does FlashInfer's paged attention overhead pay off only at high batch sizes, or is it always competitive?
- For Qwen3.5-2B's hybrid architecture, what is the optimal kernel assignment per layer type, and how much does it improve over default dispatch?
- Is FP8 attention on B200 a free lunch (measurable speedup, negligible quality loss), or does it require careful task-specific validation? How much further does FP4 (SageAttention3) push?
- What is the theoretical vs achieved throughput ceiling on B200 for attention — FA4 claims 71% utilization, can anything do better?
- How close does cross-platform Triton Attention get to vendor-specific kernels (FA4, CUTLASS)? Is portability worth the performance gap?
- For MLA workloads: is FlashMLA on Hopper comparable to CUTLASS MLA on Blackwell, or does the architecture need different kernels per GPU generation?

### T-11. Quantization Methods Comparative Analysis
Comprehensive comparison of quantization techniques across quality, performance, and mathematical accuracy dimensions.

**Methods/tools to evaluate:**
- **llama.cpp (GGUF)** — k-quant family (Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ variants), importance-matrix-guided quantization
- **llm-compressor (vLLM/Neural Magic)** — GPTQ, AWQ, SmoothQuant, FP8 (W8A8), INT8 weight-only, sparse+quantized (2:4 sparsity + INT8)
- **torchao** — INT4/INT8 weight-only, dynamic activation quantization, GPTQ via torchao, autoquant (profile-guided format selection)
- **AutoAWQ** — AWQ with per-channel scaling
- **HQQ** — half-quadratic quantization (no calibration data required)
- **bitsandbytes** — NF4/FP4 (QLoRA-style), INT8 mixed-precision decomposition
- **AQLM / QuIP#** — additive/vector quantization pushing below 3-bit

**Evaluation axes:**
- **Quality**: perplexity on held-out text (WikiText, C4), accuracy on benchmarks (MMLU, HellaSwag, ARC)
- **Math accuracy**: GSM8K, MATH — quantization can disproportionately hurt arithmetic/reasoning; measure per-difficulty-level degradation
- **Prefill performance**: time-to-first-token at varying prompt lengths (512, 2K, 8K, 32K) — compute-bound, sensitive to weight format and kernel dispatch
- **Decode performance**: tokens/sec for autoregressive generation — memory-bandwidth-bound, where quantization should shine
- **Under-load performance**: throughput and latency under concurrent requests (batch sizes 1, 8, 32, 64) — how does each method scale with batching? Does quantization help more under memory pressure?
- **Peak memory**: GPU memory footprint at each batch size
- **Quantization cost**: time and compute to produce the quantized model (calibration-free methods like HQQ vs calibration-heavy like GPTQ)

**Key questions:**
- What is the quality-performance Pareto frontier across methods and bit-widths?
- At what bit-width does math accuracy degrade catastrophically vs gracefully? Is there a method that preserves math better than others at the same compression ratio?
- Do calibration-based methods (GPTQ, AWQ) meaningfully outperform calibration-free (HQQ, RTN) on B200s, or does the hardware's native throughput mask the difference?
- For decode-bound workloads, is INT4 weight-only sufficient, or does W4A16 vs W4A8 matter?
- Under heavy load (high batch size), does FP8 (W8A8) dominate because activation quantization reduces memory bandwidth for KV cache and activations?
- Which method offers the best out-of-the-box experience (no calibration, no fuss, good defaults)?
- Are there method combinations that compound (e.g., 2:4 sparsity + INT8 from llm-compressor)?

### T-12. CUDA Graphs & torch.compile Deep Dive
In-depth investigation of CUDA graph capture and `torch.compile` behavior on Qwen3.5-2B's heterogeneous architecture (mix of Gated DeltaNet + full attention layers). Understand what actually gets optimized, what breaks, and where the real gains come from.

**Operation-level profiling (isolated microbenchmarks):**
- **RMSNorm**: eager vs compiled vs CUDA graph — measure kernel fusion with surrounding ops
- **SwiGLU MLP block**: does torch.compile fuse gate_proj + up_proj + SiLU + mul + down_proj into fewer kernels? Measure kernel count and latency before/after
- **Full attention (GQA)**: compile behavior with dynamic sequence lengths — does recompilation trigger on every new length? Cost of padding vs dynamic shapes
- **Gated DeltaNet (linear attention)**: does the recurrent/chunk formulation compile cleanly, or do custom CUDA kernels (FLA) bypass the compiler? Measure compiled-Python vs custom-kernel gap
- **Rotary embeddings (RoPE)**: eager vs compiled — is this a fusion target or negligible?
- **Residual add + norm**: classic fusion opportunity — verify it actually fuses under compile

**CUDA graph capture analysis:**
- Which ops are graph-capturable and which force graph breaks? Catalog every graph break in a full Qwen3.5-2B forward pass with `TORCH_LOGS="graph_breaks"`
- Heterogeneous layer challenge: can a single CUDA graph cover both DeltaNet and full-attention layer types, or do you need separate graphs per layer type?
- Static vs dynamic shapes: measure the cost of replaying a padded static-shape graph vs re-capturing for each new sequence length
- KV cache management: does in-place KV cache update work within captured graphs, or does it require pre-allocated static buffers?
- Decode-phase graph: capture the single-token decode step as a graph — measure launch overhead reduction vs eager (target: sub-10μs dispatch)
- Multi-graph stitching: capture prefill and decode as separate graphs, measure transition overhead

**torch.compile modes & backends:**
- Compare `mode="default"` vs `mode="reduce-overhead"` (CUDA graphs under the hood) vs `mode="max-autotune"` (autotuning + Triton kernels)
- Backend comparison: `inductor` (default) vs `cudagraphs` vs `eager` (baseline)
- Measure compilation time itself — first-call latency penalty at each mode
- Inspect generated Triton kernels for key operations — are they competitive with hand-written CUDA?
- `fullgraph=True` vs allowing graph breaks: does forcing full graph capture improve perf or just cause errors?
- `dynamic=True` vs `dynamic=False`: quantify the cost of dynamic shape support on sequence length variation

**End-to-end model benchmarks:**
- Prefill latency: eager vs compile vs CUDA graph at sequence lengths 128, 512, 2K, 8K, 32K
- Decode throughput: tokens/sec for each method, batch sizes 1, 4, 16, 64
- Memory overhead: CUDA graphs pre-allocate — measure the memory tax at each batch/seqlen combo
- Warmup cost: time from cold start to steady-state throughput for each method
- Profile with `torch.profiler` + Chrome trace: visualize kernel-level timeline for eager vs compiled, count kernel launches, measure GPU idle gaps

**Key questions:**
- What fraction of Qwen3.5-2B's forward pass is actually accelerated by torch.compile? Is it 20% or 80%?
- For the heterogeneous architecture, is the optimal strategy to compile DeltaNet layers and full-attention layers separately?
- At what batch size does CUDA graph replay overhead become negligible compared to compute?
- Does `max-autotune` find better Triton kernels than the defaults for GQA or DeltaNet ops on B200?
- What is the minimum-effort path to maximum speedup: just wrapping the model in `torch.compile`, or does it require surgical per-module compilation?
- How do these optimizations interact with quantization (T-11)? Does compile + INT4 compound, or do quantized kernels bypass the compiler?

### T-13. KV-Cache Optimization Strategies
Systematic study of KV-cache compression, quantization, eviction, and architectural techniques. KV cache is the dominant memory bottleneck at long context and high batch sizes — understanding the tradeoff landscape is critical for practical deployment.

**KV-cache quantization:**
- **FP8 (E4M3 / E5M2)** — vLLM's native KV cache dtype, per-tensor scaling factors (k_scale, v_scale). Measure quality degradation vs bf16 at 8K, 32K, 128K context. Does E4M3 vs E5M2 matter in practice?
- **KIVI (INT4/INT2 per-channel)** — transformers `QuantizedCache` with configurable `nbits` and `q_group_size`. Key insight: keeps a residual buffer of recent tokens in full precision. Sweep `residual_length` (32, 64, 128, 256) to find the quality cliff.
- **KV quantization via llm-compressor** — apply compressed-tensors format to KV cache. Compare calibrated vs uncalibrated quantization. Does calibration data from one domain transfer to another?
- **Mixed-precision KV** — quantize K to INT4 but keep V in FP8 (or vice versa). Keys need less precision than values? Test asymmetric schemes.
- **Per-head vs per-tensor vs per-channel scaling** — finer granularity should help but adds overhead. Find the sweet spot on B200.

**Architectural KV reduction:**
- **MLA (Multi-head Latent Attention)** — DeepSeek-style low-rank KV projection via `kv_lora_rank`. Measure effective KV memory reduction vs quality on models that support it. How does `kv_lora_rank` scale with model size?
- **GQA ratios** — Qwen3 uses 16 query heads / 8 KV heads. Simulate different GQA ratios (1, 2, 4, 8 KV heads) by merging/splitting heads post-hoc. Where does quality collapse?
- **Cross-Layer KV Sharing (CLA)** — vLLM v1 supports cross-layer cache reuse. Which layers can share KV without quality loss? Use layer similarity metrics from T-3 (swap cost matrix) to predict shareable pairs.
- **YOCO (You Only Cache Once)** — single KV cache shared across decoder layers. vLLM has early support (`kv_sharing_fast_prefill`). Measure prefill speedup and quality impact.

**Token eviction & sparsification:**
- **H2O (Heavy Hitter Oracle)** — retain only high-attention-score tokens + attention sinks. Sweep retention budget (256, 512, 1024, 2048 tokens) at 32K+ context.
- **SnapKV** — observation-window-based token selection, clusters important KV entries. Compare to H2O on needle-in-haystack and long-document QA.
- **PyramidKV** — layer-dependent cache budget (more cache in early layers, less in later). Aligns with finding from T-9 that attention patterns change across depth.
- **StreamingLLM / Attention Sinks** — keep first few tokens (sinks) + sliding window of recent tokens. Measure the minimum sink count needed per model. Does this work for Qwen3's GQA?
- **Scissorhands** — importance-based eviction using historical attention. Compare eviction heuristics: cumulative attention, recent attention, pivotal tokens.

**Cache management & memory:**
- **Paged KV cache (vLLM)** — block-based allocation. Sweep `block_size` (8, 16, 32, 64, 128) and measure fragmentation overhead vs allocation speed. Is there an optimal block size for B200 memory hierarchy?
- **Prefix caching** — measure cache hit rates and latency savings for system-prompt-heavy workloads. How large must the shared prefix be for prefix caching to pay off?
- **KV offloading to CPU** — vLLM's `cpu_offload_gb` and `kv_offloading_backend` (native vs LMCache). Measure throughput degradation vs context length extension. At what PCIe bandwidth does this become viable?
- **Speculative decoding interaction** — tree attention (vLLM `tree_attn`) requires branching KV cache. How does KV quantization interact with speculative decoding? Does FP8 KV hurt acceptance rates?

**Benchmarking methodology:**
- Fix model (Qwen3-1.7B for fast iteration, Qwen3-VL-2B for multimodal).
- Metrics: peak KV memory, time-to-first-token, decode tokens/sec, perplexity, needle-in-haystack accuracy, long-context retrieval accuracy (RULER benchmark).
- Sequence lengths: 2K, 8K, 32K, 128K. Batch sizes: 1, 8, 32.
- Each technique tested standalone and in combination (e.g., FP8 KV + H2O eviction + prefix caching).

**Key questions:**
- What is the maximum context length achievable on a single B200 (183GB) with each optimization, while maintaining >95% of bf16 quality?
- Is FP8 KV cache a free lunch (no quality loss, pure memory savings), or does it degrade on specific task types (math, retrieval, long-range reasoning)?
- For long-context serving, is it better to quantize the cache (keep all tokens, lower precision) or evict tokens (keep fewer tokens, full precision)? Where's the crossover?
- Can cross-layer KV sharing + FP8 quantization compound to achieve 4-8x cache reduction without meaningful quality loss?
- How do these KV optimizations interact with the attention kernel choice from T-10 (FlashAttention-2 vs FlashInfer vs FLA)?
- For Qwen3-Next's hybrid architecture: do Gated DeltaNet layers (which use recurrent state, not KV cache) reduce the overall cache pressure enough to make KV optimization less critical?

### T-14. NIXL & Disaggregated / Expert-Parallel Inference
Deep dive into [NIXL](https://github.com/ai-dynamo/nixl) (NVIDIA Inference Xfer Library) — a data movement abstraction layer for disaggregated inference — with particular focus on expert parallelism for MoE models.

**Understanding NIXL internals:**
- **Agent model** — one agent per inference process, manages GPU/memory nodes on its host. Explore the agent lifecycle: initialization, memory registration, metadata exchange (ETCD-backed), teardown.
- **Backend plugins** — UCX (RDMA, NVLink, TCP), CUDA GDS (GPUDirect Storage), POSIX, Mooncake. Profile each backend's latency and throughput on our B200 setup. Which backend wins for GPU-to-GPU vs GPU-to-CPU vs GPU-to-SSD?
- **Transfer primitives** — descriptor lists (dlist), prepared transfers (`prep_xfer_dlist` + `make_prepped_xfer` for known layouts) vs dynamic transfers (`initialize_xfer`). Measure overhead of preparation vs dynamic path.
- **Notification mechanism** — how does transfer completion signaling work? Latency of notification delivery vs polling.
- Run `nixlbench` on our 2x B200 setup to establish baseline transfer bandwidth and latency across backends.

**Expert parallelism for MoE models:**
- **All-to-all communication patterns** — MoE expert routing requires dynamic token-to-expert dispatch across GPUs. Implement a microbenchmark that simulates expert routing: N tokens dispatched to K experts across 2 GPUs via NIXL, measure latency vs NCCL all-to-all.
- **Expert load balancing** — vLLM's `eplb_state.py` (Expert Parallel Load Balancing) handles dynamic expert placement. Trace the interaction: how does EPLB decide expert-to-GPU mapping, and how does NIXL execute the resulting transfers?
- **Token routing overhead** — in MoE inference, the router decides which expert processes each token. Profile the full pipeline: router decision → token gather → NIXL transfer → expert compute → result scatter. Where is the bottleneck?
- **Expert cache** — vLLM's `ec_transfer/` (expert cache transfer) module. How are expert weights cached and moved? Does NIXL handle expert weight migration when load balancing triggers re-placement?
- **Scaling behavior** — simulate 2, 4, 8 expert parallel ranks. How does NIXL transfer overhead scale with expert count and token batch size?

**Disaggregated prefill-decode serving:**
- **vLLM NIXL connector** — `NixlConnector` in vLLM handles KV cache transfer between prefill and decode instances. Set up a disaggregated serving config (`kv_connector='nixl'`, `kv_role='kv_producer'`/`'kv_consumer'`) and profile end-to-end.
- **KV transfer latency** — measure time to move KV cache blocks from prefill GPU to decode GPU. How does this compare to just running prefill+decode on the same GPU? At what request rate does disaggregation pay off?
- **Heterogeneous TP** — NIXL supports different tensor parallelism degrees for prefill vs decode instances. Test: TP=2 prefill (compute-bound) + TP=1 decode (memory-bound). Does this improve overall throughput?
- **Layout optimization** — NIXL prefers HND (Head-Number-Data) over NHD layout for transfer performance. Measure the difference. Does layout permutation overhead negate the transfer speedup?
- **Side-channel coordination** — ZMQ-based metadata handshake between agents. Measure handshake latency and its impact on time-to-first-token for new requests.

**Comparison with alternatives:**
- **NIXL vs NCCL** — for the same all-to-all pattern, compare NIXL (point-to-point, explicit) vs NCCL (collective, optimized). When does each win?
- **NIXL vs LMCache** — vLLM supports both as KV offloading backends. Compare on the same disaggregated serving workload.
- **NIXL vs direct RDMA** — what overhead does NIXL's abstraction layer add over raw ibverbs/UCX calls?

**Key questions:**
- For a 2-GPU B200 setup, is expert parallelism via NIXL faster than just replicating experts (data parallelism) for MoE models like Mixtral or DeepSeek-V3?
- What is the minimum expert computation time for NIXL transfer latency to be hidden (overlap threshold)?
- Does NIXL's async transfer API actually enable full compute-communication overlap, or are there synchronization points that serialize the pipeline?
- For disaggregated serving: at what QPS does prefill-decode separation via NIXL outperform colocated serving?
- Can NIXL + expert parallelism + KV cache optimization (T-13) compound to serve models that wouldn't fit on a single GPU?

---

## Diffusion-Inspired Experiments

### D-1. Depth as Denoising
The residual stream h₀ → h₂₈ resembles a denoising trajectory. Test if this is more than metaphor.
- Measure cosine similarity between the layer update `h_{k+1} - h_k` and the score `∂log P(correct_token) / ∂h_k`. If layers act as denoising steps, these should align.
- Compute **SNR across depth**: project hidden states onto top singular vectors of the unembedding matrix (signal) vs complement (noise). Does SNR increase monotonically?
- Plot `‖h_{k+1} - h_k‖` across depth — diffusion models show "fast early, slow late." Does the transformer match?
- Fit a continuous-time ODE to the layer trajectory. If interpolated states match real intermediate layers, the transformer implements a smooth flow, not 28 discrete operations.

### D-2. Noise Injection & Denoising Capacity
Inject Gaussian noise at each layer, measure how well subsequent layers recover.
- For injection point k, add `h_k' = h_k + σ·ε`, run layers k+1...28, measure loss delta vs clean. Sweep σ per layer to get noise tolerance curves.
- **Per-layer denoising ratio**: `‖(h_{k+1}' - h_{k+1})‖ / ‖(h_k' - h_k)‖`. Ratio < 1 means the layer actively denoises; ratio ≈ 1 means passthrough.
- Cross-reference with T-2: are "denoising" layers the same as "critical" layers?

### D-3. Iterative Refinement via Re-execution
Feed the model's final hidden state h₂₈ back as input to layer 0 and run all layers again. Repeat N times.
- Does output quality improve with iterations? If yes, the model is a contractive map with the correct representation as its fixed point.
- **Partial loop**: re-run only the last K layers (e.g., 20-28). Best quality/compute tradeoff?
- Test on high-entropy predictions where the model is uncertain. If refinement only helps "almost right" cases, that confirms denoising interpretation.
- **Masked refinement**: zero out hidden states at specific token positions, iterate. Can the model reconstruct from context — bidirectional infilling via iterative AR passes?

### D-4. Flow Matching Between Layers
Train a lightweight flow model (~1M params) to learn v(h, t) mapping h₀ → h₂₈ as a continuous trajectory.
- Compare interpolated flow states h_t to actual layer representations h_k. If close, the transformer is a smooth flow; if far, layers make discontinuous jumps.
- **Layer skipping**: replace layers 5-20 with a single ODE step. What's the quality loss? Principled layer pruning.
- **Token-conditional flow**: train separate flows for function words vs content words. If flows differ, the model routes token types through different trajectories despite shared weights.

### D-5. AR Model as Discrete Text Denoiser
Repurpose the autoregressive LM as a discrete diffusion denoiser — zero fine-tuning.
- Corrupt a clean sequence by replacing X% of tokens with random vocabulary items. Feed to the model. Does the next-token prediction at each corrupted position recover the original token?
- Sweep corruption rate 5% → 80%. Find the breakdown threshold — measures the model's implicit clean-text prior.
- **Iterative denoising loop**: corrupt → predict → replace with argmax → repeat. Does it converge to the original text? How many steps? This is a zero-shot discrete diffusion model.
- Use logit lens (T-1) to track at which layer the model "notices" corruption — corrupted tokens should crystallize later than clean ones.

### D-6. Textual Diffusion from Scratch
Train a small discrete diffusion model for text and compare its internals to the AR transformer.
- Implement MDLM (masked discrete language model): train a BERT-sized model to predict masked tokens with a continuous-time noise schedule on the same data distribution.
- **Internal comparison**: do the diffusion model's layers show the same depth-as-denoising signature (D-1)? Is the SNR profile similar or fundamentally different from the AR model?
- **Representation alignment**: CKA or linear probe comparison between AR layer k and diffusion model layer k. Do they learn similar representations despite opposite training objectives (left-to-right vs denoise-from-anywhere)?
- **Hybrid generation**: use the AR model to draft a sequence, add noise, then use the diffusion model to refine. Does this beat either model alone? Tests complementarity of the two paradigms.
- **Noise schedule probing**: the diffusion model explicitly has a noise schedule σ(t). The AR model implicitly has one (D-1). Compare them — does the AR model's implicit schedule match common diffusion schedules (linear, cosine, learned)?

---

## Vision-Language Model Experiments (Qwen3-VL-2B-Instruct)

### VL-1. Modality Gap Across Depth
After the vision encoder, image tokens and text tokens share the same residual stream. But are they actually in the same representational space?
- At each layer, collect hidden states for image tokens vs text tokens.
- Measure the geometric gap: distance between centroids, overlap of principal subspaces (subspace angle), and whether a linear probe can classify "image token vs text token."
- Hypothesis: gap is large at early layers (vision encoder output is alien to the LM) and shrinks across depth as the model "assimilates" visual information into its language manifold.
- **Key question**: does the gap ever reach zero? If not, the model maintains a persistent modality-specific subspace — visual information is never fully "translated" into language.
- Plot: layer vs classification accuracy of modality probe. A sharp drop at layer k means that's where fusion truly happens.

### VL-2. Visual Token Redundancy & Information Compression
VLMs produce hundreds of visual tokens from a single image. How many actually carry information the model uses?
- Progressively mask/drop visual tokens (random, spatial blocks, low-attention) and measure answer accuracy on VQA benchmarks.
- Find the **Pareto frontier**: minimum visual tokens needed for X% accuracy retention.
- Measure per-token "influence": for each visual token, compute gradient of answer logit w.r.t. that token's embedding. Rank by influence.
- Hypothesis: >50% of visual tokens are redundant for most questions. The model only "reads" a small spatial region.
- Visualize: overlay influence heatmap on the original image — does the model attend to the right region for a given question?
- Practical value: aggressive token pruning for inference speedup.

### VL-3. Hallucination Localization
When the model describes something not in the image, where does the hallucinated content originate?
- Collect prompts that reliably trigger hallucinations (e.g., ask about an object that isn't present).
- Run activation patching: replace image token activations at each layer with those from a different image. At which layer does swapping the image stop affecting the (hallucinated) output?
- If hallucinations become image-independent early (layer 5), the LM is ignoring the image and relying on language priors. If they remain image-dependent until late layers, the vision encoder is actively feeding misleading information.
- Compare residual stream projections (logit lens) at each layer for hallucinated vs faithful outputs — at which layer do they diverge?
- **Causal test**: inject the "correct" activation (from an image that does contain the object) at the divergence layer — does this fix the hallucination?
- **Adversarial modality conflict**: present an image that contradicts the text ("The sky is green" + blue sky image). Which modality wins at each layer? Is there a layer where the conflict is visible in the residual stream?

### VL-4. Vision Encoder → LM Bottleneck Analysis
The vision encoder output is projected into the LM's embedding space via a connector module. How much information survives this bottleneck?
- Train linear probes at different points: (a) raw ViT output, (b) after the connector projection, (c) after LM layer 0, layer 5, etc.
- Probe for: object presence, spatial location, color, texture, scene category, OCR content.
- Which visual properties are preserved vs lost at each stage?
- Hypothesis: fine-grained spatial information (exact coordinates) is lost early; semantic categories (object identity) persist deep.
- Compare probe accuracy before/after the connector — if there's a big drop, the connector is the bottleneck. If not, the LM layers are responsible for information loss.

### VL-5. Vision Token Representation Decoding
What does the model "see" at each layer? Decode visual representations back into interpretable form.
- At each layer, take the hidden states of image tokens and project them through a trained decoder (small CNN or diffusion decoder) back to pixel space.
- How deep into the LM can you still reconstruct the image? When does the representation become purely semantic?
- Alternative: at each layer, train a linear probe to predict the original ViT patch embeddings from the LM hidden states. Measures how much raw visual information is retained.
- **Logit lens for vision**: project image token hidden states through the LM head — what text tokens do they most resemble at each layer? Early layers might map to visual descriptors ("red," "round"), late layers to task-relevant concepts.

### VL-6. Modality-Specific Layer Criticality
Extension of layer knockout (T-2) to the multimodal setting.
- Knock out each layer and measure impact on: (a) text-only tasks, (b) VQA accuracy, (c) image captioning quality.
- Are there layers that are critical for vision but not language, or vice versa?
- Hypothesis: some layers specialize in cross-modal fusion and are critical for VQA but irrelevant for text-only. Others are "language backbone" layers.
- If confirmed, this enables modality-specific pruning: for text-only inference, skip the vision-critical layers for free speedup.
- Double knockout: remove pairs of layers and look for synergistic effects (two individually unimportant layers that are jointly critical = distributed circuit).

### VL-7. Cross-Modal Feature Interference
Does processing images and text in the same residual stream cause interference?
- Compare: model performance on text-only tasks with vs without a (irrelevant) image prepended.
- Measure hidden state perturbation: how much do text token representations change when an image is present vs absent?
- **Feature competition**: find dimensions in the residual stream that are used by both image tokens and text tokens for different purposes. These shared dimensions are interference bottlenecks.
- If interference is significant, it explains why VLMs often underperform text-only models on pure language tasks — the shared residual stream is a resource contention problem.

### VL-8. Layer Shuffle Recovery on VLM
Run the existing shuffle experiment on Qwen3-VL-2B-Instruct. Do image-processing layers resist shuffling differently?
- Apply the same methodology: shuffle LM decoder layers, recover order using math-only and dataset-based methods.
- **Key question**: does the activation_flow method still work? VLM layers may have more heterogeneous fingerprints due to cross-modal processing.
- Compare swap cost between layers that handle image tokens vs text-only layers — are vision-critical layers more position-sensitive?
- If the VLM has a mix of attention types (full attention + linear attention), does layer type predict recoverability?
- Calibrate with image+text prompts vs text-only prompts — does the calibration modality affect which layers are recoverable?
