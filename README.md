# DL Experiments

A collection of deep learning research experiments focused on transformer model internals, weight analysis, and layer-level properties.

## Experiments

### Completed

**[T-1: Logit Lens](experiments/t1_logit_lens/)** — Project the residual stream at each layer through the LM head to track how predictions evolve across depth. Evaluated on 50 prompts (4,094 completion tokens). Reveals a four-phase architecture: representation building (L0–12), early semantics (L13–21), prediction formation (L22–28), and refinement (L29–35). Mean crystallization at layer 25.4, with a 2.4-layer gap to first top-1. Model: Qwen3-4B-Instruct-2507.

**[T-2: Layer Knockout](experiments/t2_layer_knockout/)** — Skip each layer and measure loss on completions. Every layer is critical (min 2.4x loss increase). Layer 0 is catastrophic (101x), Layer 6 is the critical hub (23x, appears in 4/5 top synergistic pairs). Includes activation patching for causal bottleneck analysis. Model: Qwen3-4B-Instruct-2507.

**[T-3: Layer Swap Cost](experiments/t3_layer_swap_cost/)** — Swap every pair of layers and measure loss. Zero interchangeable pairs when evaluated on completions. Same 3-zone clustering (early/middle/late) but no layer is freely relocatable. Layer 2 is the most position-sensitive. Model: Qwen3-4B-Instruct-2507.

**[T-4: Residual Stream Geometry](experiments/t4_residual_stream_geometry/)** — Track hidden-state geometry across all 36 layers via participation ratio, isotropy, norms, and category clustering. Reveals bimodal dimensionality collapse (PR=1.6 at layer 16), superlinear norm growth (1→568), and that all anisotropy is mean-direction only (centered cosine ≈ 0 everywhere). Final layer acts as a "de-anisotropifier" — norms drop, isotropy spikes, category separation jumps to 1.03. Model: Qwen3-4B-Instruct-2507.

**[T-7: Layer Linearization Gap](experiments/t7_layer_linearization_gap/)** — Measure how nonlinear each layer's computation is on real inputs via JVP-based perturbation analysis. Reveals a U-shaped nonlinearity profile: middle layers (6–18) are most linear (gap ~0.13, 54% less nonlinear than early layers), with higher nonlinearity at both ends. Layer 0 is qualitatively different (15x transform magnitude, spectral norm 5.4). Attention and MLP nonlinearity nearly identical on average (0.129 vs 0.127). Sub-quadratic nonlinearity everywhere (order 0.6–0.8). Model: Qwen3-4B-Instruct-2507.

**[T-9: Weight Spectral Structure](experiments/t9_weight_spectral_structure/)** — SVD analysis of all 252 weight matrices (7 types x 36 layers). Q/K routing matrices are dramatically lower-rank (0.25–0.38 effective rank ratio) than V/O value processing (0.52) and MLP (0.50–0.68), confirming "where to attend" is simpler than "what to extract." Q_proj rank jumps 36.7% at layer 24→25 (discrete transition in routing complexity). Layer 1 MLP is degenerate (gate/down eff rank ~0.12–0.13). Late-layer MLP compression in layers 34–35. Model: Qwen3-4B-Instruct-2507.

**[T-17: Contrastive Completion Trajectories](experiments/t17_contrastive_trajectories/)** — Force-decode semantically related completions (synonyms, antonyms, style variants, unrelated) and compare hidden-state trajectories layer-by-layer. Discovers a meaning-vs-form crossover at layer ~18: synonyms are closer than antonyms in early layers (shared meaning) but diverge more in late layers (different surface forms). Context dominates token identity in the residual stream (antonym cosine > 0.72 across layers 2–34). Layer 35 universally destroys inter-completion similarity. KL divergence follows a U-shape for all types except antonyms. 50 hand-crafted contrastive groups, 4 relationship types. Model: Qwen3-4B-Instruct-2507.

**[Layer Shuffle Recovery](experiments/layer_shuffle_recovery/)** — Shuffle all 28 layers of Qwen3-1.7B and test 13 recovery methods. Best pipeline achieves perfect recovery (100% accuracy) in ~19 seconds.

**[Fish Speech S2 Pro](experiments/fish_speech_s2_pro/)** — Architecture investigation of the Fish Speech S2 Pro TTS model.

**[ACE-Step v1.5](experiments/acestep_v15/)** — Architecture investigation of the ACE-Step 1.5 music generation model.

### Planned

See [TODO.md](TODO.md) for the full research agenda:

- **T-5, T-6, T-8**: Architecture surgery — cross-model layer transplant, layer doubling/iteration, thinking vs answer token routing.
- **T-10a/b**: Attention architecture survey & kernel benchmarks — comparative study of MHA, GQA, MLA, DeltaNet, Mamba2, RWKV-7, sparse attention, and hybrid designs; GPU kernel microbenchmarks (FA2/3/4, FlashInfer, Triton, SageAttention3).
- **T-11 to T-14**: Inference & systems — quantization methods, CUDA graphs & torch.compile, KV-cache optimization, NIXL & disaggregated inference.
- **T-15, T-16**: Component analysis — normalization layer analysis & replacement, activation function survey & ablation.
- **D-1 to D-6**: Diffusion-inspired experiments — depth-as-denoising, noise injection/recovery, iterative refinement, flow matching, AR as discrete denoiser, textual diffusion from scratch.
- **VL-1 to VL-8**: Vision-language model experiments (Qwen3-VL-2B-Instruct) — modality gap, visual token redundancy, hallucination localization, bottleneck analysis, representation decoding, modality-specific criticality, cross-modal interference, VLM layer shuffle.

## Evaluation Data

Experiments T-1 through T-4, T-7, and T-9 use pre-generated greedy completions as evaluation data. T-17 uses hand-crafted contrastive pairs (`data/text_completions/contrastive_pairs.json`).
- Prompts: 50 question/instruction-format prompts across 7 categories (factual, reasoning, linguistic, code, world knowledge, technical, rare)
- Completions generated via vLLM (temp=0, max 2048 tokens) with system message to prevent echo
- Loss computed only on completion tokens, not prompt/template tokens

Generate completions for a new model:
```bash
poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Instruct-2507
```

## Setup

```bash
poetry install --no-root
```

### Requirements

- Python 3.11-3.12
- CUDA-capable GPU (tested on 2x NVIDIA B200, 183GB each)
- Poetry for dependency management

### Key dependencies

- PyTorch 2.10+ (CUDA 12.8)
- Transformers 5.3.x
- vLLM 0.18.x
- scipy, scikit-learn, numpy, datasets

## Project structure

```
experiments/            # Each experiment in its own subfolder
  <experiment>/
    run.py              # Main entry point
    README.md           # Full write-up: motivation, methods, results, conclusions
    results/            # Outputs (JSON, plots, logs)
    *.py                # Supporting scripts
data/
  text_completions/     # Greedy completions for text experiments
    prompts.json        # Shared evaluation prompts (50 prompts, 7 categories)
    generate_completions.py  # vLLM generator CLI
    <model-slug>/
      completions.json  # Prompt+completion pairs with token counts
configs/                # Shared configuration files
models/                 # Model checkpoints and saved weights
notebooks/              # Jupyter notebooks for exploration
utils/                  # Shared utility functions
```

Each experiment README is the authoritative record of the investigation — research question, setup, methods, quantitative results, and conclusions.
