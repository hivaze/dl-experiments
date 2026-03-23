# DL Experiments

## Project Overview
Deep learning research experiments repository. Focus on investigating transformer model internals, weight analysis, and layer-level properties.

## Environment Setup
- **ALWAYS use poetry env** — never use raw `python` or `pip`. All commands must go through `poetry run`.
- **Run scripts**: `poetry run python <script>` from `/mnt/dl-experiments`
- **Install packages**: `poetry run pip install <package>` or `poetry add <package>` from `/mnt/dl-experiments`
- **GPUs**: 2x NVIDIA B200 (183GB each), use `cuda:0` or `cuda:1`

## Key Dependencies
- torch 2.10+ (CUDA 12.8)
- transformers 5.3.x
- vllm 0.18.x
- datasets 4.8+
- accelerate 1.13+
- scipy 1.17+, scikit-learn 1.8+, numpy
- matplotlib, seaborn, plotly for visualization

## Project Structure
```
experiments/           # Each experiment in its own subfolder
  <experiment>/
    run.py             # Main entry point
    README.md          # Experiment description
    results/           # Outputs (JSON, plots, logs) — per-experiment
    *.py               # Supporting scripts (data generation, etc.)
configs/               # Shared configuration files
data/                  # Organized by data type / experiment group
  text_completions/    # Greedy completions for text experiments (T-1..T-15)
    prompts.json       # Shared evaluation prompts
    generate_completions.py  # vLLM generator (CLI: --model, --max-tokens)
    <model-slug>/
      completions.json # Prompt+completion pairs with token counts
models/                # Model checkpoints and saved weights
notebooks/             # Jupyter notebooks for exploration
utils/                 # Shared utility functions
```

## Experiment Structure

Each experiment lives in `experiments/<name>/` and must include a comprehensive `README.md` that serves as the primary record of the investigation. The README should contain:

1. **Motivation & Research Question** — What are we investigating and why? Frame as a clear question.
2. **Setup** — Model(s) used, configuration, hardware, any calibration data or preprocessing.
3. **Methods** — Detailed description of each approach/technique tried, including algorithms, key implementation choices, and rationale.
4. **Results** — Quantitative results with tables, metrics, and visualizations. Include per-method breakdowns where applicable.
5. **Conclusions & Key Findings** — What did we learn? What worked, what didn't, and why? Actionable insights for future experiments.
6. **Usage** — How to reproduce (`poetry run python experiments/<name>/run.py`), any prerequisites (data generation, model downloads).

The README is the authoritative document for the experiment — anyone reading it should understand the full investigation without needing to read the code. Results in `results/` (JSON, plots) are supporting artifacts; the README interprets them.

## Code Requirements
- Use `torch.bfloat16` for model loading (B200 native format)
- Always set `use_cache=False` when doing non-standard forward passes
- Results go in `experiments/<name>/results/`, not the top-level `results/`
- Use `CUDA_VISIBLE_DEVICES` or `device_map` for GPU selection
- Scripts should be self-contained with clear section headers
- Include timing for each method/step
- Report quantitative metrics (not just qualitative)
- Set random seeds for reproducibility
- Use `poetry run python` to execute, never raw `python`

## Model Notes
- Qwen3-4B-Instruct-2507 (`Qwen/Qwen3-4B-Instruct-2507`): 36 homogeneous decoder layers, Qwen3RMSNorm, GQA (32q/8kv heads), SwiGLU MLP (hidden=2560, intermediate=6144). Primary model for text experiments (T-1 through T-3).
- Qwen3-4B-Thinking-2507 (`Qwen/Qwen3-4B-Thinking-2507`): Same architecture as Instruct, with extended thinking/reasoning training. Previously used for T-1..T-3.
- Qwen3-1.7B (`Qwen/Qwen3-1.7B`): 28 homogeneous decoder layers, RMSNorm, GQA (16q/8kv heads), SwiGLU MLP (hidden=4096, intermediate=6144), ~3.4GB in bf16
- Qwen3-VL-2B-Instruct (`Qwen/Qwen3-VL-2B-Instruct`): Vision-language model with ViT encoder + connector + LM decoder, used for VL-* experiments
- Qwen3.5-2B (`Qwen/Qwen3.5-2B`): Hybrid with heterogeneous layers (mix of Gated DeltaNet + full attention, 24 layers) - NOT a standard transformer. Requires transformers 5.3+.
- **Chat templates**: Always use `tokenizer.apply_chat_template()` for instruct/thinking models, never raw `tokenizer()`. For Qwen3 thinking models the template adds `<think>` tags.
- When shuffling layers, `layer_idx` in attention modules becomes stale - must use `use_cache=False`

## Experiment Tracking
- Research agenda in `TODO.md` — T-1..T-17, D-1..D-6, VL-1..VL-8
- Completed text experiments: T-1 Logit Lens, T-2 Layer Knockout, T-3 Layer Swap Cost, T-4 Residual Stream Geometry, T-7 Layer Linearization Gap, T-9 Weight Spectral Structure, T-17 Contrastive Completion Trajectories
- Completed standalone: Layer Shuffle Recovery (`experiments/layer_shuffle_recovery/`), Fish Speech S2 Pro (`experiments/fish_speech_s2_pro/`), ACE-Step v1.5 (`experiments/acestep_v15/`)
- Planned text: T-5 Cross-Model Layer Transplant, T-6 Layer Doubling, T-8 Thinking vs Answer Token Routing, T-10a/b Attention Architecture & Kernels, T-11 Quantization, T-12 CUDA Graphs, T-13 KV-Cache, T-14 NIXL, T-15 Normalization, T-16 Activation Functions
- Planned diffusion: D-1..D-6
- Planned vision-language: VL-1..VL-8
