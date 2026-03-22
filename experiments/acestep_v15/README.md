# ACE-Step 1.5 — Architecture Investigation

Inspect the internals of [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5), a hybrid LM + Diffusion Transformer text-to-music model. MIT licensed.

## Model Overview

**~4.4B generation params** + 600M text embedder | bfloat16 native | generates up to 10-minute songs

The core architectural insight is **separating planning from synthesis**: a standard LLM generates a structured song blueprint (metadata, lyrics, captions) via Chain-of-Thought, then a Diffusion Transformer synthesizes the actual audio conditioned on that blueprint plus lyric/timbre encodings.

## Full Pipeline

```
Text prompt + Lyrics
       │
       ▼
┌──────────────────┐     ┌──────────────────┐
│  Qwen3-Embed-0.6B │────▶│  Text Projector   │──┐
│  (1024-dim embs)  │     │  [2048, 1024]     │  │
└──────────────────┘     └──────────────────┘  │
                                                │
┌──────────────────┐                            ├──▶ Cross-Attention
│  Lyric Encoder    │──── 8 bidir layers ───────┤    Conditioning
│  (405M params)    │                            │    Sequence
└──────────────────┘                            │
                                                │
┌──────────────────┐                            │
│  Timbre Encoder   │──── 4 bidir layers ───────┘
│  (202M, ref audio)│
└──────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  LM Planner       │────▶│  DiT (24 layers)  │────▶│  VAE Decoder  │──▶ Audio
│  Qwen3-1.7B       │     │  Flow Matching    │     │  Oobleck      │    48kHz
│  (song blueprint) │     │  8 turbo steps    │     │  (84M)        │    stereo
└──────────────────┘     └──────────────────┘     └──────────────┘
```

## Architecture Details

### LM Planner — 1,854M params

A fine-tuned Qwen3-1.7B that generates structured song blueprints via Chain-of-Thought.

| Parameter | Value |
|---|---|
| Architecture | `Qwen3Model` (standard, not CausalLM) |
| Layers | 28 (all full attention) |
| Hidden dim | 2048 |
| Attention | 16q / 8kv GQA, head_dim=128 |
| MLP | SwiGLU, intermediate=6144 |
| Vocab size | **217,204** (65K more than standard Qwen3's 151K) |
| Max position | 40,960 |
| Tied embeddings | Yes |
| Per-layer params | 50.3M |

Weight breakdown:
- `embed_tokens`: 444.8M (1 tensor) — the 217K vocab embedding is 24% of the model
- `layers`: 1,409.4M (308 tensors, 28 × 11 weights per layer)
- `norm`: 0.002M (final RMSNorm)

The extended vocab (217,204 vs Qwen3's 151,643) adds ~65K music-specific tokens for genre tags, instrument names, musical notation, and structured output formatting. Trained with SFT + **intrinsic RL** (no external reward models).

### Diffusion Transformer (DiT) — 1,576M params (decoder + time embeddings)

The audio synthesis core. A 24-layer transformer that generates audio latents via flow matching, conditioned on timestep + text/lyric/timbre encodings.

| Parameter | Value |
|---|---|
| Layers | 24 |
| Hidden dim | 2048 |
| Attention | 16q / 8kv GQA, head_dim=128 |
| MLP | SwiGLU via `Qwen3MLP` (gate/up/down), intermediate=6144 |
| Norm | RMSNorm (eps=1e-6) |
| RoPE base | 1,000,000 |
| Max position | 32,768 |
| Patch size | 2 (1D Conv, in_channels=192 → 2048) |
| Input channels | 192 (= 64 latent channels × 3: noisy + condition + mask) |

**Alternating attention pattern** — explicitly configured in `layer_types`:
```
Layer  0: sliding_attention (window=128)
Layer  1: full_attention
Layer  2: sliding_attention (window=128)
Layer  3: full_attention
...
Layer 22: sliding_attention (window=128)
Layer 23: full_attention
```
12 sliding + 12 full, strictly alternating. This Longformer-style pattern enables processing long audio sequences (10-minute songs) while maintaining global context through full-attention layers.

**Each DiT layer contains:**
- `self_attn` — self-attention (q/k/v/o projections + q_norm/k_norm)
- `cross_attn` — cross-attention to condition sequence (q/k/v/o + norms)
- `mlp` — SwiGLU (gate_proj/up_proj/down_proj)
- `self_attn_norm`, `cross_attn_norm`, `mlp_norm` — RMSNorm before each sublayer
- `scale_shift_table: [1, 6, 2048]` — **AdaLN modulation**: 6 values (scale, shift, gate for 2 of the 3 sublayers) derived from timestep embedding

Per-layer params: ~62.9M (1,510.4M / 24 layers)

**Dual timestep embedding** (the mean-flow matching mechanism):
- `time_embed`: 29.9M — embeds timestep `t` via sinusoidal → `linear_1` [2048, 256] → SiLU → `linear_2` [2048, 2048] → `time_proj` [12288, 2048]
- `time_embed_r`: 29.9M — identical architecture, embeds timestep `r`
- The `time_proj` output [12288] = 6 × 2048, providing the 6 per-layer modulation values for AdaLN
- `data_proportion=0.5`: half the time `r=t` (standard flow matching), half the time `r<t` (mean-flow)
- `timestep_mu=-0.4`, `timestep_sigma=1.0`: log-normal sampling of timesteps

**Global modulation:**
- `decoder.scale_shift_table: [1, 2, 2048]` — 2 global scale/shift values applied to the final output
- `decoder.condition_embedder: [2048, 2048]` — projects condition tokens before cross-attention

### Condition Encoder — 607M params total

Three parallel encoders whose outputs are concatenated into a single cross-attention sequence:

**Text Projector** (2.1M):
- Single linear `[2048, 1024]` — projects Qwen3-Embedding-0.6B outputs to DiT dim
- No bias

**Lyric Encoder** (404.8M):
- 8 bidirectional transformer layers, same Qwen3-block architecture
- `embed_tokens`: [2048, 1024] with bias — projects 1024-dim text embeddings
- Each layer: self-attn (q/k/v/o + norms) + SwiGLU MLP + layernorms
- Bidirectional (no causal mask) — lyrics need full context in both directions
- Final norm layer

**Timbre Encoder** (201.5M):
- 4 bidirectional transformer layers
- `embed_tokens`: [2048, 64] with bias — projects 64-dim acoustic features
- `special_token: [1, 1, 2048]` — CLS token for pooling
- `timbre_fix_frame=750` — processes up to 750 frames of reference audio (~30s at 25Hz)
- Same block architecture as lyric encoder, just fewer layers

### Audio Tokenizer — 105M params

Converts VAE latents (25Hz) to discrete FSQ tokens (5Hz) for the LM planner.

**Attention Pooler** (104.9M):
- `embed_tokens`: [2048, 2048] with bias
- 2 transformer layers (standard Qwen3 blocks with self-attention + MLP)
- `special_token: [1, 1, 2048]` — CLS token prepended to each window
- `pool_window_size=5` — processes 5 consecutive frames, outputs 1 token per window
- This is the 25Hz → 5Hz downsampling step

**FSQ Quantizer** (0.024M — tiny!):
- `project_in`: [6, 2048] — project from hidden dim to 6 scalar dimensions
- Finite Scalar Quantization at levels `[8, 8, 8, 5, 5, 5]` — round each of 6 dimensions to discrete levels
- `project_out`: [2048, 6] — project back to hidden dim
- Effective codebook size: 8×8×8×5×5×5 = **64,000** codes
- `vocab_size=64,003` (64K codes + 3 special tokens)
- **No learned codebook** — unlike VQ-VAE, FSQ uses deterministic rounding, completely avoiding codebook collapse

**Acoustic projection** (0.1M):
- `audio_acoustic_proj`: [2048, 64] with bias — projects raw acoustic features (64-dim) into hidden dim

### Audio Detokenizer — 105M params

Expands FSQ tokens back from 5Hz to 25Hz (inverse of the tokenizer's pooling step).

- `embed_tokens`: [2048, 2048] with bias
- 2 transformer layers (same architecture as attention pooler)
- `proj_out`: [64, 2048] — project back to VAE latent dim (64 channels)
- `special_tokens: [1, 5, 2048]` — 5 special tokens (one per position in the expanded window)

### VAE (AutoencoderOobleck) — 168.7M params

Continuous audio codec. Converts raw audio waveforms to/from continuous latent representations.

| Parameter | Value |
|---|---|
| Architecture | `AutoencoderOobleck` (from diffusers) |
| Encoder | 84.3M (OobleckEncoder) |
| Decoder | 84.4M (OobleckDecoder) |
| Audio channels | 2 (stereo) |
| Sampling rate | 48,000 Hz |
| Encoder hidden | 128 |
| Channel multiples | [1, 2, 4, 8, 16] |
| Downsampling ratios | [2, 4, 4, 6, 10] |
| Decoder channels | 128, input=64 |

**Verified forward pass** (10s stereo audio):
```
Input:   [1, 2, 480000]  (batch, stereo, 48kHz × 10s)
Latent:  [1, 64, 250]    (batch, channels, frames)
Decoded: [1, 2, 480000]  (perfect shape reconstruction)

Total compression: 2 × 4 × 4 × 6 × 10 = 1920×
Latent rate: 48000 / 1920 = 25.0 Hz
Latent channels: 64
```

### Text Embedder — Qwen3-Embedding-0.6B

Separate embedding model, **not** part of the main generation pipeline weights.

| Parameter | Value |
|---|---|
| Architecture | `Qwen3Model` |
| Layers | 28 (all full attention) |
| Hidden dim | 1024 |
| Intermediate | 3072 |
| Attention | 16q / 8kv GQA |
| Vocab | 151,669 (standard Qwen3) |
| Output | 1024-dim embeddings |

## Multi-Rate Tokenization Hierarchy

```
48,000 Hz  ──▶  raw stereo audio              [2 ch × 480K samples / 10s]
              │
         VAE Encode (1920× compression)
              │
   25.0 Hz  ──▶  VAE latent                    [64 ch × 250 frames / 10s]
              │
         Attention Pooler (5× pooling)
              │
    5.0 Hz  ──▶  FSQ tokens                    [2048-dim × 50 tokens / 10s]
              │
         FSQ Quantize (6 scalar dims)
              │
    5.0 Hz  ──▶  Discrete codes                [6-dim × 50 codes / 10s]
                  from {8,8,8,5,5,5} levels
```

For a 10-minute song (600s): 600 × 5 = **3,000 discrete tokens** for the LM to plan. The DiT works at the 25Hz latent level: 600 × 25 = 15,000 frames, patched to 7,500 with patch_size=2.

## Parameter Census

| Component | Params | Tensors | % of total |
|---|---|---|---|
| DiT decoder layers | 1,510.4M | 456 | 34.2% |
| LM planner (Qwen3-1.7B) | 1,854.2M | 310 | 42.0% |
| Lyric encoder | 404.8M | 91 | 9.2% |
| Timbre encoder | 201.5M | 48 | 4.6% |
| VAE | 168.7M | — | 3.8% |
| Audio tokenizer (pooler) | 104.9M | 26 | 2.4% |
| Audio detokenizer | 105.1M | 28 | 2.4% |
| DiT time embeddings (t+r) | 59.8M | 12 | 1.4% |
| DiT other (proj, norms) | 5.3M | 8 | 0.1% |
| Text projector | 2.1M | 1 | <0.1% |
| Acoustic projection | 0.1M | — | <0.01% |
| FSQ quantizer | 0.024M | 4 | <0.01% |
| **Total (excl. text embedder)** | **~4,418M** | — | — |
| Text embedder (Qwen3-0.6B) | ~600M | — | separate |

## Architectural Observations

1. **Everything is Qwen3 blocks.** The DiT layers, lyric encoder, timbre encoder, tokenizer, and detokenizer all use the same transformer block: GQA (16q/8kv), SwiGLU MLP (6144 intermediate), QK normalization, RMSNorm. The only additions for the DiT are cross-attention and AdaLN. This means the entire system can be served with a single optimized kernel.

2. **The FSQ quantizer is almost nothing.** Just 24K params — two linear projections. The real compression work is done by the attention pooler (105M) and the VAE (169M). FSQ's role is purely discretization with zero learned parameters in the quantization step itself.

3. **The LM planner is the largest single component** (42% of total params). It has 65K extra vocab tokens beyond standard Qwen3 — these are music-specific tokens for genre, instruments, structure markers, and CoT formatting.

4. **Conditioning is expensive.** The lyric + timbre encoders together (607M) are nearly as large as the text embedder (600M). The lyric encoder alone (8 layers, 405M) is bigger than many standalone language models.

5. **Dual timestep is architecturally visible.** The `time_embed` and `time_embed_r` are completely separate networks (30M each) with identical architecture but independent weights. The `time_proj` layer in each produces [12288] = 6 × 2048, giving per-layer AdaLN modulation. The `data_proportion=0.5` parameter controls how often `r=t` vs `r<t` during training — a clean implementation of mean-flow matching.

6. **The attention pooler is an unusual tokenizer.** Rather than a learned codebook or simple strided convolution, it uses 2 full transformer layers with a CLS token to aggregate every 5 frames. This is computationally expensive but allows the pooling to be context-dependent.

## Running

```bash
poetry run python experiments/acestep_v15/run.py
```

Results saved to `results/`: configs.json, summary.json.

## References

- HuggingFace: https://huggingface.co/ACE-Step/Ace-Step1.5
- Paper: arXiv:2602.00744
- GitHub: https://github.com/ACE-Step/ACE-Step-1.5
