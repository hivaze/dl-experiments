# Fish Speech S2 Pro — Architecture Investigation

Inspect the internals of [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro), a 4.56B-parameter Dual-AR text-to-speech model built on a Qwen3-style backbone.

## Model Overview

**4.56B total params** | 9.12 GB in bf16 | 358 weight tensors across 2 safetensor shards

Model type `fish_qwen3_omni` — a custom architecture not registered in HuggingFace transformers (requires the `fish-speech` package for inference). We inspected it via direct safetensors analysis.

The core idea is **Dual-AR factorization**: one large autoregressive model generates semantic audio tokens along the time axis, then a smaller model fills in acoustic detail across codebooks at each timestep.

## Pipeline

```
Text Input
     │
     ▼
┌───────────────────────────────┐
│  Slow AR — 36L Qwen3 (4.0B)  │  autoregressive over time
│  unified text + audio vocab   │  155,776 tokens
└───────────┬───────────────────┘
            │ semantic tokens + hidden states
            ▼
┌───────────────────────────────┐
│  Fast AR — 4L decoder (530M)  │  autoregressive over codebooks
│  max_seq_len = 11             │  fills 9 residual codebooks
└───────────┬───────────────────┘
            │ 10 codes per timestep
            ▼
┌───────────────────────────────┐
│  Codec — transformer (695M)   │  16-layer quantizer
│  codes → waveform             │  semantic + acoustic VQ
└───────────┬───────────────────┘
            ▼
       Audio Output
```

## Architecture Details

### Slow AR (text_model) — 4,032M params (88.4%)

The main generation backbone. Structurally isomorphic to a standard Qwen3 decoder-only LLM.

| Parameter | Value |
|---|---|
| Layers | 36 |
| Hidden dim | 2560 |
| Attention | 32q / 8kv GQA (head_dim=128) |
| MLP | SwiGLU, intermediate=9728 (w1/w2/w3) |
| QK normalization | Yes |
| RoPE base | 1,000,000 |
| Max sequence length | 32,768 |
| Vocab size | 155,776 |
| Tied embeddings | Yes |
| Norm | RMSNorm (eps=1e-6) |
| No biases | attention Q/K/V/O all bias-free |

Per-layer breakdown (each layer = 100.9M params):
- `attention.wqkv`: [6144, 2560] — packed Q/K/V projection (15.7M)
- `attention.wo`: [2560, 4096] — output projection (10.5M)
- `attention.q_norm`, `attention.k_norm`: [128] each — QK normalization
- `feed_forward.w1`: [9728, 2560] — gate projection (24.9M)
- `feed_forward.w2`: [2560, 9728] — down projection (24.9M)
- `feed_forward.w3`: [9728, 2560] — up projection (24.9M)
- `attention_norm`, `ffn_norm`: [2560] each — RMSNorm

Component breakdown: MLP 66.7%, attention 23.4%, embedding 9.9%, norms <0.1%

**MoE scaffolding present but disabled**: config includes `use_moe=False`, `num_experts=1`, `moe_intermediate_size=768`, `router_gamma=0.001`. The architecture supports MoE but S2 Pro uses dense layers.

### Fast AR (audio_decoder) — 530M params (11.6%)

Generates the 9 residual codebooks for acoustic detail at each timestep. A key design insight: it operates **across codebooks, not across time** — its max_seq_len is just 11 (10 codebooks + 1).

| Parameter | Value |
|---|---|
| Layers | 4 |
| Hidden dim | 2560 |
| Audio hidden dim | 5120 |
| Attention | 32q / 8kv GQA (head_dim=128) |
| MLP | SwiGLU, intermediate=9728 |
| QK normalization | **No** (unlike Slow AR) |
| Max sequence length | **11** |
| Num codebooks | 10 |
| Vocab size | 4096 (codebook token space) |
| Tied embeddings | No |
| Text dim | 2560 (receives Slow AR hidden states) |

Each of the 4 layers is also 100.9M — identical block design to Slow AR. Plus:
- `codebook_embeddings`: [40960, 2560] (10 codebooks × 4096 entries = 104.9M)
- `embeddings`: [4096, 2560] (token embeddings, 10.5M)
- `output`: [4096, 2560] (output head, 10.5M)

### Audio Codec (codec.pth) — 695M params

Separate from the main model. A sophisticated neural codec, **not** a simple VQ-VAE:

| Component | Params | Details |
|---|---|---|
| Encoder | 346.3M | Conv-based with weight normalization, parametrized weights |
| Decoder | 54.1M | Conv-based, mirrors encoder |
| Quantizer | 294.6M | **Transformer-based** (pre + post modules) |

The quantizer is the most interesting part:
- `pre_module`: 8 transformer layers (dim=1024, SwiGLU, causal attention with precomputed `freqs_cis` and `causal_mask` [4096×4096])
- `post_module`: 8 transformer layers (same architecture)
- Both modules use **LayerScale** (`attention_layer_scale.gamma`, `ffn_layer_scale.gamma`) — learnable per-layer residual scaling
- `semantic_quantizer`: 1 codebook, **4096 entries** of dim 8 (the primary semantic codes the Slow AR predicts)
- `quantizer`: **9 residual codebooks**, 1024 entries of dim 8 each (the acoustic codes the Fast AR generates)
- Codebook projections use weight normalization (`weight_g` / `weight_v` parametrization)
- 2 ConvNeXt downsample blocks + 2 ConvNeXt upsample blocks (depthwise conv + pointwise conv + LayerNorm)

This means the codec itself is a 16-layer transformer that converts between audio waveforms and discrete tokens, with semantic and acoustic quantization levels.

### Tokenizer / Token Space

| Range | IDs | Count | Purpose |
|---|---|---|---|
| Base Qwen3 | 0 – 151,642 | 151,643 | Standard text tokens |
| `<|im_start|>` / `<|im_end|>` | 151,644 – 151,645 | 2 | Chat turn markers |
| `<|pad|>` | 151,669 | 1 | Padding |
| `<|phoneme_start|>` / `<|phoneme_end|>` | 151,670 – 151,671 | 2 | Phoneme boundaries |
| `<|text|>`, `<|voice|>`, `<|interleave|>` | 151,672 – 151,674 | 3 | Mode markers |
| `<|audio_start|>` / `<|audio_end|>` / `<|audio_pad|>` | 151,675 – 151,677 | 3 | Audio stream delimiters |
| **`<|semantic:0|>`** through **`<|semantic:4095|>`** | 151,678 – 155,773 | **4,096** | Audio semantic tokens |

Total vocab: 155,776. The model operates in a unified text-audio token space — a single sequence can contain text tokens, control tokens, and audio semantic tokens interleaved.

## Weight Statistics

### Slow AR — Frobenius Norms Across Depth

Three distinct phases visible in the layer norms:

```
Layer  0:  F=146.4  (initialization effects — higher than neighbors)
Layer  1:  F=107.8  (dip)
Layers 2-4: F=103–141  (ramp-up)
Layers 5-21: F=150–159  (plateau, stable core)
Layers 22-35: F=156–164  (slight upward trend, final layers)
```

**Spectral norm (σ₁)** shows the opposite pattern — it rises at the edges:
- Layer 0: σ₁=18.8, Layer 7: σ₁=13.8 (minimum), Layer 35: σ₁=24.5 (maximum)
- This suggests early/late layers have more "spiky" singular value distributions

**Effective rank** (exponential of spectral entropy) increases with depth:
- Early layers (0-4): eff_rank ≈ 1955–2113
- Middle layers (5-21): eff_rank ≈ 2096–2160
- Late layers (22-35): eff_rank ≈ 2124–2230

Higher effective rank in later layers suggests they use more of their representational capacity (more distributed singular values, less low-rank structure).

### Fast AR — Weight Statistics

| Layer | mean ‖W‖_F | mean σ₁ | mean eff_rank |
|---|---|---|---|
| 0 | 111.5 | 17.2 | 2127.5 |
| 1 | 114.3 | 12.0 | 2200.2 |
| 2 | 109.8 | 9.3 | 2126.8 |
| **3** | **92.5** | **16.5** | **1943.6** |

Layer 3 (the final layer) stands out: 17% lower Frobenius norm and lower effective rank, with a spike in spectral norm. This is the layer closest to the output head — it may be more specialized/low-rank.

### Embedding Statistics

| Weight | Shape | ‖W‖_F | σ₁ | eff_rank |
|---|---|---|---|---|
| text_model embeddings | [155776, 2560] | 1123.8 | 823.7 | — |
| codebook embeddings | [40960, 2560] | 449.4 | 175.3 | — |
| audio token embeddings | [4096, 2560] | 170.3 | 27.7 | 1682.0 |

The text embedding matrix has an extremely dominant first singular value (σ₁=823.7, 73% of Frobenius norm), suggesting a strong mean direction in the embedding space. The audio token embeddings are much more distributed (σ₁/‖W‖_F = 0.16).

### Selected Layer Details

Layer 0 (first), 17 (middle), 35 (last) of Slow AR:

| Layer | Weight | ‖W‖_F | σ₁ | eff_rank | top1_sv |
|---|---|---|---|---|---|
| 0 | attention.wqkv | 129.3 | 25.1 | 2169.8 | 0.0045 |
| 0 | attention.wo | 94.9 | 17.1 | 1911.2 | 0.0046 |
| 17 | attention.wqkv | 138.9 | 17.9 | 2176.6 | 0.0030 |
| 17 | attention.wo | 121.2 | 8.6 | 2109.4 | 0.0017 |
| 35 | attention.wqkv | 118.2 | 17.7 | 2182.8 | 0.0035 |
| 35 | attention.wo | 116.3 | 22.7 | 2065.5 | 0.0048 |

The `wo` (output projection) in middle layers has notably lower spectral norm (8.6 at L17 vs 17.1/22.7 at L0/L35), indicating the middle layers produce less "peaky" attention outputs.

## Running

```bash
poetry run python experiments/fish_speech_s2_pro/run.py
```

Results saved to `results/`: config.json, tensor_map.json, weight_statistics.json.

## References

- HuggingFace: https://huggingface.co/fishaudio/s2-pro
- Paper: arXiv:2603.08823
- GitHub: https://github.com/fishaudio/fish-speech
