"""
Fish Speech S2 Pro — Architecture Investigation
=================================================
Inspect fishaudio/s2-pro internals via direct safetensors + config analysis:
  - Model architecture (Slow AR + Fast AR + Codec)
  - Layer shapes, parameter counts per component
  - Weight statistics (norms, spectral properties)
  - Tokenizer inspection
  - Basic forward pass test

Model: fishaudio/s2-pro (4.56B params, Dual-AR TTS)
  - Slow AR (~4B): 36 Qwen3-style layers, dim=2560, 32q/8kv GQA, SwiGLU
  - Fast AR (~400M): 4 layers, max_seq_len=11 (cross-codebook, not cross-time)
  - Audio codec: RVQ with 10 codebooks @ ~21Hz

Note: This model uses custom model_type 'fish_qwen3_omni' which is NOT
registered in HuggingFace transformers. We inspect weights directly via
safetensors rather than loading through AutoModel.
"""

import json
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "fishaudio/s2-pro"
DEVICE = "cuda:0"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================================
# 1. Load config and inspect architecture
# ============================================================================

def inspect_config():
    """Download and inspect model config."""
    print("=" * 70)
    print("1. MODEL CONFIGURATION")
    print("=" * 70)

    config_path = hf_hub_download(MODEL_NAME, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"\nModel type: {config['model_type']}")

    # Text model (Slow AR)
    tc = config["text_config"]
    print(f"\n--- Slow AR (text_model / text_config) ---")
    print(f"  Model type:       {tc['model_type']}")
    print(f"  Layers:           {tc['n_layer']}")
    print(f"  Hidden dim:       {tc['dim']}")
    print(f"  Attention heads:  {tc['n_head']}q / {tc['n_local_heads']}kv (GQA)")
    print(f"  Head dim:         {tc['head_dim']}")
    print(f"  Intermediate:     {tc['intermediate_size']} (SwiGLU)")
    print(f"  QK norm:          {tc['attention_qk_norm']}")
    print(f"  Max seq len:      {tc['max_seq_len']}")
    print(f"  Vocab size:       {tc['vocab_size']}")
    print(f"  Tied embeddings:  {tc['tie_word_embeddings']}")
    print(f"  RoPE base:        {tc['rope_base']}")
    print(f"  MoE:              {tc['use_moe']} (experts={tc['num_experts']})")

    # Audio decoder (Fast AR)
    ac = config["audio_decoder_config"]
    print(f"\n--- Fast AR (audio_decoder / audio_decoder_config) ---")
    print(f"  Model type:       {ac['model_type']}")
    print(f"  Layers:           {ac['n_layer']}")
    print(f"  Hidden dim:       {ac['dim']}")
    print(f"  Audio hidden dim: {ac['audio_hidden_dim']}")
    print(f"  Attention heads:  {ac['n_head']}q / {ac['n_local_heads']}kv (GQA)")
    print(f"  Head dim:         {ac['head_dim']}")
    print(f"  Intermediate:     {ac['intermediate_size']} (SwiGLU)")
    print(f"  QK norm:          {ac['attention_qk_norm']}")
    print(f"  Max seq len:      {ac['max_seq_len']} (cross-codebook only!)")
    print(f"  Num codebooks:    {ac['num_codebooks']}")
    print(f"  Vocab size:       {ac['vocab_size']} (codebook tokens)")
    print(f"  Text dim:         {ac['text_dim']}")

    # Special token ranges
    print(f"\n--- Special token IDs ---")
    print(f"  semantic_start:   {config['semantic_start_token_id']}")
    print(f"  semantic_end:     {config['semantic_end_token_id']}")
    print(f"  audio_pad:        {config['audio_pad_token_id']}")
    print(f"  Audio token range: {config['semantic_start_token_id']} - {config['semantic_end_token_id']}")
    print(f"  Num audio tokens:  {config['semantic_end_token_id'] - config['semantic_start_token_id']}")

    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return config


# ============================================================================
# 2. Download and inspect weight tensors
# ============================================================================

def inspect_weights():
    """Download safetensors and inspect all weight shapes."""
    print("\n" + "=" * 70)
    print("2. WEIGHT TENSOR INSPECTION")
    print("=" * 70)

    # Download index and both shards
    index_path = hf_hub_download(MODEL_NAME, filename="model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    print(f"\nTotal size (from index): {index['metadata']['total_size']:,} bytes "
          f"({index['metadata']['total_size'] / 1e9:.2f} GB)")

    # Get unique shard files
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Shard files: {shard_files}")

    # Download all shards
    shard_paths = {}
    for shard in shard_files:
        print(f"  Downloading {shard}...")
        shard_paths[shard] = hf_hub_download(MODEL_NAME, filename=shard)

    # Inspect all tensors
    all_tensors = {}  # name -> (shape, dtype, numel, shard)
    total_params = 0

    for shard, path in shard_paths.items():
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                all_tensors[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": tensor.numel(),
                    "shard": shard,
                }
                total_params += tensor.numel()

    print(f"\nTotal tensors: {len(all_tensors)}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e9:.2f}B)")

    # Group by component (text_model vs audio_decoder vs other)
    components = defaultdict(lambda: {"params": 0, "tensors": 0, "keys": []})
    for key, info in all_tensors.items():
        parts = key.split(".")
        if parts[0] in ("text_model", "audio_decoder"):
            comp = parts[0]
        else:
            comp = "other"
        components[comp]["params"] += info["numel"]
        components[comp]["tensors"] += 1
        components[comp]["keys"].append(key)

    print(f"\nParameter breakdown by component:")
    for comp in sorted(components.keys()):
        v = components[comp]
        pct = 100.0 * v["params"] / total_params
        print(f"  {comp}: {v['params'] / 1e6:.1f}M params ({pct:.1f}%), {v['tensors']} tensors")

    # Show unique weight patterns (collapse layer indices)
    print(f"\nUnique weight name patterns:")
    patterns = defaultdict(list)
    for key, info in all_tensors.items():
        pattern = re.sub(r'\.\d+\.', '.N.', key)
        patterns[pattern].append(info)

    for pattern in sorted(patterns.keys()):
        infos = patterns[pattern]
        shape = infos[0]["shape"]
        count = len(infos)
        suffix = f" (x{count})" if count > 1 else ""
        print(f"  {pattern}: {shape}{suffix}")

    with open(RESULTS_DIR / "tensor_map.json", "w") as f:
        json.dump(all_tensors, f, indent=2)

    return all_tensors, shard_paths


# ============================================================================
# 3. Parameter census by layer type
# ============================================================================

def parameter_census(all_tensors):
    """Detailed parameter breakdown by layer type within each component."""
    print("\n" + "=" * 70)
    print("3. PARAMETER CENSUS BY LAYER TYPE")
    print("=" * 70)

    # For each component, group by weight type (attention, mlp, norm, embed)
    for component in ["text_model", "audio_decoder"]:
        comp_tensors = {k: v for k, v in all_tensors.items() if k.startswith(component)}
        if not comp_tensors:
            continue

        print(f"\n--- {component} ---")
        type_groups = defaultdict(lambda: {"params": 0, "count": 0})

        for key, info in comp_tensors.items():
            if "wqkv" in key or "wo" in key:
                wtype = "attention"
            elif "w1" in key or "w2" in key or "w3" in key:
                wtype = "mlp"
            elif "norm" in key:
                wtype = "norm"
            elif "embed" in key or "tok_embed" in key:
                wtype = "embedding"
            elif "output" in key or "lm_head" in key:
                wtype = "output_head"
            else:
                wtype = "other"

            type_groups[wtype]["params"] += info["numel"]
            type_groups[wtype]["count"] += 1

        total = sum(v["params"] for v in type_groups.values())
        for wtype in sorted(type_groups.keys()):
            v = type_groups[wtype]
            pct = 100.0 * v["params"] / total
            print(f"  {wtype}: {v['params'] / 1e6:.1f}M ({pct:.1f}%), {v['count']} tensors")

        # Per-layer parameter count
        layer_params = defaultdict(int)
        for key, info in comp_tensors.items():
            match = re.search(r'layers\.(\d+)\.', key)
            if match:
                layer_idx = int(match.group(1))
                layer_params[layer_idx] += info["numel"]

        if layer_params:
            layers = sorted(layer_params.keys())
            params_per_layer = [layer_params[i] / 1e6 for i in layers]
            print(f"\n  Per-layer params: {len(layers)} layers")
            print(f"  Range: {min(params_per_layer):.1f}M - {max(params_per_layer):.1f}M")
            print(f"  Mean:  {np.mean(params_per_layer):.1f}M")
            if len(layers) <= 10:
                for i in layers:
                    print(f"    Layer {i}: {layer_params[i] / 1e6:.1f}M")


# ============================================================================
# 4. Weight statistics (norms, spectral properties)
# ============================================================================

def weight_statistics(all_tensors, shard_paths):
    """Compute weight norms and spectral stats for key matrices."""
    print("\n" + "=" * 70)
    print("4. WEIGHT STATISTICS")
    print("=" * 70)

    # Open all shards
    shard_handles = {}
    for shard, path in shard_paths.items():
        shard_handles[shard] = safe_open(path, framework="pt", device=DEVICE)

    stats = {}
    # Focus on 2D+ weight matrices (skip norms/biases)
    weight_keys = [k for k, v in all_tensors.items() if len(v["shape"]) >= 2]

    print(f"\nComputing stats for {len(weight_keys)} weight matrices...")
    t0 = time.time()

    for i, key in enumerate(sorted(weight_keys)):
        info = all_tensors[key]
        tensor = shard_handles[info["shard"]].get_tensor(key).to(DEVICE)
        w = tensor.float()

        frobenius = torch.norm(w).item()
        spectral = torch.linalg.norm(w, ord=2).item()

        entry = {
            "shape": info["shape"],
            "frobenius_norm": round(frobenius, 4),
            "spectral_norm": round(spectral, 4),
            "mean": round(w.mean().item(), 6),
            "std": round(w.std().item(), 6),
            "condition_ratio": round(frobenius / (spectral + 1e-10), 4),
        }

        # Effective rank for reasonably sized matrices
        if w.shape[0] <= 8192 and w.shape[1] <= 8192:
            try:
                s = torch.linalg.svdvals(w)
                s_norm = s / s.sum()
                entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum().item()
                entry["effective_rank"] = round(np.exp(entropy), 2)
                entry["top1_sv_ratio"] = round((s[0] / s.sum()).item(), 4)
            except Exception:
                pass

        stats[key] = entry

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(weight_keys)} done...")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    # Print summary: per-layer Frobenius norms for text_model
    print(f"\n--- Slow AR (text_model) Frobenius norms per layer ---")
    for layer_idx in range(36):
        layer_keys = [k for k in stats if f"text_model.layers.{layer_idx}." in k]
        if layer_keys:
            norms = [stats[k]["frobenius_norm"] for k in layer_keys]
            eff_ranks = [stats[k].get("effective_rank", 0) for k in layer_keys if "effective_rank" in stats[k]]
            mean_norm = np.mean(norms)
            mean_rank = np.mean(eff_ranks) if eff_ranks else 0
            print(f"  Layer {layer_idx:2d}: mean ||W||_F = {mean_norm:8.1f}, mean eff_rank = {mean_rank:.1f}")

    # Print summary for audio_decoder
    print(f"\n--- Fast AR (audio_decoder) Frobenius norms per layer ---")
    for layer_idx in range(4):
        layer_keys = [k for k in stats if f"audio_decoder.layers.{layer_idx}." in k]
        if layer_keys:
            norms = [stats[k]["frobenius_norm"] for k in layer_keys]
            eff_ranks = [stats[k].get("effective_rank", 0) for k in layer_keys if "effective_rank" in stats[k]]
            mean_norm = np.mean(norms)
            mean_rank = np.mean(eff_ranks) if eff_ranks else 0
            print(f"  Layer {layer_idx}: mean ||W||_F = {mean_norm:8.1f}, mean eff_rank = {mean_rank:.1f}")

    with open(RESULTS_DIR / "weight_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nFull stats saved to {RESULTS_DIR / 'weight_statistics.json'}")

    return stats


# ============================================================================
# 5. Tokenizer inspection
# ============================================================================

def inspect_tokenizer():
    """Inspect the tokenizer — vocab size, special tokens, audio token ranges."""
    print("\n" + "=" * 70)
    print("5. TOKENIZER INSPECTION")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # Special tokens
    print(f"\nSpecial tokens:")
    for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        val = getattr(tokenizer, attr, None)
        tok_id = getattr(tokenizer, f"{attr}_id", None)
        if val:
            print(f"  {attr}: '{val}' (id={tok_id})")

    # Additional special tokens
    if hasattr(tokenizer, "additional_special_tokens"):
        addl = tokenizer.additional_special_tokens
        if addl:
            print(f"\n  Additional special tokens ({len(addl)} total):")
            for t in addl[:30]:
                print(f"    '{t}' -> {tokenizer.convert_tokens_to_ids(t)}")
            if len(addl) > 30:
                print(f"    ... and {len(addl) - 30} more")

    # Test encoding
    test_text = "Hello, this is a test of Fish Speech."
    ids = tokenizer.encode(test_text)
    print(f"\nTest encoding: '{test_text}'")
    print(f"  Token IDs ({len(ids)}): {ids}")
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"  Tokens: {tokens}")

    # Analyze token ranges
    print(f"\n--- Token space analysis ---")
    print(f"  Text tokens: 0 - 151677 (standard Qwen3 vocab)")
    print(f"  Audio semantic tokens: 151678 - 155773 ({155773 - 151678 + 1} tokens)")
    print(f"  Total vocab: {tokenizer.vocab_size}")

    return tokenizer


# ============================================================================
# 6. Codec file inspection
# ============================================================================

def inspect_codec():
    """Download and inspect the codec.pth file."""
    print("\n" + "=" * 70)
    print("6. AUDIO CODEC INSPECTION")
    print("=" * 70)

    codec_path = hf_hub_download(MODEL_NAME, filename="codec.pth")
    print(f"\nCodec file: {codec_path}")

    # Load the checkpoint
    codec_state = torch.load(codec_path, map_location="cpu", weights_only=False)

    if isinstance(codec_state, dict):
        print(f"Codec type: dict with keys: {list(codec_state.keys())}")

        # If it has a state_dict key
        if "state_dict" in codec_state:
            sd = codec_state["state_dict"]
        elif "model" in codec_state:
            sd = codec_state["model"]
        else:
            # Might be a raw state dict
            sd = codec_state

        if isinstance(sd, dict):
            # Analyze the state dict
            total_params = 0
            components = defaultdict(lambda: {"params": 0, "tensors": 0})

            print(f"\nTotal tensors: {len(sd)}")
            for key, tensor in sd.items():
                if isinstance(tensor, torch.Tensor):
                    total_params += tensor.numel()
                    top = key.split(".")[0]
                    components[top]["params"] += tensor.numel()
                    components[top]["tensors"] += 1

            print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
            print(f"\nComponents:")
            for comp in sorted(components.keys()):
                v = components[comp]
                print(f"  {comp}: {v['params'] / 1e6:.1f}M params, {v['tensors']} tensors")

            # Show unique patterns
            patterns = defaultdict(list)
            for key in sd:
                if isinstance(sd[key], torch.Tensor):
                    pattern = re.sub(r'\.\d+\.', '.N.', key)
                    patterns[pattern].append((key, list(sd[key].shape)))

            print(f"\nUnique weight patterns ({len(patterns)}):")
            for pattern in sorted(patterns.keys()):
                examples = patterns[pattern]
                shape = examples[0][1]
                count = len(examples)
                suffix = f" (x{count})" if count > 1 else ""
                print(f"  {pattern}: {shape}{suffix}")
    else:
        print(f"Codec type: {type(codec_state)}")

    return codec_state


# ============================================================================
# Main
# ============================================================================

def main():
    print("Fish Speech S2 Pro — Architecture Investigation")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Config
    config = inspect_config()

    # Step 2: Weight inspection
    all_tensors, shard_paths = inspect_weights()

    # Step 3: Parameter census
    parameter_census(all_tensors)

    # Step 4: Weight statistics
    weight_statistics(all_tensors, shard_paths)

    # Step 5: Tokenizer
    inspect_tokenizer()

    # Step 6: Codec
    inspect_codec()

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total investigation time: {total_time:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
