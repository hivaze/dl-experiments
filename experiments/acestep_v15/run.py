"""
ACE-Step 1.5 — Architecture Investigation
==========================================
Load ACE-Step/Ace-Step1.5 and explore its internals:
  - Multi-component architecture (LM planner + DiT + Condition Encoder + VAE + Tokenizer)
  - Layer shapes, parameter counts per component
  - DiT attention pattern (alternating sliding/full)
  - Flow matching dual-timestep mechanism
  - FSQ tokenizer inspection
  - Weight statistics

Model: ACE-Step/Ace-Step1.5 (hybrid LM + DiT, text-to-music)
  - LM: Qwen3-1.7B (28 layers) as song blueprint planner
  - DiT: 24 layers, alternating sliding/full attention, AdaLN, dual-timestep flow matching
  - Condition encoder: text projector + 8-layer lyric encoder + 4-layer timbre encoder
  - Audio tokenizer: VAE (AutoencoderOobleck, 1920x compression) → FSQ (levels [8,8,8,5,5,5])
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "ACE-Step/Ace-Step1.5"
DEVICE = "cuda:1"  # Use second GPU to keep first free
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Submodel paths within the HF repo
SUBMODELS = {
    "dit": "acestep-v15-turbo",
    "lm": "acestep-5Hz-lm-1.7B",
    "vae": "vae",
    "text_embedder": "Qwen3-Embedding-0.6B",
}


# ============================================================================
# 1. Download and inspect configs
# ============================================================================

def inspect_configs():
    """Download and inspect all component configs."""
    print("=" * 70)
    print("1. MODEL CONFIGURATIONS")
    print("=" * 70)

    from huggingface_hub import hf_hub_download, list_repo_tree
    import os

    # List all files in the repo to understand structure
    print("\nRepository file structure:")
    try:
        files = list(list_repo_tree(MODEL_NAME))
        tree = defaultdict(list)
        for f in files:
            parts = f.path.split("/")
            if len(parts) == 1:
                tree["root"].append(f.path)
            else:
                tree[parts[0]].append("/".join(parts[1:]))

        for folder in sorted(tree.keys()):
            print(f"\n  {folder}/")
            for f in sorted(tree[folder])[:15]:
                print(f"    {f}")
            if len(tree[folder]) > 15:
                print(f"    ... and {len(tree[folder]) - 15} more")
    except Exception as e:
        print(f"  Could not list repo tree: {e}")

    # Download config files
    configs = {}
    for component, subpath in SUBMODELS.items():
        try:
            config_path = hf_hub_download(
                MODEL_NAME,
                filename=f"{subpath}/config.json",
            )
            with open(config_path) as f:
                cfg = json.load(f)
            configs[component] = cfg
            print(f"\n--- {component} ({subpath}) config ---")
            # Print key fields
            for k, v in cfg.items():
                if not isinstance(v, (list, dict)) or len(str(v)) < 100:
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"\n--- {component} ({subpath}): config not found ({e}) ---")

    with open(RESULTS_DIR / "configs.json", "w") as f:
        json.dump(configs, f, indent=2, default=str)

    return configs


# ============================================================================
# 2. Load and inspect DiT (the core audio generator)
# ============================================================================

def inspect_dit():
    """Load the DiT model and inspect its architecture."""
    print("\n" + "=" * 70)
    print("2. DIFFUSION TRANSFORMER (DiT)")
    print("=" * 70)

    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    # Try loading as a transformers model with trust_remote_code
    try:
        config_path = hf_hub_download(MODEL_NAME, filename="acestep-v15-turbo/config.json")
        # Also need the modeling code
        for fname in [
            "acestep-v15-turbo/modeling_acestep_v15_turbo.py",
            "acestep-v15-turbo/configuration_acestep_v15.py",
        ]:
            try:
                hf_hub_download(MODEL_NAME, filename=fname)
            except Exception:
                pass

        config = AutoConfig.from_pretrained(
            hf_hub_download(MODEL_NAME, filename="acestep-v15-turbo/config.json").replace("/config.json", ""),
            trust_remote_code=True,
        )
        print(f"\nDiT config type: {type(config).__name__}")
        print(f"DiT config: {config}")
    except Exception as e:
        print(f"\nCould not load DiT config via AutoConfig: {e}")
        # Fallback: load the safetensors directly
        print("Falling back to direct safetensors inspection...")

    # Load the safetensors file to inspect weight shapes
    try:
        from safetensors import safe_open
        model_path = hf_hub_download(MODEL_NAME, filename="acestep-v15-turbo/model.safetensors")
        print(f"\nDiT weights file: {model_path}")

        with safe_open(model_path, framework="pt") as f:
            keys = f.keys()
            print(f"Total weight tensors: {len(keys)}")

            # Group by component
            components = defaultdict(list)
            total_params = 0
            for key in sorted(keys):
                tensor = f.get_tensor(key)
                total_params += tensor.numel()
                parts = key.split(".")
                if len(parts) >= 2:
                    comp = ".".join(parts[:2])
                else:
                    comp = parts[0]
                components[comp].append((key, list(tensor.shape), tensor.numel()))

            print(f"Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
            print(f"\nComponents:")
            for comp in sorted(components.keys()):
                entries = components[comp]
                comp_params = sum(e[2] for e in entries)
                print(f"  {comp}: {comp_params / 1e6:.1f}M params, {len(entries)} tensors")

            # Print all unique weight name patterns (collapse layer indices)
            import re
            patterns = set()
            for key in keys:
                pattern = re.sub(r'\.\d+\.', '.N.', key)
                patterns.add(pattern)
            print(f"\nUnique weight patterns ({len(patterns)}):")
            for p in sorted(patterns):
                # Show shape for one example
                example_key = [k for k in keys if re.sub(r'\.\d+\.', '.N.', k) == p][0]
                tensor = f.get_tensor(example_key)
                print(f"  {p}: {list(tensor.shape)}")

    except Exception as e:
        print(f"\nCould not inspect DiT weights: {e}")


# ============================================================================
# 3. Load and inspect LM planner
# ============================================================================

def inspect_lm():
    """Load the Qwen3-1.7B LM planner and inspect it."""
    print("\n" + "=" * 70)
    print("3. LM PLANNER (Qwen3-1.7B)")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Download entire LM subdirectory via snapshot_download
    from huggingface_hub import hf_hub_download, snapshot_download
    from safetensors import safe_open

    lm_config_path = hf_hub_download(MODEL_NAME, filename="acestep-5Hz-lm-1.7B/config.json")
    lm_dir = str(Path(lm_config_path).parent)

    config = AutoConfig.from_pretrained(lm_dir, trust_remote_code=True)
    print(f"\nLM config type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Num KV heads: {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")

    # Inspect LM weights via safetensors directly
    try:
        lm_weights_path = hf_hub_download(MODEL_NAME, filename="acestep-5Hz-lm-1.7B/model.safetensors")
        print(f"\nLM weights file: {lm_weights_path}")

        with safe_open(lm_weights_path, framework="pt") as f:
            keys = list(f.keys())
            total_params = sum(f.get_tensor(k).numel() for k in keys)
            print(f"Total tensors: {len(keys)}")
            print(f"Total LM params: {total_params:,} ({total_params / 1e6:.1f}M)")

            # Group by component
            components = defaultdict(lambda: {"params": 0, "tensors": 0})
            for key in keys:
                tensor = f.get_tensor(key)
                parts = key.split(".")
                comp = parts[0] if len(parts) > 1 else key
                components[comp]["params"] += tensor.numel()
                components[comp]["tensors"] += 1

            print(f"\nLM architecture (weight groups):")
            for comp in sorted(components.keys()):
                v = components[comp]
                print(f"  {comp}: {v['params'] / 1e6:.1f}M params, {v['tensors']} tensors")

            # Per-layer param count
            import re
            layer_params = defaultdict(int)
            for key in keys:
                match = re.search(r'layers\.(\d+)\.', key)
                if match:
                    layer_params[int(match.group(1))] += f.get_tensor(key).numel()

            if layer_params:
                layers = sorted(layer_params.keys())
                ppl = [layer_params[i] / 1e6 for i in layers]
                print(f"\n  {len(layers)} layers, {min(ppl):.1f}M - {max(ppl):.1f}M params each")

    except Exception as e:
        print(f"\nCould not inspect LM weights: {e}")

    # Try tokenizer
    try:
        tok_path = hf_hub_download(MODEL_NAME, filename="acestep-5Hz-lm-1.7B/tokenizer.json")
        tok_dir = str(Path(tok_path).parent)
        tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        print(f"\nTokenizer vocab size: {tok.vocab_size}")
        test = "A pop song with upbeat rhythm"
        ids = tok.encode(test)
        print(f"Test encoding: '{test}' → {len(ids)} tokens")
    except Exception as e:
        print(f"\nTokenizer: {e}")

    return config


# ============================================================================
# 4. Inspect VAE (AutoencoderOobleck)
# ============================================================================

def inspect_vae():
    """Load the audio VAE and inspect its architecture."""
    print("\n" + "=" * 70)
    print("4. AUDIO VAE (AutoencoderOobleck)")
    print("=" * 70)

    from huggingface_hub import hf_hub_download

    vae_config_path = hf_hub_download(MODEL_NAME, filename="vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)

    print(f"\nVAE config:")
    for k, v in vae_config.items():
        print(f"  {k}: {v}")

    # Download the VAE weights explicitly
    vae_weights_path = hf_hub_download(MODEL_NAME, filename="vae/diffusion_pytorch_model.safetensors")
    vae_dir = str(Path(vae_weights_path).parent)

    # Load via diffusers if available
    try:
        from diffusers import AutoencoderOobleck

        vae = AutoencoderOobleck.from_pretrained(vae_dir, torch_dtype=torch.bfloat16).to(DEVICE)
        vae.eval()

        total_params = sum(p.numel() for p in vae.parameters())
        print(f"\nVAE total params: {total_params:,} ({total_params / 1e6:.1f}M)")

        # Architecture
        print(f"\nVAE architecture:")
        for name, child in vae.named_children():
            n_params = sum(p.numel() for p in child.parameters())
            print(f"  {name}: {type(child).__name__} ({n_params / 1e6:.1f}M)")

        # Test forward pass with dummy audio
        print("\nTest forward pass (10s stereo audio @ 48kHz):")
        dummy_audio = torch.randn(1, 2, 480000, device=DEVICE, dtype=torch.bfloat16)
        with torch.no_grad():
            encoded = vae.encode(dummy_audio)
            latent = encoded.latent_dist.sample()
            print(f"  Input shape:  {list(dummy_audio.shape)} (batch, channels, samples)")
            print(f"  Latent shape: {list(latent.shape)}")
            compression = dummy_audio.shape[-1] / latent.shape[-1]
            print(f"  Temporal compression: {compression:.0f}x")
            print(f"  Latent rate: {48000 / compression:.1f} Hz")

            decoded = vae.decode(latent)
            recon = decoded.sample
            print(f"  Decoded shape: {list(recon.shape)}")

        del vae
        torch.cuda.empty_cache()

    except ImportError:
        print("\n  diffusers not installed — skipping VAE loading")
    except Exception as e:
        print(f"\n  Could not load VAE: {e}")


# ============================================================================
# 5. Inspect audio tokenizer (FSQ)
# ============================================================================

def inspect_audio_tokenizer():
    """Inspect the FSQ audio tokenizer."""
    print("\n" + "=" * 70)
    print("5. AUDIO TOKENIZER (FSQ)")
    print("=" * 70)

    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    # The tokenizer is part of the DiT model file
    model_path = hf_hub_download(MODEL_NAME, filename="acestep-v15-turbo/model.safetensors")

    with safe_open(model_path, framework="pt") as f:
        tokenizer_keys = [k for k in f.keys() if "tokenizer" in k.lower() or "quantize" in k.lower() or "fsq" in k.lower()]
        if not tokenizer_keys:
            # Try broader search
            tokenizer_keys = [k for k in f.keys() if "audio_token" in k.lower() or "detokenizer" in k.lower()]

        if tokenizer_keys:
            print(f"\nTokenizer-related weights ({len(tokenizer_keys)}):")
            for key in sorted(tokenizer_keys):
                tensor = f.get_tensor(key)
                print(f"  {key}: {list(tensor.shape)} ({tensor.dtype})")
        else:
            print("\nNo tokenizer weights found with expected names.")
            # List all top-level prefixes
            prefixes = set(k.split(".")[0] for k in f.keys())
            print(f"Available top-level prefixes: {sorted(prefixes)}")


# ============================================================================
# 6. DiT attention pattern analysis
# ============================================================================

def analyze_dit_attention_pattern():
    """Analyze the alternating sliding/full attention pattern in DiT layers."""
    print("\n" + "=" * 70)
    print("6. DiT ATTENTION PATTERN ANALYSIS")
    print("=" * 70)

    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(MODEL_NAME, filename="acestep-v15-turbo/config.json")
    with open(config_path) as f:
        config = json.load(f)

    # Check for attention type configuration
    attn_keys = [k for k in config if "attention" in k.lower() or "sliding" in k.lower() or "window" in k.lower()]
    print(f"\nAttention-related config keys:")
    for k in attn_keys:
        print(f"  {k}: {config[k]}")

    # The pattern should be defined in the config
    if "attention_type" in config:
        print(f"\nAttention types per layer: {config['attention_type']}")
    elif "attn_type" in config:
        print(f"\nAttention types per layer: {config['attn_type']}")
    else:
        print(f"\nAttention type configuration not found in standard keys.")
        print("Full config keys:", sorted(config.keys()))


# ============================================================================
# 7. Summary
# ============================================================================

def save_summary(configs):
    """Save a concise summary of all findings."""
    summary = {
        "model": MODEL_NAME,
        "components": {
            "lm_planner": "Qwen3-1.7B based, 28 layers, song blueprint via CoT",
            "dit": "24-layer DiT, alternating sliding/full attention, AdaLN, dual-timestep flow matching",
            "condition_encoder": "text projector + 8-layer lyric encoder + 4-layer timbre encoder",
            "vae": "AutoencoderOobleck, stereo 48kHz, ~1920x compression",
            "audio_tokenizer": "FSQ with levels [8,8,8,5,5,5], attention pooling 25Hz→5Hz",
            "text_embedder": "Qwen3-Embedding-0.6B (1024-dim)",
        },
        "configs": configs,
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {RESULTS_DIR / 'summary.json'}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("ACE-Step 1.5 — Architecture Investigation")
    print("=" * 70)

    t_start = time.time()

    # Step 1: Configs
    configs = inspect_configs()

    # Step 2: DiT
    inspect_dit()

    # Step 3: LM planner
    inspect_lm()

    # Step 4: VAE
    inspect_vae()

    # Step 5: Audio tokenizer
    inspect_audio_tokenizer()

    # Step 6: Attention patterns
    analyze_dit_attention_pattern()

    # Step 7: Summary
    save_summary(configs)

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"Total investigation time: {total_time:.1f}s")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
