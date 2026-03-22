#!/usr/bin/env python3
"""
T-11: Quantization Methods Comparative Analysis
================================================
Investigates quantization sensitivity at per-layer and per-matrix granularity,
correlates with T-7 (linearity gap) and T-9 (spectral structure), and compares
full-model quantization methods.

Key research questions:
1. Which layers/matrices are most sensitive to quantization noise?
2. Do T-7 linearity gap and T-9 effective rank predict quantization robustness?
3. Can spectral-informed mixed-precision outperform uniform quantization?
4. How do different quantization methods compare on quality, memory, and speed?

Usage: poetry run python experiments/t11_quantization/run.py
"""

import json
import time
import gc
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Configuration ───────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda:0"
SEED = 42

EVAL_SEQ_LEN = 4096
RTN_BIT_WIDTHS = [8, 6, 4, 3, 2]
RTN_GROUP_SIZE = 128
MATRIX_SENSITIVITY_BITS = 3

RESULTS_DIR = Path("experiments/t11_quantization/results")
T7_RESULTS = Path("experiments/t7_layer_linearization_gap/results/summary.json")
T9_RESULTS = Path("experiments/t9_weight_spectral_structure/results/summary.json")

ATTN_MATRICES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_MATRICES = ["gate_proj", "up_proj", "down_proj"]
ALL_MATRICES = ATTN_MATRICES + MLP_MATRICES

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─── Utility Functions ───────────────────────────────────────────────────────

def get_weight(model, layer_idx, matrix_name):
    layer = model.model.layers[layer_idx]
    if matrix_name in ATTN_MATRICES:
        return getattr(layer.self_attn, matrix_name).weight
    return getattr(layer.mlp, matrix_name).weight


def quantize_dequantize_rtn(weight, bits, group_size=128):
    """Symmetric per-group RTN quantization simulation (quantize then dequantize)."""
    orig_shape = weight.shape
    orig_dtype = weight.dtype
    w = weight.float()

    if group_size > 0 and w.numel() % group_size == 0:
        w = w.reshape(-1, group_size)
    else:
        w = w.reshape(1, -1)

    max_int = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / max_int
    w_q = (w / scale).round().clamp(-max_int - 1, max_int)
    return (w_q * scale).reshape(orig_shape).to(orig_dtype)


def quant_error(weight, bits, group_size=128):
    """Relative Frobenius-norm quantization error."""
    w_dq = quantize_dequantize_rtn(weight, bits, group_size)
    return (weight.float() - w_dq.float()).norm().item() / weight.float().norm().clamp(min=1e-8).item()


@torch.no_grad()
def evaluate_perplexity(model, input_ids):
    """Single forward-pass perplexity on a token sequence (loss in float32)."""
    ids = input_ids.to(model.device)
    logits = model(ids, use_cache=False).logits[:, :-1, :].float()
    labels = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return torch.exp(loss).item()


@torch.no_grad()
def measure_generation_speed(model, tokenizer, num_tokens=128, warmup=2, runs=3):
    """Measure autoregressive generation speed (tokens/sec)."""
    messages = [{"role": "user", "content": "Explain the transformer architecture."}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt",
                                               add_generation_prompt=True).to(model.device)
    for _ in range(warmup):
        model.generate(input_ids, max_new_tokens=8, do_sample=False)

    torch.cuda.synchronize()
    speeds = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=num_tokens, do_sample=False)
        torch.cuda.synchronize()
        gen = out.shape[1] - input_ids.shape[1]
        speeds.append(gen / (time.perf_counter() - t0))
    return float(np.mean(speeds))


def prepare_eval_data(tokenizer, seq_len):
    """Load WikiText-2 test set for perplexity evaluation."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        print(f"  WikiText-2 loaded: {enc.input_ids.size(1)} tokens")
        return enc.input_ids
    except Exception as e:
        print(f"  WikiText-2 unavailable ({e}), using prompts.json")
        with open("data/text_completions/prompts.json") as f:
            prompts = json.load(f)["prompts"]
        text = " ".join(p["text"] for p in prompts)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        return enc.input_ids


def adjust_bits_to_target(bits, target_avg):
    """Iteratively adjust per-layer bit allocation to match a target average."""
    bits = bits.astype(float).copy()
    for _ in range(200):
        if abs(bits.mean() - target_avg) <= 0.05:
            break
        if bits.mean() > target_avg:
            idx = int(np.argmax(bits))
            bits[idx] = max(2, bits[idx] - 1)
        else:
            idx = int(np.argmin(bits))
            bits[idx] = min(8, bits[idx] + 1)
    return bits.astype(int)


# ─── Phase 1: Per-Layer RTN Sensitivity ──────────────────────────────────────

def run_phase1(model, eval_ids, num_layers):
    print("\n" + "=" * 80)
    print("PHASE 1: Per-Layer RTN Quantization Sensitivity")
    print("=" * 80)
    t0 = time.time()

    baseline_ppl = evaluate_perplexity(model, eval_ids)
    print(f"  Baseline (BF16) perplexity: {baseline_ppl:.4f}\n")

    per_layer = {}
    for li in range(num_layers):
        # save originals
        orig = {m: get_weight(model, li, m).data.clone() for m in ALL_MATRICES}

        layer_res = {"ppl": {}, "ppl_delta": {}, "quant_error": {}}
        for bits in RTN_BIT_WIDTHS:
            errors = {}
            for m in ALL_MATRICES:
                w = orig[m]
                errors[m] = quant_error(w, bits, RTN_GROUP_SIZE)
                get_weight(model, li, m).data.copy_(quantize_dequantize_rtn(w, bits, RTN_GROUP_SIZE))

            ppl = evaluate_perplexity(model, eval_ids)
            layer_res["ppl"][bits] = ppl
            layer_res["ppl_delta"][bits] = ppl - baseline_ppl
            layer_res["quant_error"][bits] = errors

            # restore
            for m in ALL_MATRICES:
                get_weight(model, li, m).data.copy_(orig[m])

        per_layer[li] = layer_res
        tag = " | ".join(f"{b}b:{layer_res['ppl_delta'][b]:+.2f}" for b in RTN_BIT_WIDTHS)
        print(f"  Layer {li:2d}: {tag}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 done in {elapsed:.1f}s")
    return {"baseline_ppl": baseline_ppl, "bit_widths": RTN_BIT_WIDTHS,
            "group_size": RTN_GROUP_SIZE, "per_layer": per_layer, "time_s": elapsed}


# ─── Phase 1b: Per-Matrix Sensitivity ───────────────────────────────────────

def run_phase1b(model, eval_ids, num_layers, baseline_ppl):
    print("\n" + "=" * 80)
    print(f"PHASE 1b: Per-Matrix Sensitivity ({MATRIX_SENSITIVITY_BITS}-bit RTN)")
    print("=" * 80)
    t0 = time.time()

    per_layer = {}
    for li in range(num_layers):
        layer_res = {}
        for m in ALL_MATRICES:
            orig_w = get_weight(model, li, m).data.clone()
            err = quant_error(orig_w, MATRIX_SENSITIVITY_BITS, RTN_GROUP_SIZE)
            get_weight(model, li, m).data.copy_(
                quantize_dequantize_rtn(orig_w, MATRIX_SENSITIVITY_BITS, RTN_GROUP_SIZE))
            ppl = evaluate_perplexity(model, eval_ids)
            layer_res[m] = {"ppl": ppl, "ppl_delta": ppl - baseline_ppl, "quant_error": err}
            get_weight(model, li, m).data.copy_(orig_w)

        per_layer[li] = layer_res
        top3 = sorted(layer_res.items(), key=lambda x: x[1]["ppl_delta"], reverse=True)[:3]
        tag = " | ".join(f"{m}:{v['ppl_delta']:+.4f}" for m, v in top3)
        print(f"  Layer {li:2d}: {tag}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1b done in {elapsed:.1f}s")
    return {"baseline_ppl": baseline_ppl, "bits": MATRIX_SENSITIVITY_BITS,
            "per_layer": per_layer, "time_s": elapsed}


# ─── Phase 2: Full-Model Methods Comparison ─────────────────────────────────

def run_phase2(model, tokenizer, eval_ids, num_layers, baseline_ppl):
    print("\n" + "=" * 80)
    print("PHASE 2: Full-Model Quantization Methods Comparison")
    print("=" * 80)
    t0 = time.time()

    bf16_mem = torch.cuda.memory_allocated(DEVICE)
    bf16_speed = measure_generation_speed(model, tokenizer)
    methods = {
        "bf16_baseline": {
            "ppl": baseline_ppl, "ppl_delta": 0.0,
            "memory_mb": bf16_mem / 1e6, "tokens_per_sec": bf16_speed,
            "effective_bits": 16,
        }
    }
    print(f"  BF16 baseline: PPL={baseline_ppl:.4f}, mem={bf16_mem/1e9:.2f}GB, "
          f"gen={bf16_speed:.1f} tok/s")

    # RTN uniform at key bit-widths
    for bits in [8, 4, 3]:
        label = f"rtn_{bits}bit"
        print(f"\n  {label} (group_size={RTN_GROUP_SIZE})...")
        mt0 = time.time()
        orig_all = {}
        total_orig = total_quant = 0
        for li in range(num_layers):
            orig_all[li] = {}
            for m in ALL_MATRICES:
                w = get_weight(model, li, m)
                orig_all[li][m] = w.data.clone()
                total_orig += w.numel() * 2
                total_quant += w.numel() * bits / 8
                w.data.copy_(quantize_dequantize_rtn(w.data, bits, RTN_GROUP_SIZE))

        ppl = evaluate_perplexity(model, eval_ids)
        speed = measure_generation_speed(model, tokenizer)
        methods[label] = {
            "ppl": ppl, "ppl_delta": ppl - baseline_ppl,
            "compression_ratio": total_orig / total_quant,
            "effective_bits": bits, "tokens_per_sec": speed,
            "time_s": time.time() - mt0,
        }
        print(f"    PPL={ppl:.4f} (Δ={ppl-baseline_ppl:+.4f}), "
              f"compression={total_orig/total_quant:.1f}x, gen={speed:.1f} tok/s")
        for li in range(num_layers):
            for m in ALL_MATRICES:
                get_weight(model, li, m).data.copy_(orig_all[li][m])
        del orig_all; torch.cuda.empty_cache()

    # bitsandbytes NF4
    print("\n  bitsandbytes NF4 (loading fresh model)...")
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        mt0 = time.time()
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        m_nf4 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg,
                                                      device_map=DEVICE, dtype=torch.bfloat16)
        load_t = time.time() - mt0
        mem = torch.cuda.memory_allocated(DEVICE)
        ppl = evaluate_perplexity(m_nf4, eval_ids)
        speed = measure_generation_speed(m_nf4, tokenizer)
        methods["bnb_nf4"] = {
            "ppl": ppl, "ppl_delta": ppl - baseline_ppl,
            "memory_mb": mem / 1e6, "load_time_s": load_t,
            "tokens_per_sec": speed, "effective_bits": 4.0,
        }
        print(f"    PPL={ppl:.4f} (Δ={ppl-baseline_ppl:+.4f}), "
              f"mem={mem/1e9:.2f}GB, gen={speed:.1f} tok/s")
        del m_nf4; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"    Failed: {e}")
        methods["bnb_nf4"] = {"error": str(e)}

    # bitsandbytes INT8
    print("\n  bitsandbytes INT8 (loading fresh model)...")
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        mt0 = time.time()
        cfg8 = BitsAndBytesConfig(load_in_8bit=True)
        m_int8 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg8,
                                                       device_map=DEVICE, dtype=torch.bfloat16)
        load_t = time.time() - mt0
        mem = torch.cuda.memory_allocated(DEVICE)
        ppl = evaluate_perplexity(m_int8, eval_ids)
        speed = measure_generation_speed(m_int8, tokenizer)
        methods["bnb_int8"] = {
            "ppl": ppl, "ppl_delta": ppl - baseline_ppl,
            "memory_mb": mem / 1e6, "load_time_s": load_t,
            "tokens_per_sec": speed, "effective_bits": 8.0,
        }
        print(f"    PPL={ppl:.4f} (Δ={ppl-baseline_ppl:+.4f}), "
              f"mem={mem/1e9:.2f}GB, gen={speed:.1f} tok/s")
        del m_int8; torch.cuda.empty_cache(); gc.collect()
    except Exception as e:
        print(f"    Failed: {e}")
        methods["bnb_int8"] = {"error": str(e)}

    # torchao weight-only quantization
    for ao_label, ao_bits in [("torchao_int8wo", 8), ("torchao_int4wo", 4)]:
        print(f"\n  {ao_label} (loading fresh model)...")
        try:
            from transformers import AutoModelForCausalLM
            import torchao
            from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig

            mt0 = time.time()
            m_ao = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                         device_map=DEVICE)
            m_ao.eval()
            if ao_bits == 8:
                quantize_(m_ao, Int8WeightOnlyConfig())
            else:
                quantize_(m_ao, Int4WeightOnlyConfig(group_size=RTN_GROUP_SIZE))
            load_t = time.time() - mt0
            mem = torch.cuda.memory_allocated(DEVICE)
            ppl = evaluate_perplexity(m_ao, eval_ids)
            speed = measure_generation_speed(m_ao, tokenizer)
            methods[ao_label] = {
                "ppl": ppl, "ppl_delta": ppl - baseline_ppl,
                "memory_mb": mem / 1e6, "load_time_s": load_t,
                "tokens_per_sec": speed, "effective_bits": ao_bits,
            }
            print(f"    PPL={ppl:.4f} (Δ={ppl-baseline_ppl:+.4f}), "
                  f"mem={mem/1e9:.2f}GB, gen={speed:.1f} tok/s")
            del m_ao; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"    Failed: {e}")
            methods[ao_label] = {"error": str(e)}

    # Prepare calibration data for llmcompressor
    def _get_calib_dataset():
        try:
            from datasets import load_dataset, Dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"] if len(t.strip()) > 50][:256]
            return Dataset.from_dict({"text": texts})
        except Exception:
            return None

    calib_ds = _get_calib_dataset()

    for lmc_label, lmc_scheme, lmc_bits, lmc_cls in [
        ("llmcompressor_gptq_w4a16", "W4A16", 4, "GPTQModifier"),
        ("llmcompressor_gptq_w8a16", "W8A16", 8, "GPTQModifier"),
        ("llmcompressor_w8a8", "W8A8", 8, "QuantizationModifier"),
    ]:
        print(f"\n  {lmc_label} (calibration-based, loading fresh model)...")
        if calib_ds is None:
            methods[lmc_label] = {"error": "calibration data unavailable"}
            print("    Failed: calibration data unavailable")
            continue
        try:
            from transformers import AutoModelForCausalLM
            from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
            from llmcompressor import oneshot
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE.split(":")[-1]

            mt0 = time.time()
            m_lmc = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16,
                                                          device_map=DEVICE)
            m_lmc.eval()
            if lmc_cls == "GPTQModifier":
                recipe = GPTQModifier(targets="Linear", scheme=lmc_scheme,
                                       ignore=["lm_head"])
            else:
                recipe = QuantizationModifier(targets="Linear", scheme=lmc_scheme,
                                               ignore=["lm_head"])
            oneshot(model=m_lmc, dataset=calib_ds, recipe=recipe,
                    max_seq_length=2048, num_calibration_samples=256)
            load_t = time.time() - mt0
            # Move model to single device for eval if needed
            if hasattr(m_lmc, 'hf_device_map'):
                m_lmc = m_lmc.to(DEVICE)
            mem = torch.cuda.memory_allocated(DEVICE)
            ppl = evaluate_perplexity(m_lmc, eval_ids)
            speed = measure_generation_speed(m_lmc, tokenizer)
            methods[lmc_label] = {
                "ppl": ppl, "ppl_delta": ppl - baseline_ppl,
                "memory_mb": mem / 1e6, "load_time_s": load_t,
                "tokens_per_sec": speed, "effective_bits": lmc_bits,
            }
            print(f"    PPL={ppl:.4f} (Δ={ppl-baseline_ppl:+.4f}), "
                  f"mem={mem/1e9:.2f}GB, gen={speed:.1f} tok/s, quant_time={load_t:.1f}s")
            del m_lmc; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"    Failed: {e}")
            methods[lmc_label] = {"error": str(e)}

    elapsed = time.time() - t0
    print(f"\n  Phase 2 done in {elapsed:.1f}s")
    return {"baseline_ppl": baseline_ppl, "bf16_memory_mb": bf16_mem / 1e6,
            "methods": methods, "time_s": elapsed}


# ─── Phase 3: Spectral-Informed Mixed Precision ─────────────────────────────

def apply_mixed_precision_eval(model, eval_ids, bits_per_layer, num_layers):
    """Quantize each layer to its assigned bit-width, evaluate, restore."""
    orig = {}
    for li in range(num_layers):
        orig[li] = {}
        b = int(bits_per_layer[li])
        if b >= 16:
            continue
        for m in ALL_MATRICES:
            w = get_weight(model, li, m)
            orig[li][m] = w.data.clone()
            w.data.copy_(quantize_dequantize_rtn(w.data, b, RTN_GROUP_SIZE))

    ppl = evaluate_perplexity(model, eval_ids)

    for li in orig:
        for m in orig[li]:
            get_weight(model, li, m).data.copy_(orig[li][m])
    return ppl


def run_phase3(model, eval_ids, num_layers, baseline_ppl, t7_data, t9_data,
               phase1_results):
    print("\n" + "=" * 80)
    print("PHASE 3: Spectral-Informed Mixed-Precision Recipes")
    print("=" * 80)
    t0 = time.time()
    target_avg = 4.0
    recipes = {}

    # Extract per-layer metrics
    t9_rank = np.array([
        np.mean([t9_data["per_layer"][str(l)][m]["effective_rank_ratio"]
                 for m in ALL_MATRICES if m in t9_data["per_layer"][str(l)]])
        for l in range(num_layers)])
    t7_gap = np.array([t7_data["per_layer"][f"layer_{l}"]["perturb_gap_mean"] for l in range(num_layers)])
    sensitivity_4b = np.array([phase1_results["per_layer"][l]["ppl_delta"][4]
                               for l in range(num_layers)])
    sensitivity_2b = np.array([phase1_results["per_layer"][l]["ppl_delta"][2]
                               for l in range(num_layers)])

    def norm01(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def make_recipe(score, name, description):
        raw = 2 + norm01(score) * 6
        scaled = raw * (target_avg * num_layers / raw.sum())
        bits = adjust_bits_to_target(np.clip(np.round(scaled), 2, 8).astype(int), target_avg)
        ppl = apply_mixed_precision_eval(model, eval_ids, bits, num_layers)
        recipes[name] = {
            "bits_per_layer": bits.tolist(), "avg_bits": float(bits.mean()),
            "ppl": ppl, "ppl_delta": ppl - baseline_ppl, "description": description,
        }
        print(f"  {name:28s}: PPL={ppl:.4f} (Δ={ppl-baseline_ppl:+.4f}), "
              f"avg={bits.mean():.2f}b, range=[{bits.min()},{bits.max()}]")
        return bits

    # Uniform 4-bit baseline
    uniform = np.full(num_layers, 4)
    ppl_u = apply_mixed_precision_eval(model, eval_ids, uniform, num_layers)
    recipes["uniform_4bit"] = {
        "bits_per_layer": uniform.tolist(), "avg_bits": 4.0,
        "ppl": ppl_u, "ppl_delta": ppl_u - baseline_ppl,
        "description": "Uniform 4-bit baseline",
    }
    print(f"  {'uniform_4bit':28s}: PPL={ppl_u:.4f} (Δ={ppl_u-baseline_ppl:+.4f})")

    # T-9 spectral: higher rank → more bits
    make_recipe(t9_rank, "t9_spectral",
                "Higher effective rank → more bits (harder to compress)")

    # T-7 linearity: higher gap (more nonlinear) → more bits
    make_recipe(t7_gap, "t7_linearity",
                "Higher linearity gap → more bits (more nonlinear)")

    # Combined T-7 + T-9
    make_recipe(0.5 * norm01(t9_rank) + 0.5 * norm01(t7_gap), "t7_t9_combined",
                "Combined spectral rank + linearity gap")

    # Empirical sensitivity-informed (oracle-like, using 2-bit signal)
    make_recipe(sensitivity_2b, "sensitivity_oracle",
                "Direct from measured 2-bit sensitivity (oracle)")

    # First/last protected (common heuristic)
    prot = np.full(num_layers, 4)
    prot[0] = 8; prot[num_layers - 1] = 8
    prot = adjust_bits_to_target(prot, target_avg)
    ppl_p = apply_mixed_precision_eval(model, eval_ids, prot, num_layers)
    recipes["first_last_protected"] = {
        "bits_per_layer": prot.tolist(), "avg_bits": float(prot.mean()),
        "ppl": ppl_p, "ppl_delta": ppl_p - baseline_ppl,
        "description": "First and last layers at 8-bit, rest adjusted",
    }
    print(f"  {'first_last_protected':28s}: PPL={ppl_p:.4f} (Δ={ppl_p-baseline_ppl:+.4f})")

    # Per-matrix mixed precision: Q/K at 3-bit, V/MLP at 5-bit
    print("\n  Per-matrix-type mixed precision (Q/K at 3b, V/MLP at 5b)...")
    orig_all = {}
    for li in range(num_layers):
        orig_all[li] = {}
        for m in ALL_MATRICES:
            w = get_weight(model, li, m)
            orig_all[li][m] = w.data.clone()
            b = 3 if m in ["q_proj", "k_proj"] else 5
            w.data.copy_(quantize_dequantize_rtn(w.data, b, RTN_GROUP_SIZE))
    ppl_mx = evaluate_perplexity(model, eval_ids)
    for li in range(num_layers):
        for m in ALL_MATRICES:
            get_weight(model, li, m).data.copy_(orig_all[li][m])
    del orig_all
    # avg bits: (2*3 + 5*5)/7 ≈ 4.43
    avg_mx = (2 * 3 + 5 * 5) / 7
    recipes["qk3_vmul5"] = {
        "bits_per_layer": [f"Q/K=3, V/MLP=5"] * num_layers,
        "avg_bits": avg_mx, "ppl": ppl_mx, "ppl_delta": ppl_mx - baseline_ppl,
        "description": "Q/K projections at 3-bit, V/O/MLP at 5-bit (avg ~4.4b)",
    }
    print(f"  {'qk3_vmul5':28s}: PPL={ppl_mx:.4f} (Δ={ppl_mx-baseline_ppl:+.4f}), avg={avg_mx:.2f}b")

    elapsed = time.time() - t0
    print(f"\n  Phase 3 done in {elapsed:.1f}s")
    return {"baseline_ppl": baseline_ppl, "target_avg_bits": target_avg,
            "recipes": recipes, "time_s": elapsed}


# ─── Phase 4: Correlation Analysis & Visualization ──────────────────────────

def run_phase4(phase1, phase1b, phase2, phase3, t7_data, t9_data, num_layers):
    print("\n" + "=" * 80)
    print("PHASE 4: Correlation Analysis & Visualization")
    print("=" * 80)

    layers = np.arange(num_layers)

    # ── extract arrays ──
    sens = {b: np.array([phase1["per_layer"][l]["ppl_delta"][b] for l in range(num_layers)])
            for b in RTN_BIT_WIDTHS}

    t7_pl = t7_data["per_layer"]
    t7_gap = np.array([t7_pl[f"layer_{l}"]["perturb_gap_mean"] for l in range(num_layers)])
    t7_attn = np.array([t7_pl[f"layer_{l}"]["attn_gap_mean"] for l in range(num_layers)])
    t7_mlp = np.array([t7_pl[f"layer_{l}"]["mlp_gap_mean"] for l in range(num_layers)])
    t7_order = np.array([t7_pl[f"layer_{l}"]["nonlinearity_order_mean"] for l in range(num_layers)])

    t9_rank_mat = {}
    t9_avg = []
    for l in range(num_layers):
        ld = t9_data["per_layer"][str(l)]
        rs = []
        for m in ALL_MATRICES:
            if m in ld:
                t9_rank_mat.setdefault(m, []).append(ld[m]["effective_rank_ratio"])
                rs.append(ld[m]["effective_rank_ratio"])
        t9_avg.append(np.mean(rs))
    t9_avg = np.array(t9_avg)
    for m in t9_rank_mat:
        t9_rank_mat[m] = np.array(t9_rank_mat[m])

    # ── correlations (use 2-bit as primary signal — 4-bit is below BF16 noise floor) ──
    correlations = {}
    pairs = {
        "t7_perturb_gap": t7_gap, "t7_attn_gap": t7_attn,
        "t7_mlp_gap": t7_mlp, "t7_nonlinearity_order": t7_order,
        "t9_avg_eff_rank": t9_avg,
    }
    for m in ALL_MATRICES:
        pairs[f"t9_{m}_rank"] = t9_rank_mat[m]

    for ref_bits in [2, 3, 4]:
        print(f"\n  Correlations with {ref_bits}-bit sensitivity (Spearman ρ):")
        for name, arr in pairs.items():
            rs, ps = stats.spearmanr(arr, sens[ref_bits])
            rp, pp = stats.pearsonr(arr, sens[ref_bits])
            key = f"{name}__vs_{ref_bits}bit"
            correlations[key] = {"spearman_r": rs, "spearman_p": ps,
                                  "pearson_r": rp, "pearson_p": pp}
            sig = "***" if ps < 0.001 else "**" if ps < 0.01 else "*" if ps < 0.05 else ""
            if ref_bits == 2:
                print(f"    {name:30s}: ρ={rs:+.3f} (p={ps:.4f}){sig}")

    # per-matrix correlations
    print("\n  Per-matrix sensitivity ↔ rank correlations:")
    mat_corr = {}
    for m in ALL_MATRICES:
        ms = np.array([phase1b["per_layer"][l][m]["ppl_delta"] for l in range(num_layers)])
        mr = t9_rank_mat[m]
        rs, ps = stats.spearmanr(mr, ms)
        mat_corr[m] = {"spearman_r": rs, "spearman_p": ps,
                        "sens_mean": float(ms.mean()), "sens_std": float(ms.std()),
                        "rank_mean": float(mr.mean())}
        sig = "***" if ps < 0.001 else "**" if ps < 0.01 else "*" if ps < 0.05 else ""
        print(f"    {m:12s}: ρ={rs:+.3f} (p={ps:.4f}){sig}  Δppl={ms.mean():.4f}±{ms.std():.4f}")

    # ── Visualization ──
    print("\n  Generating plots...")
    sns.set_style("whitegrid")

    # Plot 1: Sensitivity heatmap + scatter overview
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    ax = axes[0, 0]
    hm = np.array([[phase1["per_layer"][l]["ppl_delta"][b] for l in range(num_layers)]
                    for b in RTN_BIT_WIDTHS])
    im = ax.imshow(hm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(len(RTN_BIT_WIDTHS)))
    ax.set_yticklabels([f"{b}-bit" for b in RTN_BIT_WIDTHS])
    ax.set_xlabel("Layer"); ax.set_title("Per-Layer Quantization Sensitivity (PPL Δ)")
    plt.colorbar(im, ax=ax, label="PPL increase")

    # Use 2-bit for scatter plots (4-bit is below noise floor for single layers)
    s2 = sens[2]
    ax = axes[0, 1]
    sc = ax.scatter(t7_gap, s2, c=layers, cmap='viridis', s=50, edgecolors='k', lw=0.5)
    r, p = stats.spearmanr(t7_gap, s2)
    ax.set_xlabel("T-7 Linearity Gap"); ax.set_ylabel("2-bit PPL Δ")
    ax.set_title(f"Quant Sensitivity vs Linearity Gap\nρ={r:.3f}, p={p:.4f}")
    plt.colorbar(sc, ax=ax, label="Layer")
    thr = np.percentile(s2, 85)
    for l in range(num_layers):
        if s2[l] > thr:
            ax.annotate(str(l), (t7_gap[l], s2[l]), fontsize=7, ha='center')

    ax = axes[1, 0]
    sc = ax.scatter(t9_avg, s2, c=layers, cmap='viridis', s=50, edgecolors='k', lw=0.5)
    r, p = stats.spearmanr(t9_avg, s2)
    ax.set_xlabel("T-9 Avg Effective Rank Ratio"); ax.set_ylabel("2-bit PPL Δ")
    ax.set_title(f"Quant Sensitivity vs Spectral Rank\nρ={r:.3f}, p={p:.4f}")
    plt.colorbar(sc, ax=ax, label="Layer")
    for l in range(num_layers):
        if s2[l] > thr:
            ax.annotate(str(l), (t9_avg[l], s2[l]), fontsize=7, ha='center')

    ax = axes[1, 1]
    mat_data = {m: [phase1b["per_layer"][l][m]["ppl_delta"] for l in range(num_layers)]
                for m in ALL_MATRICES}
    bp = ax.boxplot([mat_data[m] for m in ALL_MATRICES], labels=ALL_MATRICES, patch_artist=True)
    colors = ['#4e79a7', '#59a14f', '#9c755f', '#f28e2b', '#e15759', '#76b7b2', '#edc948']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_ylabel(f"PPL Δ ({MATRIX_SENSITIVITY_BITS}-bit, single matrix)")
    ax.set_title("Per-Matrix-Type Sensitivity")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sensitivity_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Depth profiles
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    ax = axes[0]
    for b in RTN_BIT_WIDTHS:
        ax.plot(layers, sens[b], 'o-', ms=4, label=f"{b}-bit")
    ax.set_ylabel("PPL Δ"); ax.set_title("Per-Layer Quantization Sensitivity by Bit-Width")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    l1 = ax.plot(layers, t7_gap, 'o-', color='#e15759', ms=4, label="T-7 Linearity Gap")
    ax2 = ax.twinx()
    l2 = ax2.plot(layers, sens[2], 's-', color='#4e79a7', ms=4, label="2-bit Sensitivity")
    ax.set_ylabel("Linearity Gap", color='#e15759')
    ax2.set_ylabel("PPL Δ (2-bit)", color='#4e79a7')
    ax.set_title("Linearity Gap vs Quantization Sensitivity")
    ax.legend(l1 + l2, [x.get_label() for x in l1 + l2])
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    l1 = ax.plot(layers, t9_avg, 'o-', color='#59a14f', ms=4, label="T-9 Avg Rank")
    ax2 = ax.twinx()
    l2 = ax2.plot(layers, sens[2], 's-', color='#4e79a7', ms=4, label="2-bit Sensitivity")
    ax.set_ylabel("Effective Rank Ratio", color='#59a14f')
    ax2.set_ylabel("PPL Δ (2-bit)", color='#4e79a7')
    ax.set_xlabel("Layer Index")
    ax.set_title("Spectral Rank vs Quantization Sensitivity")
    ax.legend(l1 + l2, [x.get_label() for x in l1 + l2])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "depth_profiles.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Mixed-precision recipe comparison
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax = axes[0]
    rnames = [r for r in phase3["recipes"] if "error" not in phase3["recipes"][r]]
    rdeltas = [phase3["recipes"][r]["ppl_delta"] for r in rnames]
    ravg = [phase3["recipes"][r]["avg_bits"] for r in rnames]
    cols = plt.cm.Set2(np.linspace(0, 1, len(rnames)))
    bars = ax.bar(range(len(rnames)), rdeltas, color=cols, edgecolor='k', lw=0.5)
    ax.set_xticks(range(len(rnames))); ax.set_xticklabels(rnames, rotation=30, ha='right')
    ax.set_ylabel("PPL Δ from BF16"); ax.set_title("Mixed-Precision Recipes (all ~4-bit avg)")
    for bar, d, ab in zip(bars, rdeltas, ravg):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{d:.2f}\n({ab:.1f}b)", ha='center', va='bottom', fontsize=8)

    ax = axes[1]
    for r in rnames:
        bpl = phase3["recipes"][r]["bits_per_layer"]
        if isinstance(bpl[0], (int, float)):
            ax.plot(layers, bpl, 'o-', ms=3, label=r)
    ax.set_xlabel("Layer"); ax.set_ylabel("Bit-Width")
    ax.set_title("Bit Allocation per Layer"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mixed_precision_recipes.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Per-matrix sensitivity × rank
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, m in enumerate(ALL_MATRICES):
        ax = axes[i]
        ms = np.array([phase1b["per_layer"][l][m]["ppl_delta"] for l in range(num_layers)])
        mr = t9_rank_mat[m]
        ax.scatter(mr, ms, c=layers, cmap='viridis', s=40, edgecolors='k', lw=0.3)
        r, p = stats.spearmanr(mr, ms)
        ax.set_xlabel("Eff Rank Ratio"); ax.set_ylabel("PPL Δ")
        ax.set_title(f"{m}\nρ={r:.3f}, p={p:.3f}"); ax.grid(True, alpha=0.3)
    axes[-1].axis('off')
    plt.suptitle("Per-Matrix Quant Sensitivity vs Spectral Rank (4-bit RTN)", fontsize=14)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "matrix_sensitivity_vs_rank.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Quantization error vs perplexity impact
    fig, ax = plt.subplots(figsize=(10, 8))
    for b in [4, 3, 2]:
        errs = [np.mean(list(phase1["per_layer"][l]["quant_error"][b].values()))
                for l in range(num_layers)]
        ax.scatter(errs, sens[b], label=f"{b}-bit", s=40, alpha=0.7)
        r, p = stats.spearmanr(errs, sens[b])
        ax.annotate(f"ρ={r:.2f}", xy=(max(errs), max(sens[b])), fontsize=9)
    ax.set_xlabel("Mean Relative Quant Error (Frobenius)")
    ax.set_ylabel("PPL Δ"); ax.set_title("Quantization Error vs Perplexity Impact")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "error_vs_impact.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 6: Full-model methods comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    mlist = [(k, v) for k, v in phase2["methods"].items() if "error" not in v]
    mnames = [k for k, _ in mlist]
    mppls = [v["ppl"] for _, v in mlist]
    mspeeds = [v.get("tokens_per_sec", 0) for _, v in mlist]

    ax = axes[0]
    ax.barh(mnames, mppls, color=plt.cm.Paired(np.linspace(0, 1, len(mnames))))
    ax.set_xlabel("Perplexity"); ax.set_title("Full-Model Methods: Perplexity")
    ax.axvline(x=phase2["baseline_ppl"], color='r', ls='--', label='BF16')

    ax = axes[1]
    ax.barh(mnames, mspeeds, color=plt.cm.Paired(np.linspace(0, 1, len(mnames))))
    ax.set_xlabel("Tokens/sec"); ax.set_title("Full-Model Methods: Generation Speed")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "methods_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("  All plots saved.")
    return {"correlations": correlations, "matrix_correlations": mat_corr}


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("T-11: Quantization Methods Comparative Analysis")
    print(f"Model: {MODEL_ID}  |  Device: {DEVICE}  |  Seed: {SEED}")
    print("=" * 80)
    overall_t0 = time.time()

    # Load model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE)
    model.eval()
    num_layers = len(model.model.layers)
    bf16_mem = torch.cuda.memory_allocated(DEVICE)
    print(f"  {num_layers} layers, {bf16_mem/1e9:.2f} GB")

    # Eval data
    print("\nPreparing evaluation data...")
    eval_ids = prepare_eval_data(tokenizer, EVAL_SEQ_LEN)

    # Load prior results
    print("\nLoading T-7 and T-9 results...")
    with open(T7_RESULTS) as f:
        t7_data = json.load(f)
    with open(T9_RESULTS) as f:
        t9_data = json.load(f)

    # Phase 1
    p1 = run_phase1(model, eval_ids, num_layers)
    with open(RESULTS_DIR / "phase1_per_layer.json", "w") as f:
        json.dump(p1, f, indent=2)

    # Phase 1b
    p1b = run_phase1b(model, eval_ids, num_layers, p1["baseline_ppl"])
    with open(RESULTS_DIR / "phase1b_per_matrix.json", "w") as f:
        json.dump(p1b, f, indent=2)

    # Phase 3 (before Phase 2 which loads fresh models)
    p3 = run_phase3(model, eval_ids, num_layers, p1["baseline_ppl"], t7_data, t9_data, p1)
    with open(RESULTS_DIR / "phase3_mixed_precision.json", "w") as f:
        json.dump(p3, f, indent=2)

    # Phase 2
    p2 = run_phase2(model, tokenizer, eval_ids, num_layers, p1["baseline_ppl"])
    with open(RESULTS_DIR / "phase2_full_model.json", "w") as f:
        json.dump(p2, f, indent=2)

    # Phase 4
    p4 = run_phase4(p1, p1b, p2, p3, t7_data, t9_data, num_layers)
    with open(RESULTS_DIR / "phase4_correlations.json", "w") as f:
        json.dump(p4, f, indent=2)

    # Summary
    total = time.time() - overall_t0
    summary = {
        "model": MODEL_ID, "device": DEVICE, "seed": SEED,
        "eval_tokens": int(eval_ids.size(1)), "num_layers": num_layers,
        "bf16_memory_gb": bf16_mem / 1e9,
        "baseline_ppl": p1["baseline_ppl"],
        "total_time_s": total,
        "phase_times_s": {
            "phase1": p1["time_s"], "phase1b": p1b["time_s"],
            "phase2": p2["time_s"], "phase3": p3["time_s"],
        },
        "key_findings": {
            "most_sensitive_layers_2bit": sorted(
                range(num_layers),
                key=lambda l: p1["per_layer"][l]["ppl_delta"][2], reverse=True)[:5],
            "least_sensitive_layers_2bit": sorted(
                range(num_layers),
                key=lambda l: p1["per_layer"][l]["ppl_delta"][2])[:5],
            "most_sensitive_matrix_type": max(
                ALL_MATRICES,
                key=lambda m: np.mean([p1b["per_layer"][l][m]["ppl_delta"]
                                       for l in range(num_layers)])),
            "least_sensitive_matrix_type": min(
                ALL_MATRICES,
                key=lambda m: np.mean([p1b["per_layer"][l][m]["ppl_delta"]
                                       for l in range(num_layers)])),
            "best_recipe": min(
                [(k, v["ppl_delta"]) for k, v in p3["recipes"].items()],
                key=lambda x: x[1]),
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"T-11 COMPLETE — {total:.0f}s ({total/60:.1f}m)")
    print(f"Baseline PPL: {p1['baseline_ppl']:.4f}")
    kf = summary["key_findings"]
    print(f"Most sensitive layers (2-bit): {kf['most_sensitive_layers_2bit']}")
    print(f"Least sensitive layers (2-bit): {kf['least_sensitive_layers_2bit']}")
    print(f"Most sensitive matrix type: {kf['most_sensitive_matrix_type']}")
    print(f"Best mixed-precision recipe: {kf['best_recipe']}")
    print(f"Results: {RESULTS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
