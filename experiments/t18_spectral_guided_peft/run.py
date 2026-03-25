"""
T-18: Spectral-Guided PEFT Validation
======================================
Validate spectral predictions from T-9, T-7, T-4 by training LoRA adapters
with different rank/layer configurations on Qwen3-4B-Instruct-2507.

4 configs trained in 2 rounds (2 GPUs parallel), plus post-training spectral analysis.
Total runtime: ~1.5-2h on 2x B200.
"""

import json
import time
import re
import random
import gc
import os
import sys
import subprocess
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
COMPLETIONS_PATH = (
    Path(__file__).parents[2]
    / "data/text_completions/qwen3-4b-instruct-2507/completions.json"
)
T9_RESULTS_PATH = (
    Path(__file__).parents[1]
    / "t9_weight_spectral_structure/results/summary.json"
)

# Training
NUM_TRAIN = 2000
NUM_EPOCHS = 2
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4
MAX_SEQ_LEN = 512
LORA_DROPOUT = 0.05

ALL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
NUM_LAYERS = 36


# ── Utilities ──────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_spectral_stats(weight_matrix: torch.Tensor) -> dict:
    """Compute SVD-based spectral statistics for a weight matrix (from T-9)."""
    W = weight_matrix.float()
    S = torch.linalg.svdvals(W)
    S_np = S.detach().cpu().numpy()

    rank = int((S > 1e-6).sum().item())
    s2 = S_np ** 2
    s4 = S_np ** 4
    effective_rank = float(s2.sum() ** 2 / s4.sum()) if s4.sum() > 0 else 0.0
    max_possible_rank = min(W.shape)
    effective_rank_ratio = effective_rank / max_possible_rank

    # Power-law fit
    mask = S_np > 1e-6
    S_fit = S_np[mask]
    if len(S_fit) > 2:
        log_idx = np.log(np.arange(1, len(S_fit) + 1))
        log_sv = np.log(S_fit)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_idx, log_sv)
        power_law_exp = float(-slope)
    else:
        power_law_exp = 0.0

    return {
        "shape": list(W.shape),
        "effective_rank": round(effective_rank, 2),
        "effective_rank_ratio": round(effective_rank_ratio, 4),
        "max_rank": max_possible_rank,
        "power_law_exponent": round(power_law_exp, 4),
        "top_5_sv": [round(float(v), 4) for v in S_np[:5]],
    }


# ── Data Loading ───────────────────────────────────────────────────────────

WEIGHT_NAMES = {
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


class GSM8KDataset(Dataset):
    """GSM8K formatted for causal LM training with chat template."""

    def __init__(self, tokenizer, split="train", max_samples=None, max_length=512):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        self.input_ids_list = []
        self.labels_list = []
        self.tokenizer = tokenizer

        for ex in ds:
            question = ex["question"]
            answer = ex["answer"]

            messages = [
                {"role": "system", "content": "Solve the math problem step by step. End with #### followed by the final numerical answer."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            encoded = tokenizer(
                text, truncation=True, max_length=max_length,
                return_tensors="pt", padding=False,
            )
            input_ids = encoded["input_ids"].squeeze(0)

            # Mask prompt tokens in labels (only train on assistant response)
            prompt_messages = [
                {"role": "system", "content": "Solve the math problem step by step. End with #### followed by the final numerical answer."},
                {"role": "user", "content": question},
            ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_len = len(tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze(0))

            labels = input_ids.clone()
            labels[:prompt_len] = -100  # Mask prompt tokens

            self.input_ids_list.append(input_ids)
            self.labels_list.append(labels)

        log(f"  Loaded {len(self.input_ids_list)} GSM8K examples (split={split})")

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "labels": self.labels_list[idx],
            "attention_mask": torch.ones_like(self.input_ids_list[idx]),
        }


class PaddingCollator:
    """Pad batch to max length with left-padding for input_ids, right-padding for labels."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)
        pad_id = self.tokenizer.pad_token_id or 0

        batch_input_ids = []
        batch_labels = []
        batch_attention = []

        for f in features:
            pad_len = max_len - f["input_ids"].size(0)
            batch_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), f["input_ids"]])
            )
            batch_labels.append(
                torch.cat([torch.full((pad_len,), -100, dtype=torch.long), f["labels"]])
            )
            batch_attention.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long), f["attention_mask"]])
            )

        return {
            "input_ids": torch.stack(batch_input_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_attention),
        }


# ── Evaluation ─────────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract numerical answer after #### from model output."""
    match = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # Fallback: last number in text
    numbers = re.findall(r"[+-]?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = text.index("{", idx)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i].strip()
    return None


def normalize_math_answer(ans: str) -> str:
    """Normalize a math answer for comparison."""
    if ans is None:
        return ""
    ans = ans.strip()
    # Remove \left, \right, \, spacing commands
    for cmd in ["\\left", "\\right", "\\,", "\\;", "\\!", "\\:"]:
        ans = ans.replace(cmd, "")
    # Normalize whitespace
    ans = re.sub(r"\s+", " ", ans).strip()
    # Try numeric comparison
    return ans


def answers_match(pred: Optional[str], gold: str) -> bool:
    """Check if predicted and gold answers match (exact or numeric)."""
    if pred is None:
        return False
    pred_n = normalize_math_answer(pred)
    gold_n = normalize_math_answer(gold)
    if pred_n == gold_n:
        return True
    # Try numeric comparison
    try:
        return abs(float(pred_n) - float(gold_n)) < 1e-3
    except (ValueError, TypeError):
        return False


def evaluate_math500(model, tokenizer, device, max_samples=None):
    """Evaluate on MATH-500 benchmark. Returns accuracy."""
    ds = load_dataset("math-ai/MATH-500", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    model.eval()
    correct = 0
    total = 0

    for i, ex in enumerate(ds):
        messages = [
            {"role": "system", "content": "Solve the math problem step by step. Put your final answer in \\boxed{}."},
            {"role": "user", "content": ex["problem"]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = extract_boxed(response)
        gold = ex["answer"].strip()

        if answers_match(pred, gold):
            correct += 1
        total += 1

        if (i + 1) % 50 == 0:
            log(f"    Eval progress: {i+1}/{len(ds)}, acc so far: {correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def evaluate_gsm8k(model, tokenizer, device, max_samples=None, dataset_name="gsm8k"):
    """Evaluate on GSM8K test set. Returns accuracy and per-example results."""
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    elif dataset_name == "svamp":
        ds = load_dataset("ChilleD/SVAMP", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    model.eval()
    correct = 0
    total = 0

    for i, ex in enumerate(ds):
        if dataset_name == "gsm8k":
            question = ex["question"]
            gold_answer = extract_gsm8k_answer(ex["answer"])
        else:  # svamp
            question = ex["Body"] + " " + ex["Question"]
            gold_answer = str(ex["Answer"])

        messages = [
            {"role": "system", "content": "Solve the math problem step by step. End with #### followed by the final numerical answer."},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_gsm8k_answer(response)

        if pred_answer and gold_answer:
            try:
                if abs(float(pred_answer) - float(gold_answer)) < 1e-3:
                    correct += 1
            except ValueError:
                pass
        total += 1

        if (i + 1) % 100 == 0:
            log(f"    Eval progress: {i+1}/{len(ds)}, acc so far: {correct/total:.3f}")

    accuracy = correct / total if total > 0 else 0
    return {"accuracy": accuracy, "correct": correct, "total": total}


def compute_calibration_perplexity(model, tokenizer, device):
    """Compute perplexity on existing calibration completions."""
    with open(COMPLETIONS_PATH) as f:
        data = json.load(f)

    total_loss = 0.0
    total_tokens = 0
    model.eval()

    for comp in data["completions"][:50]:
        text = comp["full_text"]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

    avg_loss = total_loss / total_tokens
    return {"perplexity": float(np.exp(avg_loss)), "avg_loss": round(avg_loss, 4)}


# ── LoRA Config Builders ──────────────────────────────────────────────────

def build_config_A():
    """Uniform rank=16, all modules, all layers."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=ALL_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_config_B():
    """Phase-aware: spectral-guided rank allocation from T-9."""
    phase_ranks = {
        "q_proj":    {"plateau": 8,  "late": 24},
        "k_proj":    {"plateau": 8,  "late": 24},
        "v_proj":    {"plateau": 24, "late": 24},
        "o_proj":    {"plateau": 12, "late": 24},
        "gate_proj": {"plateau": 12, "late": 48},
        "up_proj":   {"plateau": 24, "late": 24},
        "down_proj": {"plateau": 24, "late": 24},
    }

    rank_pattern = {}
    alpha_pattern = {}
    for layer_idx in range(NUM_LAYERS):
        region = "plateau" if layer_idx <= 16 else "late"
        for module, ranks in phase_ranks.items():
            key = f"model.layers.{layer_idx}.{'self_attn' if module in ('q_proj','k_proj','v_proj','o_proj') else 'mlp'}.{module}"
            r = ranks[region]
            rank_pattern[key] = r
            alpha_pattern[key] = r * 2  # Maintain 2x alpha/rank ratio

    return LoraConfig(
        r=16,  # Default (overridden by rank_pattern)
        lora_alpha=32,
        target_modules=ALL_MODULES,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_config_F():
    """Skip bottleneck: LoRA on layers 0-14 + 25-35 only."""
    layers = list(range(0, 15)) + list(range(25, 36))
    target_modules = []
    for layer_idx in layers:
        for module in ALL_MODULES:
            sublayer = "self_attn" if module in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            target_modules.append(f"model.layers.{layer_idx}.{sublayer}.{module}")

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_config_H():
    """Late layers only: LoRA on layers 25-35."""
    layers = list(range(25, 36))
    target_modules = []
    for layer_idx in layers:
        for module in ALL_MODULES:
            sublayer = "self_attn" if module in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            target_modules.append(f"model.layers.{layer_idx}.{sublayer}.{module}")

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


CONFIG_BUILDERS = {
    "A": ("Uniform r=16 all layers", build_config_A),
    "B": ("Phase-aware spectral-guided", build_config_B),
    "F": ("Skip bottleneck L15-24", build_config_F),
    "H": ("Late only L25-35", build_config_H),
}


# ── Training ───────────────────────────────────────────────────────────────

def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train_config(config_name, device, tokenizer, train_dataset):
    """Train a single LoRA config and return results."""
    desc, builder = CONFIG_BUILDERS[config_name]
    log(f"{'='*60}")
    log(f"Training Config {config_name}: {desc} on {device}")
    log(f"{'='*60}")

    t0 = time.time()

    # Load fresh base model
    log(f"  Loading base model on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = builder()
    model = get_peft_model(model, lora_config)
    trainable, total = count_trainable_params(model)
    log(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # Training
    output_dir = RESULTS_DIR / f"checkpoints_{config_name}"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=PaddingCollator(tokenizer),
    )

    train_result = trainer.train()
    train_time = time.time() - t0
    log(f"  Training done in {train_time:.0f}s. Loss: {train_result.training_loss:.4f}")

    # Enable cache for generation
    model.config.use_cache = True

    # Evaluate
    log(f"  Evaluating GSM8K test (200 samples)...")
    gsm8k_results = evaluate_gsm8k(model, tokenizer, device, max_samples=200)
    log(f"  GSM8K accuracy: {gsm8k_results['accuracy']:.3f} ({gsm8k_results['correct']}/{gsm8k_results['total']})")

    log(f"  Evaluating MATH-500 (OOD)...")
    math500_results = evaluate_math500(model, tokenizer, device)
    log(f"  MATH-500 accuracy: {math500_results['accuracy']:.3f} ({math500_results['correct']}/{math500_results['total']})")

    log(f"  Computing calibration perplexity...")
    ppl_results = compute_calibration_perplexity(model, tokenizer, device)
    log(f"  Calibration perplexity: {ppl_results['perplexity']:.2f}")

    results = {
        "config": config_name,
        "description": desc,
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": round(train_result.training_loss, 4),
        "train_time_s": round(train_time, 1),
        "gsm8k": gsm8k_results,
        "math500": math500_results,
        "calibration": ppl_results,
    }

    # Save results
    with open(RESULTS_DIR / f"config_{config_name}.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Config {config_name} total time: {time.time() - t0:.0f}s")

    # Return model for spectral analysis if needed
    return model, results


# ── Spectral Analysis ──────────────────────────────────────────────────────

def run_spectral_analysis(model, config_name, device):
    """Compare merged weight spectral structure vs pre-training baseline (T-9)."""
    log(f"Phase 3: Spectral analysis for Config {config_name}")

    # Merge LoRA into base weights
    log("  Merging LoRA weights...")
    merged_model = model.merge_and_unload()

    # Compute spectral stats on merged weights
    log("  Computing SVD on all 252 merged weight matrices...")
    post_spectral = {}
    for layer_idx in range(NUM_LAYERS):
        layer = merged_model.model.layers[layer_idx]
        post_spectral[layer_idx] = {}
        for name, attr_path in WEIGHT_NAMES.items():
            parts = attr_path.split(".")
            w = layer
            for p in parts:
                w = getattr(w, p)
            stats_dict = compute_spectral_stats(w.data)
            post_spectral[layer_idx][name] = stats_dict

    # Load T-9 pre-training baseline
    log("  Loading T-9 baseline spectral data...")
    with open(T9_RESULTS_PATH) as f:
        t9_data = json.load(f)

    # Compute deltas
    spectral_delta = {}
    for layer_idx in range(NUM_LAYERS):
        spectral_delta[layer_idx] = {}
        t9_layer = t9_data["per_layer"][str(layer_idx)]
        for matrix_name in WEIGHT_NAMES:
            pre = t9_layer[matrix_name]
            post = post_spectral[layer_idx][matrix_name]
            delta_eff_rank = post["effective_rank_ratio"] - pre["effective_rank_ratio"]
            pre_power = pre["power_law"]["exponent"] if isinstance(pre.get("power_law"), dict) else pre.get("power_law_exponent", 0)
            post_power = post["power_law_exponent"]
            delta_power = post_power - pre_power

            spectral_delta[layer_idx][matrix_name] = {
                "pre_eff_rank_ratio": pre["effective_rank_ratio"],
                "post_eff_rank_ratio": post["effective_rank_ratio"],
                "delta_eff_rank_ratio": round(delta_eff_rank, 4),
                "pre_power_law": round(pre_power, 4),
                "post_power_law": round(post_power, 4),
                "delta_power_law": round(delta_power, 4),
            }

    # Compute participation ratio of hidden states at key layers
    log("  Computing post-training participation ratio at key layers...")
    # (We'll skip this if we can't easily get hidden states from merged model —
    #  the weight-level spectral delta is the primary analysis)

    results = {
        "config": config_name,
        "post_spectral": {str(k): v for k, v in post_spectral.items()},
        "spectral_delta": {str(k): v for k, v in spectral_delta.items()},
    }

    with open(RESULTS_DIR / f"spectral_analysis_{config_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    log("  Spectral analysis saved.")
    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_results(all_results, spectral_results=None):
    """Generate comparison plots."""
    log("Generating plots...")

    configs = [r["config"] for r in all_results]
    gsm8k_acc = [r["gsm8k"]["accuracy"] for r in all_results]
    math500_acc = [r["math500"]["accuracy"] for r in all_results]
    ppl = [r["calibration"]["perplexity"] for r in all_results]
    params = [r["trainable_params"] for r in all_results]

    colors = {"A": "#4C72B0", "B": "#DD8452", "F": "#55A868", "H": "#C44E52"}

    # ── Plot 1: Accuracy comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GSM8K
    bars = axes[0].bar(configs, gsm8k_acc, color=[colors[c] for c in configs], edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("GSM8K (In-Distribution)")
    axes[0].set_ylim(0, max(gsm8k_acc) * 1.15)
    for bar, val in zip(bars, gsm8k_acc):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # MATH-500
    bars = axes[1].bar(configs, math500_acc, color=[colors[c] for c in configs], edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("MATH-500 (OOD)")
    axes[1].set_ylim(0, max(math500_acc) * 1.15)
    for bar, val in zip(bars, math500_acc):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Perplexity
    bars = axes[2].bar(configs, ppl, color=[colors[c] for c in configs], edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Perplexity")
    axes[2].set_title("Calibration Perplexity")
    for bar, val in zip(bars, ppl):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Add config descriptions as x-tick labels
    desc_map = {
        "A": "Uniform r=16\nAll layers",
        "B": "Phase-aware\nSpectral-guided",
        "F": "Skip bottleneck\nL0-14 + L25-35",
        "H": "Late only\nL25-35",
    }
    for ax in axes:
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels([f"{c}\n{desc_map.get(c, c)}" for c in configs], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: Parameter efficiency ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in all_results:
        c = r["config"]
        ax.scatter(r["trainable_params"] / 1e6, r["gsm8k"]["accuracy"],
                   color=colors[c], s=150, zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(f"Config {c}", (r["trainable_params"] / 1e6, r["gsm8k"]["accuracy"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight="bold")
        # Add MATH-500 as hollow marker
        ax.scatter(r["trainable_params"] / 1e6, r["math500"]["accuracy"],
                   color=colors[c], s=150, zorder=5, edgecolors="black",
                   linewidth=0.5, marker="^", facecolors="none")

    ax.set_xlabel("Trainable Parameters (M)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Parameter Efficiency: GSM8K (circles) vs MATH-500 (triangles)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "parameter_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 3: Spectral delta (if available) ──
    if spectral_results:
        delta = spectral_results["spectral_delta"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        matrix_types = list(WEIGHT_NAMES.keys())
        cmap = plt.cm.tab10

        for mi, mtype in enumerate(matrix_types):
            layers_x = list(range(NUM_LAYERS))
            deltas_y = [delta[str(l)][mtype]["delta_eff_rank_ratio"] for l in layers_x]
            axes[0].plot(layers_x, deltas_y, label=mtype, color=cmap(mi), alpha=0.7, linewidth=1.5)

        axes[0].axhline(0, color="black", linewidth=0.5, linestyle="--")
        axes[0].axvspan(15, 24, alpha=0.1, color="red", label="Bottleneck")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("Δ Effective Rank Ratio")
        axes[0].set_title("Spectral Change from LoRA Training (Config B)")
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].grid(alpha=0.3)

        # Aggregate by matrix type
        for mi, mtype in enumerate(matrix_types):
            mean_delta = np.mean([delta[str(l)][mtype]["delta_eff_rank_ratio"] for l in range(NUM_LAYERS)])
            axes[1].bar(mi, mean_delta, color=cmap(mi), edgecolor="black", linewidth=0.5)

        axes[1].set_xticks(range(len(matrix_types)))
        axes[1].set_xticklabels(matrix_types, rotation=45, ha="right")
        axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
        axes[1].set_ylabel("Mean Δ Effective Rank Ratio")
        axes[1].set_title("Average Spectral Change by Matrix Type")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "spectral_delta.png", dpi=150, bbox_inches="tight")
        plt.close()

    log("Plots saved.")


# ── Main ───────────────────────────────────────────────────────────────────

def run_single_config(config_name, device):
    """Run a single config end-to-end (for subprocess parallelism)."""
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"Loading GSM8K train data...")
    train_dataset = GSM8KDataset(tokenizer, split="train", max_samples=NUM_TRAIN, max_length=MAX_SEQ_LEN)

    model, results = train_config(config_name, device, tokenizer, train_dataset)

    # Spectral analysis only for Config B
    if config_name == "B":
        spectral = run_spectral_analysis(model, config_name, device)

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ── Phase 0: Baselines ──
    log("=" * 60)
    log("Phase 0: Computing baselines")
    log("=" * 60)

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    baseline_file = RESULTS_DIR / "baseline.json"
    if baseline_file.exists():
        log("Baseline already computed, loading from cache...")
        with open(baseline_file) as f:
            baseline_results = json.load(f)
        log(f"  Base GSM8K: {baseline_results['gsm8k']['accuracy']:.3f}, MATH-500: {baseline_results['math500']['accuracy']:.3f}, PPL: {baseline_results['calibration']['perplexity']:.2f}")
    else:
        log("Loading base model for baseline eval...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda:0",
        )

        log("Evaluating base model on GSM8K (200 samples)...")
        base_gsm8k = evaluate_gsm8k(base_model, tokenizer, "cuda:0", max_samples=200)
        log(f"  Base GSM8K accuracy: {base_gsm8k['accuracy']:.3f}")

        log("Evaluating base model on MATH-500 (OOD)...")
        base_math500 = evaluate_math500(base_model, tokenizer, "cuda:0")
        log(f"  Base MATH-500 accuracy: {base_math500['accuracy']:.3f}")

        log("Computing base calibration perplexity...")
        base_ppl = compute_calibration_perplexity(base_model, tokenizer, "cuda:0")
        log(f"  Base calibration perplexity: {base_ppl['perplexity']:.2f}")

        baseline_results = {
            "gsm8k": base_gsm8k,
            "math500": base_math500,
            "calibration": base_ppl,
        }
        with open(baseline_file, "w") as f:
            json.dump(baseline_results, f, indent=2)

        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Phases 1+2: Train configs in parallel rounds ──
    log("=" * 60)
    log("Phases 1+2: Training LoRA configs (2 rounds, 2 GPUs)")
    log("=" * 60)

    # Prepare training data (shared across configs)
    log("Loading GSM8K training data...")
    train_dataset = GSM8KDataset(tokenizer, split="train", max_samples=NUM_TRAIN, max_length=MAX_SEQ_LEN)

    # All configs run sequentially on cuda:0 (training is ~2.5 min each;
    # multi-GPU device_map causes embedding device mismatches with peft)
    config_order = ["A", "B", "F", "H"]
    device = "cuda:0"

    all_results = []
    spectral_result = None
    spectral_file = RESULTS_DIR / "spectral_analysis_B.json"
    if spectral_file.exists():
        with open(spectral_file) as f:
            spectral_result = json.load(f)

    for cfg in config_order:
        # Skip already-completed configs (resume support)
        existing = RESULTS_DIR / f"config_{cfg}.json"
        if existing.exists():
            log(f"Config {cfg} already completed, loading results from {existing}")
            with open(existing) as f:
                res = json.load(f)
            all_results.append(res)
            continue

        model, res = train_config(cfg, device, tokenizer, train_dataset)

        # Spectral analysis on Config B
        if cfg == "B":
            spectral_result = run_spectral_analysis(model, "B", device)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        all_results.append(res)

    # ── Summary ──
    log("=" * 60)
    log("Summary")
    log("=" * 60)

    summary = {
        "baseline": baseline_results,
        "configs": {r["config"]: r for r in all_results},
        "total_time_s": round(time.time() - t_start, 1),
        "predictions": {
            "B_beats_A_gsm8k": all_results[1]["gsm8k"]["accuracy"] > all_results[0]["gsm8k"]["accuracy"],
            "F_vs_A_gsm8k_gap": round(all_results[0]["gsm8k"]["accuracy"] - all_results[2]["gsm8k"]["accuracy"], 4),
            "F_vs_A_math500_gap": round(all_results[0]["math500"]["accuracy"] - all_results[2]["math500"]["accuracy"], 4),
            "H_vs_A_gsm8k_gap": round(all_results[0]["gsm8k"]["accuracy"] - all_results[3]["gsm8k"]["accuracy"], 4),
            "H_vs_A_math500_gap": round(all_results[0]["math500"]["accuracy"] - all_results[3]["math500"]["accuracy"], 4),
        },
    }

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    log(f"\n{'Config':<10} {'GSM8K':>8} {'MATH500':>8} {'PPL':>8} {'Params':>10} {'Time':>8}")
    log(f"{'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
    log(f"{'Base':<10} {baseline_results['gsm8k']['accuracy']:>8.3f} {baseline_results['math500']['accuracy']:>8.3f} {baseline_results['calibration']['perplexity']:>8.1f} {'—':>10} {'—':>8}")
    for r in all_results:
        log(f"{r['config']:<10} {r['gsm8k']['accuracy']:>8.3f} {r['math500']['accuracy']:>8.3f} {r['calibration']['perplexity']:>8.1f} {r['trainable_params']:>10,} {r['train_time_s']:>7.0f}s")

    # Key predictions
    log(f"\nPrediction checks:")
    log(f"  B > A on GSM8K? {'✓' if summary['predictions']['B_beats_A_gsm8k'] else '✗'}")
    log(f"  F vs A GSM8K gap: {summary['predictions']['F_vs_A_gsm8k_gap']:+.3f} (predicted <0.02)")
    log(f"  F vs A MATH-500 gap: {summary['predictions']['F_vs_A_math500_gap']:+.3f} (predicted >0.05)")
    log(f"  H vs A MATH-500 gap: {summary['predictions']['H_vs_A_math500_gap']:+.3f} (predicted >0.05)")

    # Plot
    plot_results(all_results, spectral_result)

    log(f"\nTotal experiment time: {time.time() - t_start:.0f}s ({(time.time() - t_start)/60:.1f} min)")
    log(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Run single config: python run.py --single <CONFIG> <DEVICE>
        config_name = sys.argv[2]
        device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
        run_single_config(config_name, device)
    else:
        main()
