"""
Generate calibration data for layer shuffle recovery experiment.

Uses vLLM to generate rollouts from Qwen3-1.7B on GSM8K prompts.
Saves tokenized input_ids to disk for use by run.py.

Usage: poetry run python experiments/layer_shuffle_recovery/generate_calibration.py
"""

import os
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-1.7B"
NUM_SAMPLES = 64
MAX_LEN = 256
OUTPUT_DIR = Path(__file__).parent / "data"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GSM8K prompts
    print("Loading GSM8K questions...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [ex["question"] for ex in ds.select(range(min(NUM_SAMPLES, len(ds))))]
    print(f"Collected {len(prompts)} prompts")

    # Generate rollouts with vLLM
    print("Starting vLLM for rollout generation...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_LEN,
    )

    outputs = llm.generate(prompts, sampling_params)
    texts = [f"{prompts[i]} {outputs[i].outputs[0].text}" for i in range(len(outputs))]
    print(f"Generated {len(texts)} rollouts")

    # Free vLLM
    del llm
    torch.cuda.empty_cache()

    # Tokenize
    encodings = tokenizer(
        texts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=MAX_LEN,
    )

    # Save
    torch.save(encodings["input_ids"], OUTPUT_DIR / "input_ids.pt")

    # Save metadata
    meta = {
        "model": MODEL_NAME,
        "source": "openai/gsm8k",
        "num_samples": len(texts),
        "max_len": MAX_LEN,
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save raw texts for inspection
    with open(OUTPUT_DIR / "texts.json", "w") as f:
        json.dump(texts, f, indent=2)

    print(f"Saved to {OUTPUT_DIR}/")
    print(f"  input_ids.pt: {encodings['input_ids'].shape}")


if __name__ == "__main__":
    main()
