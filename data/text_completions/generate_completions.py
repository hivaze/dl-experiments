"""
Generate greedy completions for a model using vLLM.

Usage:
    poetry run python data/text_completions/generate_completions.py
    poetry run python data/text_completions/generate_completions.py --model Qwen/Qwen3-4B-Thinking-2507 --max-tokens 256

Output is saved to data/text_completions/<model-slug>/completions.json.
Prompts are loaded from data/text_completions/prompts.json.
"""

import argparse
import json
import re
import time
from pathlib import Path

from vllm import LLM, SamplingParams

DATA_DIR = Path(__file__).parent
PROMPTS_PATH = DATA_DIR / "prompts.json"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_GPU_MEM = 0.5


def model_slug(model_name: str) -> str:
    """Convert 'Qwen/Qwen3-4B-Instruct-2507' → 'qwen3-4b-instruct-2507'."""
    name = model_name.split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]", "-", name).lower()


def load_prompts():
    with open(PROMPTS_PATH) as f:
        data = json.load(f)
    return data.get("system_message"), data["prompts"]


def main():
    parser = argparse.ArgumentParser(description="Generate greedy completions via vLLM")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max new tokens per prompt")
    parser.add_argument("--gpu-memory", type=float, default=DEFAULT_GPU_MEM, help="vLLM GPU memory utilization")
    parser.add_argument("--device", default="cuda:0", help="GPU device (only affects vLLM tensor placement)")
    args = parser.parse_args()

    slug = model_slug(args.model)
    output_dir = DATA_DIR / slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "completions.json"

    system_message, prompts = load_prompts()
    print(f"Model: {args.model}")
    print(f"Slug: {slug}")
    print(f"Prompts: {len(prompts)} (from {PROMPTS_PATH})")
    print(f"System message: {system_message}")
    print(f"Max new tokens: {args.max_tokens}")
    print(f"Output: {output_path}")

    t0 = time.time()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory,
    )
    tokenizer = llm.get_tokenizer()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Build chat-templated prompts (with system message if provided)
    templated_prompts = []
    for p in prompts:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": p["text"]})
        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        templated_prompts.append(templated)

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    print(f"\nGenerating completions...")
    t1 = time.time()
    outputs = llm.generate(templated_prompts, sampling_params)
    t2 = time.time()
    print(f"Generation done in {t2 - t1:.1f}s")

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for i, output in enumerate(outputs):
        prompt_entry = prompts[i]
        templated_text = templated_prompts[i]
        completion_text = output.outputs[0].text

        prompt_token_ids = output.prompt_token_ids
        completion_token_ids = list(output.outputs[0].token_ids)

        full_text = templated_text + completion_text
        full_token_ids = prompt_token_ids + completion_token_ids

        total_prompt_tokens += len(prompt_token_ids)
        total_completion_tokens += len(completion_token_ids)

        results.append({
            "idx": i,
            "prompt": prompt_entry["text"],
            "category": prompt_entry["category"],
            "templated_prompt": templated_text,
            "completion": completion_text,
            "full_text": full_text,
            "prompt_token_ids": prompt_token_ids,
            "completion_token_ids": completion_token_ids,
            "full_token_ids": full_token_ids,
            "prompt_token_count": len(prompt_token_ids),
            "completion_token_count": len(completion_token_ids),
        })

        preview = completion_text[:80].replace("\n", "\\n")
        print(f"  [{i:2d}] {prompt_entry['text'][:50]:50s} → {preview}...")

    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "model_slug": slug,
                "max_new_tokens": args.max_tokens,
                "temperature": 0,
                "num_prompts": len(prompts),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "prompts_source": str(PROMPTS_PATH),
            },
            "completions": results,
        }, f, indent=2)

    print(f"\nSaved {len(results)} completions to {output_path}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Avg completion length: {total_completion_tokens / len(results):.1f} tokens")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
