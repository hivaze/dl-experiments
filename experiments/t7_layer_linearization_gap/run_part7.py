"""
Run only Method 7 (Global Linear Replacement) using pre-existing T-2 results.
Loads T-2 results for knockout data, runs replacement training,
generates the replacement plot, and updates summary.json.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import (
    MODEL_NAME, SEED, DEVICE, RESULTS_DIR, T2_RESULTS_PATH,
    set_seed, load_calibration_data, load_t2_criticality,
    run_layer_replacements, create_plots,
)


def main():
    set_seed(SEED)

    # Load T-2 criticality data
    t2_criticality = load_t2_criticality()
    if not t2_criticality:
        print("ERROR: T-2 results not found. Run T-2 experiment first.")
        return

    # Build single_results format
    single_results = {"knockouts": {
        int(k): {"loss_delta": v} for k, v in t2_criticality.items()
    }}

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} decoder layers on {DEVICE}")

    calibration_data = load_calibration_data()

    t_start = time.time()
    replacement_results = run_layer_replacements(
        model, tokenizer, num_layers, single_results, calibration_data)
    elapsed = time.time() - t_start
    print(f"\nMethod 7 completed in {elapsed:.1f}s")

    # Update summary.json with replacement results
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["layer_replacements"] = replacement_results

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
