"""
Run only Part 6 (Jacobian Consistency) using pre-existing results.
Loads summary.json, runs Jacobian consistency analysis, generates
the consistency plot, and updates summary.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import (
    MODEL_NAME, SEED, DEVICE, RESULTS_DIR,
    set_seed, load_calibration_data, load_t2_criticality,
    compute_jacobian_consistency, create_plots,
)


def main():
    set_seed(SEED)

    # Load existing results
    results_path = RESULTS_DIR / "summary.json"
    with open(results_path) as f:
        all_results = json.load(f)

    # Reconstruct summary format
    summary = {}
    for key, val in all_results["per_layer"].items():
        layer_idx = int(key.replace("layer_", ""))
        # Convert string keys back to float in multiscale_gaps
        if "multiscale_gaps" in val:
            val["multiscale_gaps"] = {float(k): v for k, v in val["multiscale_gaps"].items()}
        summary[layer_idx] = val
    num_layers = all_results["config"]["num_layers"]

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"Model loaded: {num_layers} layers on {DEVICE}")

    calibration_data = load_calibration_data()
    t2_criticality = load_t2_criticality()

    print("\n--- Computing Jacobian Consistency ---")
    t_start = time.time()
    jacobian_consistency = compute_jacobian_consistency(
        model, tokenizer, calibration_data, num_layers)
    jc_time = time.time() - t_start
    print(f"Jacobian consistency took {jc_time:.1f}s")

    # Print summary
    print(f"\n{'Layer':>5} | {'Consistency':>12} | {'Std':>8}")
    print("-" * 35)
    for l in sorted(jacobian_consistency.keys()):
        jc = jacobian_consistency[l]
        print(f"{l:>5} | {jc['consistency_mean']:>12.4f} | {jc.get('consistency_std', 0):>8.4f}")

    # Update JSON
    for l, jc in jacobian_consistency.items():
        key = f"layer_{l}"
        if key in all_results["per_layer"]:
            all_results["per_layer"][key]["jacobian_consistency_mean"] = jc["consistency_mean"]
            all_results["per_layer"][key]["jacobian_consistency_std"] = jc.get("consistency_std", 0)
    all_results["config"]["jacobian_consistency_prompts"] = 10
    all_results["config"]["jacobian_consistency_dirs"] = 8
    all_results["timing"]["jacobian_consistency_s"] = jc_time

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nUpdated {results_path}")

    # Free GPU
    del model
    torch.cuda.empty_cache()

    # Regenerate plots
    print("\n--- Regenerating plots ---")
    create_plots(summary, num_layers, t2_criticality, jacobian_consistency)
    print("Done!")


if __name__ == "__main__":
    main()
