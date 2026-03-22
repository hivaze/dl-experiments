"""
Run enhanced Method 7 (Global Linear Replacement) using pre-existing T-2 results.

Enhancements over original:
  - Ridge regression (L2 regularization) to prevent catastrophic overfitting
  - Affine fitting (bias term) to capture mean shifts from RMSNorm
  - Train/test split (80/20) — ridge lambda selected by test-set activation MSE
    (no extra model forward passes)
  - More rank values (16, 32, 64, 128, 256, 512) for smoother rank-recovery curves
  - All 36 layers tested (not just a subset)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run import (
    MODEL_NAME, SEED, DEVICE, RESULTS_DIR, T2_RESULTS_PATH,
    set_seed, load_calibration_data, load_t2_criticality,
)

# Enhanced configuration
RANKS_TO_TEST = [16, 32, 64, 128, 256, 512]
RIDGE_LAMBDAS = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0]
TRAIN_FRACTION = 0.8


def compute_loss(model, tokenizer, calibration_data, device):
    """Compute mean cross-entropy loss on completion tokens only."""
    total_loss = 0.0
    total_tokens = 0
    for entry in calibration_data:
        full_text = entry["full_text"]
        prompt_token_count = entry["prompt_token_count"]
        tokens = tokenizer(full_text, return_tensors="pt").to(device)
        input_ids = tokens["input_ids"]
        seq_len = input_ids.shape[1]
        if seq_len < 2 or prompt_token_count >= seq_len:
            continue
        with torch.no_grad():
            outputs = model(**tokens, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        start = max(prompt_token_count - 1, 0)
        comp_logits = logits[:, start:, :]
        comp_targets = targets[:, start:]
        if comp_targets.numel() == 0:
            continue
        loss = F.cross_entropy(comp_logits.reshape(-1, comp_logits.size(-1)),
                               comp_targets.reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += comp_targets.numel()
    return total_loss / total_tokens if total_tokens > 0 else float("inf")


def collect_layer_activations(model, tokenizer, layer_idx, calibration_data, device):
    """Collect (input, output) activation pairs for a given layer."""
    inputs_list = []
    outputs_list = []

    def capture_hook(module, inp, out):
        h_in = inp[0].detach()
        h_out = out[0].detach() if isinstance(out, tuple) else out.detach()
        inputs_list.append(h_in.squeeze(0))
        outputs_list.append(h_out.squeeze(0))

    hook = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        for entry in calibration_data:
            tokens = tokenizer(entry["full_text"], return_tensors="pt").to(device)
            model(**tokens, use_cache=False)
    hook.remove()

    return torch.cat(inputs_list, dim=0), torch.cat(outputs_list, dim=0)


def split_train_test(inputs, outputs, train_frac=0.8, seed=42):
    """Split activation pairs into train/test by token index. Stays on same device."""
    n = inputs.shape[0]
    gen = torch.Generator(device='cpu').manual_seed(seed)
    perm = torch.randperm(n, generator=gen).to(inputs.device)
    n_train = int(n * train_frac)
    return (inputs[perm[:n_train]], outputs[perm[:n_train]],
            inputs[perm[n_train:]], outputs[perm[n_train:]])


def fit_ridge(X, Y, ridge_lambda=0.0):
    """Fit W = argmin ||Y - WX||² + λ||W||² via normal equations. float32."""
    d = X.shape[1]
    XtX = X.T @ X
    if ridge_lambda > 0:
        XtX = XtX + ridge_lambda * torch.eye(d, device=X.device)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY).T  # (d, d)
    return W


def fit_ridge_affine(X, Y, ridge_lambda=0.0):
    """Fit Y ≈ WX + b via ridge regression with bias."""
    x_mean = X.mean(0, keepdim=True)
    y_mean = Y.mean(0, keepdim=True)
    W = fit_ridge(X - x_mean, Y - y_mean, ridge_lambda)
    b = (y_mean - (W @ x_mean.T).T).squeeze(0)
    return W, b


def fit_lowrank_residual(X, Y, rank, ridge_lambda=0.1):
    """Fit Y ≈ X + W_r X where W_r is rank-truncated. Preserves skip connection."""
    R = Y - X
    W_r = fit_ridge(X, R, ridge_lambda)
    U, S, Vh = torch.linalg.svd(W_r, full_matrices=False)
    W_r_lr = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
    W = torch.eye(X.shape[1], device=X.device) + W_r_lr
    return W


def activation_mse(W, X, Y, bias=None):
    """Compute MSE on activations: ||Y - WX - b||² / N."""
    pred = X @ W.T
    if bias is not None:
        pred = pred + bias.unsqueeze(0)
    return ((Y - pred) ** 2).mean().item()


def evaluate_with_replacement(model, tokenizer, layer_idx, W, bias,
                               calibration_data, device):
    """Replace a layer with linear/affine map and measure loss."""
    def hook_fn(module, inp, out):
        h_in = inp[0]
        h_out = torch.einsum("ij,bsj->bsi", W, h_in)
        if bias is not None:
            h_out = h_out + bias
        return (h_out,) + out[1:] if isinstance(out, tuple) else h_out

    hook = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    loss = compute_loss(model, tokenizer, calibration_data, device)
    hook.remove()
    return loss


def run_enhanced_replacements(model, tokenizer, num_layers, t2_criticality,
                               calibration_data):
    """Enhanced Method 7 with ridge, affine, train/test split, all layers."""
    print("\n" + "=" * 70)
    print("ENHANCED METHOD 7: Global Linear Replacement")
    print("=" * 70)

    device = next(model.parameters()).device

    # Baseline loss
    print("  Computing baseline loss...")
    baseline_loss = compute_loss(model, tokenizer, calibration_data, device)
    print(f"  Baseline loss: {baseline_loss:.4f}")

    # Knockout losses for all layers
    print(f"  Computing knockout losses for all {num_layers} layers...")
    original_layers = list(model.model.layers)
    knockout_losses = {}
    for skip_idx in tqdm(range(num_layers), desc="  Knockout"):
        remaining = [l for i, l in enumerate(original_layers) if i != skip_idx]
        model.model.layers = torch.nn.ModuleList(remaining)
        knockout_losses[skip_idx] = compute_loss(model, tokenizer, calibration_data, device)
    model.model.layers = torch.nn.ModuleList(original_layers)

    results = {
        "baseline_loss": baseline_loss,
        "config": {
            "ranks": RANKS_TO_TEST,
            "ridge_lambdas": RIDGE_LAMBDAS,
            "train_fraction": TRAIN_FRACTION,
        },
        "replacements": {},
    }

    for layer_idx in tqdm(range(num_layers), desc="Layer replacements"):
        t0 = time.time()

        # Collect activations (one forward pass through all data)
        inputs, outputs = collect_layer_activations(
            model, tokenizer, layer_idx, calibration_data, device)
        n_tokens = inputs.shape[0]

        # Train/test split on GPU in float32
        X_tr, Y_tr, X_te, Y_te = split_train_test(
            inputs.float(), outputs.float(), TRAIN_FRACTION, SEED)

        knockout_delta = knockout_losses[layer_idx] - baseline_loss

        layer_results = {
            "knockout_loss": knockout_losses[layer_idx],
            "knockout_delta": knockout_delta,
            "n_tokens": n_tokens,
        }
        replacements = {}

        # ============================================================
        # 1. OLS full-rank (original method, no regularization, no bias)
        # ============================================================
        W_ols = fit_ridge(X_tr, Y_tr, 0.0)
        ols_test_mse = activation_mse(W_ols, X_te, Y_te)

        W_dev = W_ols.to(torch.bfloat16).to(device)
        ols_loss = evaluate_with_replacement(
            model, tokenizer, layer_idx, W_dev, None, calibration_data, device)
        ols_delta = ols_loss - baseline_loss
        ols_rec = (1.0 - ols_delta / knockout_delta) if knockout_delta > 0 else 0.0

        replacements["full_rank"] = {
            "loss": ols_loss, "loss_delta": ols_delta,
            "recovery_vs_knockout": ols_rec,
            "test_mse": ols_test_mse,
        }
        del W_dev

        # ============================================================
        # 2. Ridge full-rank — select best λ by test-set activation MSE
        #    (cheap: no model forward pass needed for selection)
        # ============================================================
        best_lam, best_mse = 0.0, float("inf")
        for lam in RIDGE_LAMBDAS:
            W = fit_ridge(X_tr, Y_tr, lam)
            mse = activation_mse(W, X_te, Y_te)
            if mse < best_mse:
                best_lam, best_mse = lam, mse
                W_best = W

        W_dev = W_best.to(torch.bfloat16).to(device)
        ridge_loss = evaluate_with_replacement(
            model, tokenizer, layer_idx, W_dev, None, calibration_data, device)
        ridge_delta = ridge_loss - baseline_loss
        ridge_rec = (1.0 - ridge_delta / knockout_delta) if knockout_delta > 0 else 0.0

        replacements["ridge_best"] = {
            "loss": ridge_loss, "loss_delta": ridge_delta,
            "recovery_vs_knockout": ridge_rec,
            "best_lambda": best_lam, "test_mse": best_mse,
        }
        del W_dev

        # ============================================================
        # 3. Ridge + affine — select best λ by test-set MSE
        # ============================================================
        best_lam_a, best_mse_a = 0.0, float("inf")
        for lam in RIDGE_LAMBDAS:
            W, b = fit_ridge_affine(X_tr, Y_tr, lam)
            mse = activation_mse(W, X_te, Y_te, b)
            if mse < best_mse_a:
                best_lam_a, best_mse_a = lam, mse
                W_best_a, b_best_a = W, b

        W_dev = W_best_a.to(torch.bfloat16).to(device)
        b_dev = b_best_a.to(torch.bfloat16).to(device)
        affine_loss = evaluate_with_replacement(
            model, tokenizer, layer_idx, W_dev, b_dev, calibration_data, device)
        affine_delta = affine_loss - baseline_loss
        affine_rec = (1.0 - affine_delta / knockout_delta) if knockout_delta > 0 else 0.0

        replacements["affine_best"] = {
            "loss": affine_loss, "loss_delta": affine_delta,
            "recovery_vs_knockout": affine_rec,
            "best_lambda": best_lam_a, "test_mse": best_mse_a,
        }
        del W_dev, b_dev

        # ============================================================
        # 4. Low-rank variants (residual-preserving, mild ridge)
        # ============================================================
        for rank in RANKS_TO_TEST:
            W_lr = fit_lowrank_residual(X_tr, Y_tr, rank, ridge_lambda=0.1)
            lr_test_mse = activation_mse(W_lr, X_te, Y_te)

            W_dev = W_lr.to(torch.bfloat16).to(device)
            lr_loss = evaluate_with_replacement(
                model, tokenizer, layer_idx, W_dev, None, calibration_data, device)
            lr_delta = lr_loss - baseline_loss
            lr_rec = (1.0 - lr_delta / knockout_delta) if knockout_delta > 0 else 0.0

            replacements[f"rank_{rank}"] = {
                "loss": lr_loss, "loss_delta": lr_delta,
                "recovery_vs_knockout": lr_rec,
                "test_mse": lr_test_mse, "rank": rank,
            }
            del W_dev

        layer_results["replacements"] = replacements
        results["replacements"][layer_idx] = layer_results

        elapsed = time.time() - t0
        # Print compact per-layer summary
        print(f"  L{layer_idx:>2} ({elapsed:>4.1f}s) "
              f"KO={knockout_delta:+.3f}  "
              f"OLS={ols_rec:>6.1%}  Ridge={ridge_rec:>6.1%}  "
              f"Affine={affine_rec:>6.1%}  "
              f"R256={replacements['rank_256']['recovery_vs_knockout']:>6.1%}  "
              f"R64={replacements['rank_64']['recovery_vs_knockout']:>6.1%}")
        sys.stdout.flush()

        del inputs, outputs, X_tr, Y_tr, X_te, Y_te
        torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*100}")
    print(f"{'Layer':>5} {'KO Δ':>8} {'OLS':>8} {'Ridge':>8} {'Affine':>8} "
          f"{'R512':>8} {'R256':>8} {'R128':>8} {'R64':>8} {'R32':>8} {'R16':>8}")
    print(f"{'-'*100}")
    for li in range(num_layers):
        lr = results["replacements"][li]
        ko = lr["knockout_delta"]
        vals = []
        for key in ["full_rank", "ridge_best", "affine_best",
                     "rank_512", "rank_256", "rank_128", "rank_64", "rank_32", "rank_16"]:
            r = lr["replacements"].get(key, {}).get("recovery_vs_knockout", 0)
            vals.append(f"{r:>7.1%}")
        print(f"  L{li:>3} {ko:>8.3f} {' '.join(vals)}")
    print(f"{'='*100}")

    return results


def create_enhanced_plots(results, num_layers, model_short="Qwen3-4B-Instruct"):
    """Generate enhanced Method 7 plots."""
    rep = results["replacements"]
    layers = sorted(int(k) for k in rep.keys())

    # ---- Plot 1: Recovery comparison across methods (all layers) ----
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    x = np.arange(len(layers))
    width = 0.13

    ax = axes[0]
    methods = [
        ("full_rank", "OLS", "#1976D2"),
        ("ridge_best", "Ridge", "#7B1FA2"),
        ("affine_best", "Affine", "#C62828"),
        ("rank_512", "R-512", "#00897B"),
        ("rank_256", "R-256", "#FFA726"),
        ("rank_64", "R-64", "#66BB6A"),
    ]
    for i, (key, label, color) in enumerate(methods):
        recs = [rep[l]["replacements"].get(key, {}).get("recovery_vs_knockout", 0) * 100
                for l in layers]
        # Clip for readability
        recs_clip = [max(r, -200) for r in recs]
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, recs_clip, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Recovery vs Knockout (%)")
    ax.set_title("Layer Replacement Recovery — All Methods (All 36 Layers)")
    ax.axhline(y=100, color="black", linestyle="--", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=7, ncol=6, loc="lower left")
    ax.set_ylim(bottom=-210)

    # Bottom: loss deltas for knockout vs best methods
    ax = axes[1]
    knockout_deltas = [rep[l]["knockout_delta"] for l in layers]
    ridge_deltas = [rep[l]["replacements"].get("ridge_best", {}).get("loss_delta", 0) for l in layers]
    affine_deltas = [rep[l]["replacements"].get("affine_best", {}).get("loss_delta", 0) for l in layers]
    r256_deltas = [rep[l]["replacements"].get("rank_256", {}).get("loss_delta", 0) for l in layers]

    w2 = 0.2
    ax.bar(x - 1.5*w2, knockout_deltas, w2, label="Knockout", color="#E53935", alpha=0.85)
    ax.bar(x - 0.5*w2, ridge_deltas, w2, label="Ridge", color="#7B1FA2", alpha=0.85)
    ax.bar(x + 0.5*w2, affine_deltas, w2, label="Affine", color="#C62828", alpha=0.85)
    ax.bar(x + 1.5*w2, r256_deltas, w2, label="R-256", color="#FFA726", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers], fontsize=7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Loss Delta vs Baseline")
    ax.set_title("Loss Impact: Knockout vs Best Replacement Methods")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)

    fig.suptitle(f"Enhanced Global Linear Replacement (Method 7) — {model_short}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "layer_replacement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved layer_replacement.png")

    # ---- Plot 2: Rank-recovery curves + Ridge vs OLS scatter ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    interesting = [0, 1, 3, 6, 8, 12, 17, 23, 32, 35]
    interesting = [l for l in interesting if l in rep]

    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 1, len(interesting)))
    for i, l in enumerate(interesting):
        recs = []
        for rank in RANKS_TO_TEST:
            rec = rep[l]["replacements"].get(f"rank_{rank}", {}).get(
                "recovery_vs_knockout", 0) * 100
            recs.append(rec)
        ax.plot(RANKS_TO_TEST, recs, 'o-', color=cmap[i], label=f"L{l}", markersize=4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Recovery (%)")
    ax.set_title("Rank vs Recovery (Selected Layers)")
    ax.legend(fontsize=7, ncol=2)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)

    # OLS vs Ridge scatter
    ax = axes[1]
    ols_recs = [rep[l]["replacements"].get("full_rank", {}).get(
        "recovery_vs_knockout", 0) * 100 for l in layers]
    ridge_recs = [rep[l]["replacements"].get("ridge_best", {}).get(
        "recovery_vs_knockout", 0) * 100 for l in layers]

    ax.scatter(ols_recs, ridge_recs, c=layers, cmap="viridis", s=40, zorder=3)
    for i, l in enumerate(layers):
        if abs(ols_recs[i] - ridge_recs[i]) > 20 or ols_recs[i] < -100:
            ax.annotate(f"L{l}", (ols_recs[i], ridge_recs[i]), fontsize=6,
                       textcoords="offset points", xytext=(5, 5))
    lims = [min(min(ols_recs), min(ridge_recs), -50), max(max(ridge_recs), 105)]
    ax.plot(lims, lims, 'k--', alpha=0.3, label="y=x")
    ax.set_xlabel("OLS Recovery (%)")
    ax.set_ylabel("Ridge Recovery (%)")
    ax.set_title("Ridge vs OLS: Recovery Improvement")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "layer_replacement_enhanced.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved layer_replacement_enhanced.png")


def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t2_criticality = load_t2_criticality()
    if not t2_criticality:
        print("ERROR: T-2 results not found. Run T-2 experiment first.")
        return

    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    num_layers = len(model.model.layers)
    print(f"Model loaded: {num_layers} decoder layers on {DEVICE}")

    calibration_data = load_calibration_data()

    t_start = time.time()
    results = run_enhanced_replacements(
        model, tokenizer, num_layers, t2_criticality, calibration_data)
    elapsed = time.time() - t_start
    print(f"\nEnhanced Method 7 completed in {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()

    create_enhanced_plots(results, num_layers)

    # Update summary.json
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results["layer_replacements"] = results

    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
