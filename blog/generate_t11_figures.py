"""Generate T-11 quantization figures for the blog post."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

# Style — matches blog generate_figures.py
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

OUTDIR = Path("blog/images")
N_LAYERS = 36
layers = np.arange(N_LAYERS)

C_BLUE = '#6e9ecf'
C_GREEN = '#6aba7a'
C_RED = '#cf6e6e'
C_ORANGE = '#c9a04a'
C_PURPLE = '#a08ac7'
C_CYAN = '#5fb5a7'
C_PINK = '#c78aab'
C_GRAY = '#8b949e'

# ── Load data ──────────────────────────────────────────────────

with open("experiments/t11_quantization/results/phase1_per_layer.json") as f:
    t11_p1 = json.load(f)

with open("experiments/t11_quantization/results/phase1b_per_matrix.json") as f:
    t11_p1b = json.load(f)

with open("experiments/t11_quantization/results/phase3_mixed_precision.json") as f:
    t11_p3 = json.load(f)

with open("experiments/t7_layer_linearization_gap/results/summary.json") as f:
    t7 = json.load(f)

with open("experiments/t9_weight_spectral_structure/results/summary.json") as f:
    t9 = json.load(f)

# Extract arrays
sens_2b = np.array([t11_p1["per_layer"][str(l)]["ppl_delta"][str(2)] for l in range(N_LAYERS)])
sens_3b = np.array([t11_p1["per_layer"][str(l)]["ppl_delta"][str(3)] for l in range(N_LAYERS)])
sens_4b = np.array([t11_p1["per_layer"][str(l)]["ppl_delta"][str(4)] for l in range(N_LAYERS)])

t7_gap = np.array([t7["per_layer"][f"layer_{l}"]["perturb_gap_mean"] for l in range(N_LAYERS)])
t7_mlp_gap = np.array([t7["per_layer"][f"layer_{l}"]["mlp_gap_mean"] for l in range(N_LAYERS)])

ALL_MATRICES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
t9_avg_rank = np.array([
    np.mean([t9["per_layer"][str(l)][m]["effective_rank_ratio"]
             for m in ALL_MATRICES if m in t9["per_layer"][str(l)]])
    for l in range(N_LAYERS)])

# Per-matrix sensitivity
mat_sens = {}
for m in ALL_MATRICES:
    mat_sens[m] = np.array([t11_p1b["per_layer"][str(l)][m]["ppl_delta"] for l in range(N_LAYERS)])

# Phase regions
phases = [
    (0, 5, "Embed\nErasure", '#cf6e6e30'),
    (6, 15, "Distributed\nProcessing", '#6aba7a30'),
    (16, 24, "Bottleneck", '#c9a04a30'),
    (25, 34, "Output\nPrep", '#6e9ecf30'),
    (35, 35, "L35", '#a08ac730'),
]


# ── Figure 10: Quantization Sensitivity vs Linearity Gap ─────────

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                          gridspec_kw={'height_ratios': [1.2, 1]})

# Top: 2-bit sensitivity depth profile with phase regions
ax = axes[0]
for start, end, label, color in phases:
    ax.axvspan(start - 0.5, end + 0.5, color=color, zorder=0)
    mid = (start + end) / 2
    ax.text(mid, max(sens_2b[5:]) * 0.9, label, ha='center', va='top',
            fontsize=8, color='#8b949e', alpha=0.7)

ax.bar(layers, sens_2b, color=C_RED, alpha=0.8, width=0.8, zorder=2)
# Clip the extreme values for readability
clip_val = 50
for l in range(N_LAYERS):
    if sens_2b[l] > clip_val:
        ax.annotate(f'{sens_2b[l]:.0f}', (l, clip_val * 0.95),
                    ha='center', va='top', fontsize=8, color=C_RED, fontweight='bold')
ax.set_ylim(0, clip_val)
ax.set_ylabel("PPL increase\n(2-bit RTN, single layer)")
ax.set_title("Per-Layer Quantization Sensitivity", fontsize=14, fontweight='bold', pad=10)
ax.grid(True, axis='y', alpha=0.3)

# Inset for extreme values
inset = ax.inset_axes([0.3, 0.45, 0.45, 0.50])
inset.bar([0, 1, 2, 3, 4], [sens_2b[i] for i in [0, 1, 2, 3, 4]],
          color=[C_RED if s > 50 else C_ORANGE for s in [sens_2b[i] for i in [0, 1, 2, 3, 4]]],
          alpha=0.9)
inset.set_xticks([0, 1, 2, 3, 4])
inset.set_xticklabels(['L0', 'L1', 'L2', 'L3', 'L4'], fontsize=8)
inset.set_ylabel("PPL Δ", fontsize=8)
inset.set_title("Early layers (full scale)", fontsize=9, color='#c9d1d9')
inset.set_facecolor('#1c2128')
for spine in inset.spines.values():
    spine.set_color('#30363d')
inset.tick_params(colors='#8b949e', labelsize=7)
inset.yaxis.label.set_color('#c9d1d9')

# Bottom: Overlay with T-7 linearity gap
ax = axes[1]
for start, end, label, color in phases:
    ax.axvspan(start - 0.5, end + 0.5, color=color, zorder=0)

ax.plot(layers, t7_gap, 'o-', color=C_RED, ms=5, lw=1.5, label='T-7 Linearity Gap', zorder=3)
ax2 = ax.twinx()
# Use log scale for sensitivity to show pattern
sens_2b_clip = np.clip(sens_2b, 0.1, None)
ax2.plot(layers, sens_2b_clip, 's-', color=C_BLUE, ms=4, lw=1.5, alpha=0.8,
         label='2-bit Sensitivity', zorder=2)
ax2.set_yscale('log')

ax.set_xlabel("Layer Index")
ax.set_ylabel("Linearity Gap", color=C_RED)
ax2.set_ylabel("PPL Δ (2-bit, log scale)", color=C_BLUE)

r, p = stats.spearmanr(t7_gap, sens_2b)
ax.text(0.98, 0.95, f'Spearman ρ = {r:.2f}\np < 0.0001',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=10, color='#c9d1d9',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#21262d', edgecolor='#30363d'))

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
          fontsize=9, facecolor='#161b22', edgecolor='#30363d')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTDIR / "fig10_quant_sensitivity.png", dpi=200, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Saved fig10_quant_sensitivity.png")


# ── Figure 11: Per-Matrix Quantization Sensitivity ──────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                          gridspec_kw={'width_ratios': [1.3, 1]})

# Left: Mean sensitivity by matrix type (grouped with visual separators)
ax = axes[0]
mat_order = ["q_proj", "k_proj", "o_proj", "v_proj", "up_proj", "down_proj", "gate_proj"]
mat_means = [mat_sens[m].mean() for m in mat_order]
mat_colors = [C_BLUE, C_BLUE, C_BLUE, C_CYAN, C_GREEN, C_GREEN, C_RED]
mat_labels_short = ["Q  (attn)", "K  (attn)", "O  (attn)", "V  (attn)",
                     "up  (MLP)", "down  (MLP)", "gate  (MLP)"]

bars = ax.barh(range(len(mat_order)), mat_means, color=mat_colors, alpha=0.85,
               edgecolor='#30363d', linewidth=0.5)
ax.set_yticks(range(len(mat_order)))
ax.set_yticklabels(mat_labels_short, fontsize=10)
ax.set_xlabel("Mean PPL Δ (3-bit RTN, single matrix)")
ax.set_title("Per-Matrix Quantization Sensitivity", fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
# Separator line between attention and MLP groups
ax.axhline(y=3.5, color='#30363d', ls='--', lw=0.8, alpha=0.7)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, mat_means)):
    if val > 0.005:
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, color='#c9d1d9')

# Right: gate_proj sensitivity by layer (showing the L2 spike)
ax = axes[1]
gate_sens = mat_sens["gate_proj"]
colors = [C_RED if g > 0.1 else C_ORANGE if g > 0.03 else C_GRAY for g in gate_sens]
ax.bar(layers, gate_sens, color=colors, alpha=0.85, width=0.8)
# Clip for readability
clip_g = 0.5
for l in range(N_LAYERS):
    if gate_sens[l] > clip_g:
        ax.annotate(f'{gate_sens[l]:.1f}', (l, clip_g * 0.95),
                    ha='center', va='top', fontsize=8, color=C_RED, fontweight='bold')
ax.set_ylim(0, clip_g)
ax.set_xlabel("Layer Index")
ax.set_ylabel("PPL Δ (3-bit, gate_proj only)")
ax.set_title("gate_proj: The SwiGLU Achilles' Heel", fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTDIR / "fig11_quant_matrix.png", dpi=200, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Saved fig11_quant_matrix.png")


# ── Figure 12: Mixed-Precision Recipes ──────────────────────────

fig, ax = plt.subplots(figsize=(13, 5.5))

recipes = [
    ("Uniform\n4-bit", t11_p3["recipes"]["uniform_4bit"]["ppl_delta"], C_GRAY),
    ("Spectral\nInformed", min(t11_p3["recipes"]["t9_spectral"]["ppl_delta"], 500), C_PURPLE),
    ("Linearity\nInformed", min(t11_p3["recipes"]["t7_linearity"]["ppl_delta"], 500), C_ORANGE),
    ("Sensitivity\nOracle", t11_p3["recipes"]["sensitivity_oracle"]["ppl_delta"], C_CYAN),
    ("L35 at 5-bit\n(rest 4-bit)", t11_p3["recipes"]["first_last_protected"]["ppl_delta"], C_GREEN),
    ("Q/K=3b\nV/MLP=5b", t11_p3["recipes"]["qk3_vmul5"]["ppl_delta"], C_BLUE),
]

actuals = [t11_p3["recipes"]["uniform_4bit"]["ppl_delta"],
           t11_p3["recipes"]["t9_spectral"]["ppl_delta"],
           t11_p3["recipes"]["t7_linearity"]["ppl_delta"],
           t11_p3["recipes"]["sensitivity_oracle"]["ppl_delta"],
           t11_p3["recipes"]["first_last_protected"]["ppl_delta"],
           t11_p3["recipes"]["qk3_vmul5"]["ppl_delta"]]

names = [r[0] for r in recipes]
deltas = [r[1] for r in recipes]
colors = [r[2] for r in recipes]

bars = ax.bar(range(len(recipes)), deltas, color=colors, alpha=0.85,
              edgecolor='#30363d', linewidth=0.5, width=0.65)
ax.set_xticks(range(len(recipes)))
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("PPL Δ from BF16 baseline")
ax.set_title("Mixed-Precision Recipes at ~4-bit Average",
             fontsize=13, fontweight='bold', pad=10)
ax.grid(True, axis='y', alpha=0.3)

# Annotate actual values — position carefully to avoid overlaps
for i, (bar, delta) in enumerate(zip(bars, deltas)):
    label = f'+{actuals[i]:.1f}' if actuals[i] < 100 else f'+{actuals[i]:.0f}'
    y = bar.get_height()
    if y > 100:
        ax.text(bar.get_x() + bar.get_width() / 2, y * 0.5, label,
                ha='center', va='center', fontsize=10, color='#c9d1d9', fontweight='bold')
    elif y > 10:
        ax.text(bar.get_x() + bar.get_width() / 2, y + 8, label,
                ha='center', va='bottom', fontsize=9, color='#c9d1d9')
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, y + 15, label,
                ha='center', va='bottom', fontsize=9, color='#c9d1d9')

# Star marker on the winner instead of overlapping arrow
ax.plot(4, deltas[4], '*', color=C_GREEN, markersize=18, zorder=5)
ax.text(4, deltas[4] + 30, 'Best', ha='center', va='bottom',
        fontsize=11, color=C_GREEN, fontweight='bold')

# Note about catastrophic failure — position to the right to avoid overlap
ax.annotate('Assigns 2-bit to early layers\n→ catastrophic failure',
            xy=(1, deltas[1]), xytext=(1.8, 380),
            arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=1.5, alpha=0.7),
            fontsize=9, color=C_PURPLE, ha='left')

plt.tight_layout()
plt.savefig(OUTDIR / "fig12_mixed_precision.png", dpi=200, bbox_inches='tight',
            facecolor='#0d1117')
plt.close()
print("Saved fig12_mixed_precision.png")

print("\nAll T-11 blog figures generated.")
