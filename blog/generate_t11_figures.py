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
#    Single panel: dual-axis overlay showing the correlation

fig, ax = plt.subplots(figsize=(14, 5.5))

# Phase background bands
for start, end, label, color in phases:
    ax.axvspan(start - 0.5, end + 0.5, color=color, zorder=0)
    mid = (start + end) / 2
    ax.text(mid, 0.255, label, ha='center', va='top',
            fontsize=9, color='#8b949e', alpha=0.8)

# Left axis: linearity gap (line)
l1 = ax.plot(layers, t7_gap, 'o-', color=C_RED, ms=5, lw=1.8,
             label='Linearity Gap', zorder=3)
ax.set_ylabel("Linearity Gap (perturbation)", color=C_RED, fontsize=12)
ax.set_xlabel("Layer Index", fontsize=12)
ax.tick_params(axis='y', labelcolor=C_RED)
ax.set_ylim(0.12, 0.26)

# Right axis: 2-bit sensitivity (bars, log scale)
ax2 = ax.twinx()
sens_2b_clip = np.clip(sens_2b, 0.15, None)  # clip zeros for log
bars = ax2.bar(layers, sens_2b_clip, color=C_BLUE, alpha=0.35, width=0.7, zorder=1)
l2 = ax2.plot(layers, sens_2b_clip, 's', color=C_BLUE, ms=4, zorder=2,
              label='2-bit Quant Sensitivity')
ax2.set_yscale('log')
ax2.set_ylabel("PPL Δ at 2-bit (log scale)", color=C_BLUE, fontsize=12)
ax2.tick_params(axis='y', labelcolor=C_BLUE)

# Annotate the extreme early-layer values
for l_idx in [0, 1, 2, 3]:
    if sens_2b[l_idx] > 10:
        ax2.annotate(f'+{sens_2b[l_idx]:.0f}',
                     (l_idx, sens_2b[l_idx]),
                     textcoords='offset points', xytext=(0, 8),
                     ha='center', fontsize=8, color=C_BLUE, fontweight='bold')

# Annotate layer 35
ax2.annotate(f'+{sens_2b[35]:.1f}',
             (35, sens_2b[35]),
             textcoords='offset points', xytext=(0, 8),
             ha='center', fontsize=8, color=C_BLUE, fontweight='bold')

# Correlation stat box
r, p = stats.spearmanr(t7_gap, sens_2b)
ax.text(0.98, 0.95, f'Spearman ρ = {r:.2f}\np < 0.0001',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, color='#c9d1d9',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262d',
                  edgecolor='#30363d', alpha=0.9))

# Combined legend
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='upper left', fontsize=10,
          facecolor='#161b22', edgecolor='#30363d')

ax.set_title("Linearity Gap Predicts Quantization Sensitivity",
             fontsize=14, fontweight='bold', pad=12)
ax.grid(True, axis='y', alpha=0.2)
ax.set_xlim(-0.8, 35.8)

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
# Plot with clipped bars; use log scale to show the L2 spike naturally
gate_clipped = np.clip(gate_sens, 1e-4, None)
colors = [C_RED if g > 0.1 else C_ORANGE if g > 0.03 else C_GRAY for g in gate_sens]
ax.bar(layers, gate_clipped, color=colors, alpha=0.85, width=0.8)
ax.set_yscale('log')
ax.set_ylim(5e-4, 15)
ax.set_xlabel("Layer Index")
ax.set_ylabel("PPL Δ (3-bit, gate_proj only, log)")
ax.set_title("gate_proj: The SwiGLU Achilles' Heel", fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
# Annotate the L2 spike
ax.annotate(f'L2: +{gate_sens[2]:.1f}', (2, gate_sens[2]),
            textcoords='offset points', xytext=(14, 6),
            ha='left', fontsize=9, color=C_RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2))

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
