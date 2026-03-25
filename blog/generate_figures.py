"""Generate synthesis figures for the blog post."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Style
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
OUTDIR.mkdir(parents=True, exist_ok=True)

# Colors — muted, professional palette
C_BLUE = '#6e9ecf'
C_GREEN = '#6aba7a'
C_RED = '#cf6e6e'
C_ORANGE = '#c9a04a'
C_PURPLE = '#a08ac7'
C_CYAN = '#5fb5a7'
C_PINK = '#c78aab'
C_GRAY = '#8b949e'

# ── Load data ──────────────────────────────────────────────────

with open("experiments/t4_residual_stream_geometry/results/summary.json") as f:
    t4 = json.load(f)

with open("experiments/t7_layer_linearization_gap/results/summary_v2.json") as f:
    t7 = json.load(f)

with open("experiments/t9_weight_spectral_structure/results/summary.json") as f:
    t9 = json.load(f)

N_LAYERS = 36
layers = np.arange(N_LAYERS)

# Extract T-4 data
t4_pr = []
t4_norm = []
t4_cosine_raw = []
t4_update_alignment = []
t4_frac_final = []
for i in range(N_LAYERS):
    key = str(i) if str(i) in t4['per_layer'] else f"layer_{i}"
    if key not in t4['per_layer']:
        key = str(i)
    t4_pr.append(t4['per_layer'][key]['participation_ratio'])
    t4_norm.append(t4['per_layer'][key]['mean_norm'])
    t4_cosine_raw.append(t4['per_layer'][key]['mean_cosine_raw'])

    imp_key = f"layer_{i}"
    t4_update_alignment.append(t4['layer_impact'][imp_key]['update_orthogonality'])

    rd_key = f"layer_{i}"
    t4_frac_final.append(t4['residual_decomposition'][rd_key]['fraction_of_final_norm'])

t4_pr = np.array(t4_pr)
t4_norm = np.array(t4_norm)
t4_cosine_raw = np.array(t4_cosine_raw)
t4_update_alignment = np.array(t4_update_alignment)
t4_frac_final = np.array(t4_frac_final)

# Extract T-7 data
t7_gap = []
t7_attn_gap = []
t7_mlp_gap = []
for i in range(N_LAYERS):
    key = f"layer_{i}"
    t7_gap.append(t7['per_layer'][key]['perturb_gap_mean'])
    t7_attn_gap.append(t7['per_layer'][key]['attn_gap_mean'])
    t7_mlp_gap.append(t7['per_layer'][key]['mlp_gap_mean'])
t7_gap = np.array(t7_gap)
t7_attn_gap = np.array(t7_attn_gap)
t7_mlp_gap = np.array(t7_mlp_gap)

# Extract T-7 CE recovery + R²
t7_recovery = []
t7_r2 = []
t7_knockout_delta = []
replacements = t7['layer_replacements']['replacements']
for i in range(N_LAYERS):
    r = replacements[str(i)]
    t7_knockout_delta.append(r['knockout_delta'])
    # Best recovery across methods
    best_rec = -999
    best_r2 = 0
    for method in ['residual_ridge', 'full_output_ridge', 'norm_residual_ridge']:
        if method in r['replacements']:
            rec = r['replacements'][method]['recovery_vs_knockout']
            if rec > best_rec:
                best_rec = rec
            if method == 'residual_ridge':
                best_r2 = r['replacements'][method]['residual_r2']
    t7_recovery.append(best_rec)
    t7_r2.append(best_r2)
t7_recovery = np.array(t7_recovery)
t7_r2 = np.array(t7_r2)
t7_knockout_delta = np.array(t7_knockout_delta)

# Extract T-9 data - mean effective rank ratio per layer
t9_mean_rank = []
t9_qk_rank = []
t9_vo_rank = []
t9_mlp_rank = []
for i in range(N_LAYERS):
    layer_data = t9['per_layer'][str(i)]
    ratios = [layer_data[m]['effective_rank_ratio'] for m in
              ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']]
    t9_mean_rank.append(np.mean(ratios))
    t9_qk_rank.append(np.mean([layer_data['q_proj']['effective_rank_ratio'],
                                layer_data['k_proj']['effective_rank_ratio']]))
    t9_vo_rank.append(np.mean([layer_data['v_proj']['effective_rank_ratio'],
                                layer_data['o_proj']['effective_rank_ratio']]))
    t9_mlp_rank.append(np.mean([layer_data['gate_proj']['effective_rank_ratio'],
                                 layer_data['up_proj']['effective_rank_ratio'],
                                 layer_data['down_proj']['effective_rank_ratio']]))
t9_mean_rank = np.array(t9_mean_rank)
t9_qk_rank = np.array(t9_qk_rank)
t9_vo_rank = np.array(t9_vo_rank)
t9_mlp_rank = np.array(t9_mlp_rank)


# ── Figure 1: Hero — The Pipeline (3 experiments unified) ──────

def make_hero():
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Three Views of the Same Transformer", fontsize=16, fontweight='bold', y=0.95)

    # Phase background shading — very subtle
    phases = [
        (0, 0.5, C_BLUE, 'Expansion'),
        (0.5, 5.5, C_RED, 'First\nCompression'),
        (5.5, 15.5, C_GREEN, 'Distributed\nProcessing'),
        (15.5, 24.5, C_RED, 'Second\nCompression'),
        (24.5, 34.5, C_ORANGE, 'Output\nPreparation'),
        (34.5, 35.5, C_PURPLE, 'Dispersal'),
    ]

    for ax in axes:
        for x0, x1, color, label in phases:
            ax.axvspan(x0, x1, alpha=0.05, color=color, zorder=0)
        ax.set_xlim(-0.5, 35.5)
        ax.grid(True, alpha=0.2)

    # Panel 1: Representation Geometry (T-4)
    ax = axes[0]
    ax.set_ylabel("Participation\nRatio", fontweight='bold')
    ax.plot(layers, t4_pr, color=C_BLUE, linewidth=2, zorder=5)
    ax.fill_between(layers, 0, t4_pr, alpha=0.08, color=C_BLUE)
    ax.set_ylim(0, 220)
    ax.annotate(f'PR = {t4_pr[16]:.1f}', xy=(16, t4_pr[16]), xytext=(19, 55),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
                fontsize=9.5, color=C_RED)
    ax.set_title("Exp. 1: Residual Stream Dimensionality", fontsize=11, loc='left', color=C_GRAY)

    # Panel 2: Linearization Gap
    ax = axes[1]
    ax.set_ylabel("Perturbation\nGap", fontweight='bold')
    ax.plot(layers, t7_gap, color=C_ORANGE, linewidth=2, zorder=5)
    ax.fill_between(layers, 0, t7_gap, alpha=0.08, color=C_ORANGE)
    ax.set_ylim(0.1, 0.28)
    min_idx = np.argmin(t7_gap[5:20]) + 5
    ax.annotate(f'Min = {t7_gap[min_idx]:.3f} (87% linear)', xy=(min_idx, t7_gap[min_idx]),
                xytext=(min_idx + 6, 0.24),
                arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.2),
                fontsize=9.5, color=C_ORANGE)
    ax.set_title("Exp. 2: Layer Nonlinearity", fontsize=11, loc='left', color=C_GRAY)

    # Panel 3: Weight Spectral Structure
    ax = axes[2]
    ax.set_ylabel("Effective Rank\nRatio", fontweight='bold')
    ax.plot(layers, t9_qk_rank, color=C_RED, linewidth=1.8, label='Q/K (routing)', zorder=5, alpha=0.85)
    ax.plot(layers, t9_vo_rank, color=C_CYAN, linewidth=1.8, label='V/O (extraction)', zorder=5, alpha=0.85)
    ax.plot(layers, t9_mlp_rank, color=C_GREEN, linewidth=1.8, label='MLP (processing)', zorder=5, alpha=0.85)
    ax.set_ylim(0, 0.85)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.3)
    ax.set_title("Exp. 3: Weight Matrix Capacity", fontsize=11, loc='left', color=C_GRAY)

    # Phase labels at top — subdued
    for x0, x1, color, label in phases:
        mid = (x0 + x1) / 2
        axes[0].text(mid, 215, label, ha='center', va='top', fontsize=7.5,
                     color=color, alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTDIR / "fig1_hero_pipeline.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig1_hero_pipeline.png")

make_hero()


# ── Figure 2: The Paradox (R² vs CE Recovery) ──────────────────

def make_paradox():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title("The Linearization Paradox: Fit Quality ≠ Downstream Utility",
                 fontsize=14, pad=15)

    # Color by depth — muted colormap
    scatter = ax.scatter(t7_r2, t7_recovery * 100, c=layers, cmap='viridis',
                         s=80, zorder=5, edgecolors='#30363d', linewidths=0.5, alpha=0.85)

    # Highlight only the 3 key layers — well-separated annotations
    highlights = {
        6: ("L6: R²=.997\nRecovery=54%", (-130, 15), C_RED),
        16: ("L16: R²=.768\nRecovery=44%", (-55, 30), C_RED),
        0: ("L0: R²=.787\nRecovery=98%", (15, -20), C_GREEN),
    }

    for idx, (text, offset, color) in highlights.items():
        ax.annotate(text, xy=(t7_r2[idx], t7_recovery[idx] * 100),
                    xytext=offset, textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                    fontsize=9, color=color,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22',
                              edgecolor=color, alpha=0.85))

    # Diagonal "expectation" line — subtle
    ax.plot([0.4, 1.0], [40, 100], '--', color=C_GRAY, alpha=0.2, linewidth=1, zorder=1)

    cbar = plt.colorbar(scatter, ax=ax, label='Layer Index', shrink=0.75, pad=0.02)

    ax.set_xlabel("Residual R² (activation-space fit quality)", fontweight='bold')
    ax.set_ylabel("CE Recovery % (actual downstream utility)", fontweight='bold')
    ax.set_xlim(0.38, 1.04)
    ax.set_ylim(30, 105)
    ax.grid(True, alpha=0.2)

    fig.savefig(OUTDIR / "fig2_paradox.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig2_paradox.png")

make_paradox()


# ── Figure 3: The Bottleneck ──────────────────────────────────

def make_bottleneck():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.suptitle("The Information Bottleneck: 2560 Dimensions → 2.3 Effective Dimensions",
                 fontsize=14)

    # Left: PR with bottleneck highlighted
    ax = axes[0]
    ax.fill_between(layers, 0, t4_pr, alpha=0.06, color=C_BLUE)
    ax.plot(layers, t4_pr, color=C_BLUE, linewidth=2)

    # Highlight bottlenecks — subtle
    ax.fill_between(range(1, 6), 0, t4_pr[1:6], alpha=0.18, color=C_RED, label='Bottleneck 1 (L1-5)')
    ax.fill_between(range(16, 25), 0, t4_pr[16:25], alpha=0.18, color=C_RED, label='Bottleneck 2 (L16-24)')

    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Participation Ratio\n(effective dimensions)", fontweight='bold')
    ax.set_ylim(0, 250)
    ax.legend(fontsize=9, loc='upper left', framealpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.set_title("Effective Dimensionality Across Depth", fontsize=12, color=C_GRAY)

    # Annotate the extremes
    ax.annotate(f'PR = {t4_pr[16]:.1f}\n(0.09% of space)', xy=(16, t4_pr[16]),
                xytext=(20, 120), arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5),
                fontsize=10, color=C_RED)
    ax.annotate(f'PR = {t4_pr[1]:.1f}', xy=(1, t4_pr[1]),
                xytext=(3, 80), arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
                fontsize=9.5, color=C_RED)

    # Right: Per-layer contribution to final output
    ax = axes[1]

    # Color bars by sign of update alignment
    bar_colors = [C_GREEN if a > 0 else C_RED for a in t4_update_alignment]
    bar_colors[35] = C_PURPLE
    ax.bar(layers, t4_frac_final * 100, color=bar_colors, alpha=0.7, width=0.8)
    ax.set_ylabel("Contribution to Final Output (%)", fontweight='bold')
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylim(0, 36)
    ax.grid(True, alpha=0.2)

    # Annotate layer 35
    ax.annotate(f'L35: {t4_frac_final[35]*100:.1f}%\n(cos = {t4_update_alignment[35]:.2f})',
                xy=(35, t4_frac_final[35] * 100),
                xytext=(22, 32), arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=1.5),
                fontsize=9.5, color=C_PURPLE)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_GREEN, alpha=0.7, label='Reinforcing (cos > 0)'),
        Patch(facecolor=C_RED, alpha=0.7, label='Opposing (cos < 0)'),
        Patch(facecolor=C_PURPLE, alpha=0.7, label='Layer 35 (cos = −0.73)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left', framealpha=0.3)

    ax.text(7, 28, 'L0-15: <2%\ncombined', fontsize=8.5, color=C_GRAY, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor=C_GRAY, alpha=0.4))
    ax.text(30, 22, 'L25-35: >90%\ncombined', fontsize=8.5, color=C_ORANGE, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor=C_ORANGE, alpha=0.4))

    ax.set_title("Where Does the Final Representation Come From?", fontsize=12, color=C_GRAY)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig3_bottleneck.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig3_bottleneck.png")

make_bottleneck()


# ── Figure 4: Routing vs Content ──────────────────────────────

def make_routing_vs_content():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Routing Is Cheap, Thinking Is Expensive",
                 fontsize=14)

    ax = axes[0]
    ax.plot(layers, t9_qk_rank, color=C_RED, linewidth=1.8, label='Q/K (routing)', marker='o', markersize=2.5, alpha=0.85)
    ax.plot(layers, t9_vo_rank, color=C_CYAN, linewidth=1.8, label='V/O (extraction)', marker='s', markersize=2.5, alpha=0.85)
    ax.plot(layers, t9_mlp_rank, color=C_GREEN, linewidth=1.8, label='MLP (processing)', marker='^', markersize=2.5, alpha=0.85)

    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Effective Rank Ratio", fontweight='bold')
    ax.legend(fontsize=10, loc='lower right', framealpha=0.3)
    ax.grid(True, alpha=0.2)
    ax.set_title("Weight Capacity by Function", fontsize=12, color=C_GRAY)
    ax.set_ylim(0, 0.85)

    # Right: Summary bar chart
    ax = axes[1]
    matrices = ['q_proj', 'k_proj', 'o_proj', 'gate_proj', 'v_proj', 'down_proj', 'up_proj']
    labels = ['Q', 'K', 'O', 'Gate', 'V', 'Down', 'Up']
    mean_ranks = []
    for m in matrices:
        vals = [t9['per_layer'][str(i)][m]['effective_rank_ratio'] for i in range(N_LAYERS)]
        mean_ranks.append(np.mean(vals))

    bar_colors = [C_RED, C_RED, C_CYAN, C_GREEN, C_CYAN, C_GREEN, C_GREEN]
    bars = ax.barh(range(len(matrices)), mean_ranks, color=bar_colors, alpha=0.7, height=0.6)

    ax.set_yticks(range(len(matrices)))
    ax.set_yticklabels(labels, fontweight='bold')
    ax.set_xlabel("Mean Effective Rank Ratio", fontweight='bold')
    ax.set_title("Average Capacity Usage by Matrix Type", fontsize=12, color=C_GRAY)
    ax.set_xlim(0, 0.8)
    ax.grid(True, alpha=0.2, axis='x')

    for i, (v, label) in enumerate(zip(mean_ranks, labels)):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=10, color='#c9d1d9')

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig4_routing_vs_content.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig4_routing_vs_content.png")

make_routing_vs_content()


# ── Figure 5: The Counterintuitive Correlation ────────────────

def make_rank_linearity():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("Higher Weight Rank → More Linear Behavior",
                 fontsize=14, pad=15)

    scatter = ax.scatter(t9_mean_rank, t7_gap, c=layers, cmap='viridis',
                         s=90, zorder=5, edgecolors='#30363d', linewidths=0.5, alpha=0.85)

    # Fit line — subtle
    coeffs = np.polyfit(t9_mean_rank, t7_gap, 1)
    x_fit = np.linspace(t9_mean_rank.min(), t9_mean_rank.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, '--', color=C_RED, linewidth=1.5, alpha=0.5)

    # Correlation stat — placed in lower-left clear area (away from data)
    r = np.corrcoef(t9_mean_rank, t7_gap)[0, 1]
    ax.text(0.03, 0.05, f'r = {r:.2f}, p = 0.009',
            fontsize=11, color=C_ORANGE,
            transform=ax.transAxes, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#161b22', edgecolor=C_ORANGE, alpha=0.85))

    cbar = plt.colorbar(scatter, ax=ax, label='Layer Index', shrink=0.75, pad=0.02)

    ax.set_xlabel("Mean Weight Effective Rank Ratio (Exp. 3)", fontweight='bold')
    ax.set_ylabel("Perturbation Gap (Exp. 2) — higher = more nonlinear", fontweight='bold')
    ax.grid(True, alpha=0.2)

    fig.savefig(OUTDIR / "fig5_rank_vs_linearity.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig5_rank_vs_linearity.png")

make_rank_linearity()


# ── Figure 6: The Self-Destruct (Layer 35 story) ─────────────

def make_self_destruct():
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    fig.suptitle("Layer 35: The Dispersal Mechanism",
                 fontsize=15, y=0.98)

    # Panel 1: Cosine similarity
    ax = axes[0]
    ax.plot(layers, t4_cosine_raw, color=C_BLUE, linewidth=2)
    ax.fill_between(layers, 0, t4_cosine_raw, alpha=0.06, color=C_BLUE)
    ax.axvspan(34.5, 35.5, alpha=0.12, color=C_PURPLE)
    ax.annotate(f'{t4_cosine_raw[34]:.2f} → {t4_cosine_raw[35]:.2f}',
                xy=(35, t4_cosine_raw[35]), xytext=(20, 0.30),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5),
                fontsize=10, color=C_RED)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Mean Pairwise Cosine Similarity", fontweight='bold')
    ax.set_title("Token Similarity\n(lower = more separable)", fontsize=12, color=C_GRAY)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, 35.5)

    # Panel 2: Norms
    ax = axes[1]
    ax.plot(layers, t4_norm, color=C_ORANGE, linewidth=2)
    ax.fill_between(layers, 0, t4_norm, alpha=0.06, color=C_ORANGE)
    ax.axvspan(34.5, 35.5, alpha=0.12, color=C_PURPLE)
    ax.annotate(f'{t4_norm[34]:.0f} → {t4_norm[35]:.0f}\n(norm drops)',
                xy=(35, t4_norm[35]), xytext=(20, 480),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5),
                fontsize=10, color=C_RED)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Mean Representation Norm", fontweight='bold')
    ax.set_title("Norm Growth\n(drops at final layer)", fontsize=12, color=C_GRAY)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-0.5, 35.5)

    # Panel 3: Update-residual alignment
    ax = axes[2]
    bar_colors = [C_GREEN if v > 0 else C_RED for v in t4_update_alignment]
    bar_colors[35] = C_PURPLE
    ax.bar(layers, t4_update_alignment, color=bar_colors, alpha=0.6, width=0.8)
    ax.axhline(0, color=C_GRAY, linewidth=1, alpha=0.4)
    ax.annotate(f'cos = {t4_update_alignment[35]:.2f}',
                xy=(35, t4_update_alignment[35]), xytext=(18, -0.55),
                arrowprops=dict(arrowstyle='->', color=C_PURPLE, lw=1.5),
                fontsize=10, color=C_PURPLE)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("cos(update, residual)", fontweight='bold')
    ax.set_title("Update Direction\n(negative = opposes residual)", fontsize=12, color=C_GRAY)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-0.85, 0.55)
    ax.set_xlim(-0.5, 35.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / "fig6_self_destruct.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig6_self_destruct.png")

make_self_destruct()


# ── Figure 7: Practical Implications — LoRA Rank Guide ────────

def make_lora_guide():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Non-Uniform LoRA: Match Rank to Actual Capacity",
                 fontsize=14, pad=15)

    matrices = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    labels = ['Q', 'K', 'V', 'O', 'Gate', 'Up', 'Down']

    plateau_ranks = []
    late_ranks = []
    for m in matrices:
        p_vals = [t9['per_layer'][str(i)][m]['effective_rank_ratio'] for i in range(17)]
        l_vals = [t9['per_layer'][str(i)][m]['effective_rank_ratio'] for i in range(17, 36)]
        plateau_ranks.append(np.mean(p_vals))
        late_ranks.append(np.mean(l_vals))

    x = np.arange(len(matrices))
    width = 0.35
    bars1 = ax.bar(x - width/2, plateau_ranks, width, label='Plateau (L0-16)', color=C_BLUE, alpha=0.7)
    bars2 = ax.bar(x + width/2, late_ranks, width, label='Late (L17-35)', color=C_ORANGE, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylabel("Mean Effective Rank Ratio", fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')

    gate_idx = 4
    diff = late_ranks[gate_idx] - plateau_ranks[gate_idx]
    ax.annotate(f'+{diff:.2f}', xy=(gate_idx + width/2, late_ranks[gate_idx]),
                xytext=(gate_idx - 1.5, late_ranks[gate_idx] + 0.12),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
                fontsize=9.5, color=C_RED)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8.5, color=C_BLUE)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.2f}',
                ha='center', va='bottom', fontsize=8.5, color=C_ORANGE)

    ax.set_ylim(0, 0.85)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig7_lora_guide.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig7_lora_guide.png")

make_lora_guide()


# ── Load additional data for new figures ─────────────────────

with open("experiments/t2_layer_knockout/results/results.json") as f:
    t2 = json.load(f)

with open("experiments/t1_logit_lens/results/summary.json") as f:
    t1 = json.load(f)

# T-2 criticality per layer
t2_criticality = []
for i in range(N_LAYERS):
    t2_criticality.append(t2['single_knockouts']['knockouts'][str(i)]['loss_ratio'])
t2_criticality = np.array(t2_criticality)

# T-1 logit lens accuracy per layer
t1_acc = []
for i in range(N_LAYERS):
    key = f"layer_{i}"
    t1_acc.append(t1['summary']['per_layer'][key]['top1_accuracy'])
t1_acc = np.array(t1_acc)

# T-7 PCA-aligned gap data
t7_on_manifold = []
t7_off_manifold = []
t7_gap_ratio = []
t7_eff_rank = []
for i in range(N_LAYERS):
    key = f"layer_{i}"
    d = t7['pca_aligned_gap'][key]
    t7_on_manifold.append(d['on_manifold_gap_mean'])
    t7_off_manifold.append(d['off_manifold_gap_mean'])
    t7_gap_ratio.append(d['gap_ratio'])
    t7_eff_rank.append(d['effective_rank'])
t7_on_manifold = np.array(t7_on_manifold)
t7_off_manifold = np.array(t7_off_manifold)
t7_gap_ratio = np.array(t7_gap_ratio)
t7_eff_rank = np.array(t7_eff_rank)

# T-4 singular value spectra
t4_sv = {}
for i in range(N_LAYERS):
    key = f"layer_{i}"
    t4_sv[i] = np.array(t4['sv_spectra'][key])

# T-4 update correlation matrix
t4_ucm = np.array(t4['update_correlation_matrix'])


# ── Figure 8: Cross-Experiment Alignment (the "money figure") ─

def make_cross_experiment():
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle("Four Views of the Same Transformer — Phase Boundaries Align",
                 fontsize=16, y=0.96)

    phases = [
        (0, 0.5, C_BLUE, 'Expansion'),
        (0.5, 5.5, C_RED, 'First\nCompr.'),
        (5.5, 15.5, C_GREEN, 'Distributed\nProcessing'),
        (15.5, 24.5, C_RED, 'Second\nCompr.'),
        (24.5, 34.5, C_ORANGE, 'Output\nPreparation'),
        (34.5, 35.5, C_PURPLE, 'Dispersal'),
    ]

    for ax in axes:
        for x0, x1, color, label in phases:
            ax.axvspan(x0, x1, alpha=0.05, color=color, zorder=0)
        ax.set_xlim(-0.5, 35.5)
        ax.grid(True, alpha=0.2)

    # Panel 1: Participation Ratio
    ax = axes[0]
    ax.plot(layers, t4_pr, color=C_BLUE, linewidth=2, zorder=5)
    ax.fill_between(layers, 0, t4_pr, alpha=0.07, color=C_BLUE)
    ax.set_ylabel("Participation\nRatio", fontweight='bold')
    ax.set_ylim(0, 230)
    ax.annotate(f'PR={t4_pr[16]:.1f}', xy=(16, t4_pr[16]), xytext=(19, 50),
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
                fontsize=9.5, color=C_RED)
    ax.set_title("Effective dimensionality of representations", fontsize=10, loc='left', color=C_GRAY)

    # Panel 2: Perturbation Gap
    ax = axes[1]
    ax.plot(layers, t7_gap, color=C_ORANGE, linewidth=2, zorder=5)
    ax.fill_between(layers, 0, t7_gap, alpha=0.07, color=C_ORANGE)
    ax.set_ylabel("Perturbation\nGap", fontweight='bold')
    ax.set_ylim(0.1, 0.28)
    ax.set_title("Nonlinearity — higher means more nonlinear", fontsize=10, loc='left', color=C_GRAY)

    # Panel 3: Mean Weight Rank
    ax = axes[2]
    ax.plot(layers, t9_mean_rank, color=C_GREEN, linewidth=2, zorder=5)
    ax.fill_between(layers, 0, t9_mean_rank, alpha=0.07, color=C_GREEN)
    ax.set_ylabel("Mean Weight\nRank", fontweight='bold')
    ax.set_ylim(0.3, 0.6)
    ax.set_title("Weight matrix effective capacity", fontsize=10, loc='left', color=C_GRAY)

    # Panel 4: Knockout Criticality (log scale)
    ax = axes[3]
    ax.bar(layers, t2_criticality, color=C_CYAN, alpha=0.65, width=0.8, zorder=5)
    ax.set_yscale('log')
    ax.set_ylabel("Knockout\nLoss Ratio", fontweight='bold')
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_title("Layer criticality — loss increase when removed", fontsize=10, loc='left', color=C_GRAY)
    ax.annotate(f'L0: {t2_criticality[0]:.0f}×', xy=(0, t2_criticality[0]),
                xytext=(3, 60), arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
                fontsize=9.5, color=C_RED)
    ax.annotate(f'L6: {t2_criticality[6]:.0f}×', xy=(6, t2_criticality[6]),
                xytext=(9, 40), arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.2),
                fontsize=9.5, color=C_RED)

    # Phase labels at top
    for x0, x1, color, label in phases:
        mid = (x0 + x1) / 2
        axes[0].text(mid, 225, label, ha='center', va='top', fontsize=7.5,
                     color=color, alpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTDIR / "fig8_cross_experiment.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig8_cross_experiment.png")

make_cross_experiment()


# ── Figure: PCA-Aligned Gap (dark theme replacement) ─────────

def make_pca_gap_dark():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("On-Manifold vs Off-Manifold Nonlinearity",
                 fontsize=14, y=1.0)

    # Panel 1: On vs Off manifold gap
    ax = axes[0]
    ax.plot(layers, t7_on_manifold, color=C_GREEN, linewidth=1.8, label='On-manifold (PCA)', marker='o', markersize=2.5, alpha=0.85)
    ax.plot(layers, t7_off_manifold, color=C_RED, linewidth=1.8, label='Off-manifold (random)', marker='s', markersize=2.5, alpha=0.85)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Perturbation Gap", fontweight='bold')
    ax.set_title("Nonlinearity by Direction", fontsize=12, color=C_GRAY)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 35.5)

    # Panel 2: Gap ratio
    ax = axes[1]
    ax.plot(layers, t7_gap_ratio, color=C_ORANGE, linewidth=1.8, marker='o', markersize=2.5, alpha=0.85)
    ax.axhline(1.0, color=C_RED, linestyle='--', alpha=0.35, linewidth=1.5)
    ax.text(20, 1.06, 'ratio = 1.0', fontsize=8, color=C_RED, alpha=0.6, ha='center')
    above = t7_gap_ratio >= 1.0
    ax.fill_between(layers, 1.0, t7_gap_ratio, where=above, alpha=0.15, color=C_RED)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("On/Off Gap Ratio", fontweight='bold')
    ax.set_title("Gap Ratio (>1 = more nonlinear on manifold)", fontsize=12, color=C_GRAY)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 35.5)

    # Panel 3: Effective manifold dimension
    ax = axes[2]
    ax.bar(layers, t7_eff_rank, color=C_CYAN, alpha=0.6, width=0.8)
    ax.axhline(np.mean(t7_eff_rank), color=C_ORANGE, linestyle='--', alpha=0.4, linewidth=1.5)
    ax.text(25, np.mean(t7_eff_rank) + 1.5, f'mean = {np.mean(t7_eff_rank):.0f}',
            fontsize=9, color=C_ORANGE)
    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Effective Rank\n(95% variance)", fontweight='bold')
    ax.set_title("Input Manifold Dimensionality", fontsize=12, color=C_GRAY)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, 35.5)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_pca_gap.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig_pca_gap.png (dark theme)")

make_pca_gap_dark()


# ── Figure: Update Correlation Matrix (dark theme replacement) ─

def make_update_correlation_dark():
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_title("Layer Update Correlation Matrix\ncos(mean_delta_l, mean_delta_m)",
                 fontsize=14, pad=15)

    # Use a diverging colormap that works on dark bg
    im = ax.imshow(t4_ucm, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')
    cbar.ax.yaxis.label.set_color('#c9d1d9')
    cbar.ax.tick_params(colors='#8b949e')

    ax.set_xlabel("Layer", fontweight='bold')
    ax.set_ylabel("Layer", fontweight='bold')
    ax.set_xticks(range(0, 36, 4))
    ax.set_yticks(range(0, 36, 4))

    # Block boundary annotations
    for boundary in [5.5, 15.5, 34.5]:
        ax.axhline(boundary, color='white', linewidth=1.0, alpha=0.4, linestyle='--')
        ax.axvline(boundary, color='white', linewidth=1.0, alpha=0.4, linestyle='--')

    # Block labels — along bottom x-axis, below tick labels
    ax.text(2.5, 37.5, 'Early', fontsize=9, color=C_RED, ha='center', alpha=0.8)
    ax.text(10.5, 37.5, 'Mid', fontsize=9, color=C_GREEN, ha='center', alpha=0.8)
    ax.text(25, 37.5, 'Late', fontsize=9, color=C_ORANGE, ha='center', alpha=0.8)
    ax.text(35, 37.5, 'L35', fontsize=9, color=C_PURPLE, ha='center', alpha=0.8)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_update_correlation.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig_update_correlation.png (dark theme)")

make_update_correlation_dark()


# ── Figure: SV Spectra at Key Layers ─────────────────────────

def make_sv_spectra():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Singular Value Spectra: Why Layer 16 Is a Pinhole",
                 fontsize=14, y=1.0)

    showcase = [(10, 'Layer 10 — Distributed Processing', C_GREEN),
                (16, 'Layer 16 — The Bottleneck', C_RED),
                (35, 'Layer 35 — Dispersal', C_PURPLE)]

    for ax, (layer_idx, title, color) in zip(axes, showcase):
        sv = t4_sv[layer_idx]
        # Normalize to fraction of total variance
        sv_var = sv ** 2
        sv_frac = sv_var / sv_var.sum()
        cumulative = np.cumsum(sv_frac)

        n_show = min(50, len(sv))
        x = np.arange(1, n_show + 1)

        # Bar chart of variance fraction
        ax.bar(x, sv_frac[:n_show] * 100, color=color, alpha=0.6, width=0.8)
        ax.set_xlabel("Singular Value Index", fontweight='bold')
        ax.set_ylabel("% of Total Variance", fontweight='bold')
        ax.set_title(title, fontsize=11, color=color, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate top-1
        ax.annotate(f'SV₁ = {sv_frac[0]*100:.1f}%', xy=(1, sv_frac[0]*100),
                    xytext=(10, sv_frac[0]*100 * 0.85),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                    fontsize=10, color='white')

        # Add cumulative line on twin axis
        ax2 = ax.twinx()
        ax2.plot(x, cumulative[:n_show] * 100, color='white', linewidth=1.5, alpha=0.6, linestyle='--')
        ax2.set_ylabel("Cumulative %", color='#8b949e', fontsize=9)
        ax2.set_ylim(0, 105)
        ax2.tick_params(axis='y', colors='#8b949e')

        # PR annotation
        pr = t4_pr[layer_idx]
        ax.text(0.97, 0.85, f'PR = {pr:.1f}', transform=ax.transAxes,
                fontsize=11, color=color, ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#161b22', edgecolor=color, alpha=0.9))

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig_sv_spectra.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig_sv_spectra.png")

make_sv_spectra()


# ── Figure: Logit Lens + PR Overlay ──────────────────────────

def make_logit_lens_overlay():
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.suptitle("When Predictions Form: Logit Lens Accuracy vs Geometric Bottleneck",
                 fontsize=14, y=0.98)

    phases = [
        (0, 0.5, C_BLUE, ''),
        (0.5, 5.5, C_RED, ''),
        (5.5, 15.5, C_GREEN, ''),
        (15.5, 24.5, C_RED, ''),
        (24.5, 34.5, C_ORANGE, ''),
        (34.5, 35.5, C_PURPLE, ''),
    ]
    for x0, x1, color, label in phases:
        ax1.axvspan(x0, x1, alpha=0.04, color=color, zorder=0)

    # Left axis: logit lens accuracy
    color_acc = C_CYAN
    ax1.plot(layers, t1_acc * 100, color=color_acc, linewidth=2, label='Top-1 Accuracy', zorder=5)
    ax1.fill_between(layers, 0, t1_acc * 100, alpha=0.06, color=color_acc)
    ax1.set_xlabel("Layer", fontweight='bold')
    ax1.set_ylabel("Top-1 Accuracy %", color=color_acc, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 105)
    ax1.set_xlim(-0.5, 35.5)
    ax1.grid(True, alpha=0.3)

    # Right axis: participation ratio
    ax2 = ax1.twinx()
    ax2.plot(layers, t4_pr, color=C_BLUE, linewidth=1.8, linestyle='--', alpha=0.55,
             label='Participation Ratio', zorder=4)
    ax2.set_ylabel("Participation Ratio", color=C_BLUE, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=C_BLUE)
    ax2.set_ylim(0, 230)

    ax1.axvspan(15.5, 24.5, alpha=0.08, color=C_RED, zorder=1)
    ax1.text(20, 92, 'Bottleneck\n(PR = 2-17)', fontsize=9.5, color=C_RED,
             ha='center', alpha=0.7)

    ax1.annotate('Predictions form\nafter the bottleneck', xy=(25, t1_acc[25]*100),
                xytext=(12, 75), arrowprops=dict(arrowstyle='->', color=C_ORANGE, lw=1.5),
                fontsize=10, color=C_ORANGE)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=10, framealpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTDIR / "fig9_logit_lens_overlay.png", dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved fig9_logit_lens_overlay.png")

make_logit_lens_overlay()


print("\nAll figures generated!")
