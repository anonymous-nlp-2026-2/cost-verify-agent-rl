"""Credit Assignment: Why Context Window Size Matters for Episode-Level RL

Fig 1: Analogy using discounted MDP to illustrate reward visibility.
Fig 2: Causal dilution model for step_independent mode (combinatorial).
"""

import numpy as np
from math import comb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

gamma = 0.99
T = 35  # average episode length from ALFWorld training logs
K = 4   # critical decision points per episode (typical for ALFWorld)
h_values = np.array([2, 5, 10, 20, 35])
h_continuous = np.linspace(1, 35, 200)

# --- Fig 1: Effective discount analogy ---
effective_discount_cont = gamma ** np.maximum(T - h_continuous, 0)
effective_discount_pts = gamma ** np.maximum(T - h_values, 0)

# --- Fig 2: Causal dilution P(window contains >= 1 critical point) ---
def p_contains_critical(h, T_total, K_crit):
    """P(window of size h contains at least 1 of K critical points in T steps)."""
    h_clamp = min(int(h), T_total)
    if T_total - K_crit < 0 or h_clamp > T_total - K_crit:
        return 1.0
    return 1.0 - comb(T_total - K_crit, h_clamp) / comb(T_total, h_clamp)

p_critical_cont = np.array([p_contains_critical(h, T, K) for h in h_continuous])
p_critical_pts = np.array([p_contains_critical(h, T, K) for h in h_values])

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

fig.suptitle('Credit Assignment: Why Context Window Size Matters for Episode-Level RL',
             fontsize=12, fontweight='bold', y=0.98)

# Plot 1: Effective Discount (analogy)
ax1.plot(h_continuous, effective_discount_cont, 'b-', linewidth=1.5)
ax1.scatter(h_values, effective_discount_pts, c='b', zorder=5, s=30)

exp028c_val = gamma ** (T - 2)
ax1.annotate('exp028c (h=2)\n$\\gamma^{33}$=%.3f' % exp028c_val,
             xy=(2, exp028c_val), xytext=(12, exp028c_val - 0.15),
             fontsize=8.5, ha='left',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             color='red')

ax1.annotate('exp029 (h=T)\n$\\gamma^{0}$=1.0',
             xy=(35, 1.0), xytext=(28, 0.72),
             fontsize=8.5, ha='left',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2),
             color='darkgreen')

ax1.set_xlabel('Context window size $h$')
ax1.set_ylabel('Effective discount $\\gamma^{\\max(T-h,\\,0)}$')
ax1.set_title('Analogy: Reward Visibility\nunder Discounted MDP')
ax1.set_xlim(0, 38)
ax1.set_ylim(0, 1.05)
ax1.xaxis.set_major_locator(MultipleLocator(5))
ax1.grid(True, alpha=0.3)

# Plot 2: Causal Dilution
ax2.plot(h_continuous, p_critical_cont, 'b-', linewidth=1.5)
ax2.scatter(h_values, p_critical_pts, c='b', zorder=5, s=30)

p_h2 = p_contains_critical(2, T, K)
ax2.annotate('exp028c (h=2)\nP=%.2f' % p_h2,
             xy=(2, p_h2), xytext=(8, p_h2 - 0.12),
             fontsize=8.5, ha='left',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2),
             color='red')

p_full = p_contains_critical(T, T, K)
ax2.annotate('exp029 (h=T)\nP=%.2f' % p_full,
             xy=(35, p_full), xytext=(28, 0.72),
             fontsize=8.5, ha='left',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2),
             color='darkgreen')

ax2.set_xlabel('Context window size $h$')
ax2.set_ylabel('$P$(window contains $\\geq 1$ critical point)')
ax2.set_title('P(Window Contains Critical Decision Point)\n$T=35,\\; K=4$ critical points')
ax2.set_xlim(0, 38)
ax2.set_ylim(0, 1.05)
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.grid(True, alpha=0.3)

# Footnote for Fig 1
fig.text(0.02, 0.01,
         "Note: GRPO uses flat per-window advantages without γ discount.\n"
         "This analogy illustrates the information-theoretic argument\n"
         "for why larger context windows improve credit assignment.",
         fontsize=7.5, va='bottom', ha='left', style='italic', color='0.3')

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

out_dir = './analysis/figures'
import os
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f'{out_dir}/credit_assignment.pdf', bbox_inches='tight')
plt.savefig(f'{out_dir}/credit_assignment.png', bbox_inches='tight', dpi=200)
print("Saved: credit_assignment.pdf, credit_assignment.png")
print(f"  exp028c (h=2): P(critical)={p_h2:.4f}")
print(f"  exp029  (h=T): P(critical)={p_full:.4f}")
print(f"  Causal dilution ratio: {p_full/p_h2:.2f}x")
