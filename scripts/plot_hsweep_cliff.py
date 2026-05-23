#!/usr/bin/env python3
"""
Plot h-sweep cliff figure for paper.
Shows performance collapse as context window size h decreases.
Input: metrics from h-sweep runs (h={2,5,10,20,full} x 3 seeds x 15 steps)
Output: PDF + PNG figures for paper submission
"""
import argparse
import re
import os
import glob
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


H_VALUES = [2, 5, 10, 20, 'full']
H_LABELS = ['2', '5', '10', '20', 'full']
COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
N_STEPS = 15

VAL_SUCCESS_KEY = r"val-env/AlfworldSG/success:np\.float64\(([^)]+)\)"
TRAIN_SUCCESS_KEY = r"train/AlfworldSG/success:np\.float64\(([^)]+)\)"


def parse_log_file(log_path):
    """Parse a tee'd log or Ray worker log for step metrics."""
    steps = {}
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            # Strip ANSI codes
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            # Strip Ray worker prefix like "(TaskRunner pid=123456) "
            line = re.sub(r'^\([^)]+\)\s*', '', line)
            if not line.startswith('step:'):
                continue
            step_match = re.match(r'step:(\d+)', line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            val_match = re.search(VAL_SUCCESS_KEY, line)
            train_match = re.search(TRAIN_SUCCESS_KEY, line)
            val_success = float(val_match.group(1)) if val_match else None
            train_success = float(train_match.group(1)) if train_match else None
            steps[step] = {
                'val_success': val_success,
                'train_success': train_success,
            }
    return steps


def find_ray_worker_logs(ray_base="/data/tmp/ray"):
    """Find all Ray worker log files, sorted by session (newest first)."""
    sessions = sorted(glob.glob(os.path.join(ray_base, "session_*")), reverse=True)
    worker_logs = []
    for session in sessions:
        logs_dir = os.path.join(session, "logs")
        if os.path.isdir(logs_dir):
            for f in sorted(os.listdir(logs_dir)):
                if f.startswith("worker-") and f.endswith(".out"):
                    worker_logs.append(os.path.join(logs_dir, f))
    return worker_logs


def load_real_data(data_dir, log_dir=None):
    """
    Load h-sweep metrics from log files.
    
    Strategy:
    1. Look for tee'd logs at log_dir/hsweep-h{H}-seed{N}.log
    2. Fall back to scanning Ray worker logs
    
    Returns: dict[h_value][seed] = {step: {'val_success': float, ...}}
    """
    results = defaultdict(dict)
    
    # Strategy 1: tee'd logs
    if log_dir is None:
        log_dir = "./runs/cost-verify-agent-rl"
    
    for h in H_VALUES:
        h_str = str(h)
        for seed in range(1, 6):
            log_path = os.path.join(log_dir, f"hsweep-h{h_str}-seed{seed}.log")
            if os.path.exists(log_path):
                steps = parse_log_file(log_path)
                if steps:
                    results[h_str][seed] = steps
    
    # Strategy 2: Ray worker logs (scan for step: lines with val metrics)
    if not results:
        ray_logs = find_ray_worker_logs()
        for log_path in ray_logs:
            steps = parse_log_file(log_path)
            if steps:
                # Try to identify h_value from wandb run name in the log
                # For now, store as unknown
                pass
    
    return results


def generate_demo_data():
    """Generate synthetic data demonstrating the expected cliff effect."""
    np.random.seed(42)
    results = {}
    
    # Expected final performance curve (cliff between h=5 and h=10)
    final_targets = {'full': 0.28, '20': 0.25, '10': 0.15, '5': 0.08, '2': 0.03}
    
    # Training dynamics: larger h learns faster and plateaus higher
    for h_str, target in final_targets.items():
        results[h_str] = {}
        for seed in range(1, 4):
            steps = {}
            noise_scale = 0.02
            # Sigmoid-like learning curve with different rates
            h_int = 50 if h_str == 'full' else int(h_str)
            rate = 0.15 + h_int * 0.005
            baseline = 0.05  # initial performance from SFT
            
            for step in range(N_STEPS + 1):
                if step == 0:
                    # val_before_train: all start near SFT baseline
                    val = baseline + np.random.normal(0, 0.01)
                else:
                    progress = 1 - np.exp(-rate * step)
                    val = baseline + (target - baseline) * progress
                    val += np.random.normal(0, noise_scale)
                val = max(0.0, min(1.0, val))
                steps[step] = {'val_success': val, 'train_success': val * 1.2}
            results[h_str][seed] = steps
    
    return results


def extract_curves(results):
    """
    Extract per-h training curves as arrays.
    Uses val_success when available, falls back to train_success.
    Returns: dict[h_str] = {'steps': array, 'mean': array, 'std': array,
                            'val_steps': array, 'val_mean': array, 'val_std': array}
    """
    curves = {}
    for h_str in H_LABELS:
        if h_str not in results:
            continue
        seeds_data = results[h_str]
        all_steps = set()
        for seed_steps in seeds_data.values():
            all_steps.update(seed_steps.keys())
        all_steps = sorted(all_steps)
        
        # Build matrix using best available metric per step
        matrix = []
        val_matrix = []
        for seed_steps in seeds_data.values():
            row = []
            val_row = []
            for s in all_steps:
                if s not in seed_steps:
                    row.append(np.nan)
                    val_row.append(np.nan)
                    continue
                d = seed_steps[s]
                # For overall curve: prefer val, fall back to train
                if d['val_success'] is not None:
                    row.append(d['val_success'])
                    val_row.append(d['val_success'])
                elif d['train_success'] is not None:
                    row.append(d['train_success'])
                    val_row.append(np.nan)
                else:
                    row.append(np.nan)
                    val_row.append(np.nan)
            matrix.append(row)
            val_matrix.append(val_row)
        
        matrix = np.array(matrix)
        val_matrix = np.array(val_matrix)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
            val_mean = np.nanmean(val_matrix, axis=0)
            val_std_arr = np.nanstd(val_matrix, axis=0)
        val_mask = ~np.isnan(val_mean)
        
        curves[h_str] = {
            'steps': np.array(all_steps),
            'mean': mean,
            'std': std,
            'val_steps': np.array(all_steps)[val_mask],
            'val_mean': val_mean[val_mask],
            'val_std': val_std_arr[val_mask],
            'n_seeds': matrix.shape[0],
        }
    
    return curves


def plot_cliff_figure(curves, output_dir, fmt=('pdf', 'png'), width='double'):
    """Generate the cliff figure (1a + 1b side by side)."""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    })
    
    if width == 'double':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.8))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 2.8))
    
    # === Figure 1a: Cliff Plot (final performance vs h) ===
    h_positions = list(range(len(H_LABELS)))
    final_means = []
    final_stds = []
    colors_used = []
    
    for i, h_str in enumerate(H_LABELS):
        if h_str in curves:
            curve = curves[h_str]
            # Use last val evaluation point as "final"
            if len(curve['val_mean']) > 0:
                final_means.append(curve['val_mean'][-1] * 100)
                final_stds.append(curve['val_std'][-1] * 100)
            else:
                final_means.append(curve['mean'][-1] * 100)
                final_stds.append(curve['std'][-1] * 100)
            colors_used.append(COLORS[i])
        else:
            final_means.append(np.nan)
            final_stds.append(0)
            colors_used.append(COLORS[i])
    
    final_means = np.array(final_means)
    final_stds = np.array(final_stds)
    
    ax1.errorbar(h_positions, final_means, yerr=final_stds,
                 fmt='o-', color='#2c3e50', markerfacecolor='#2c3e50',
                 markersize=6, capsize=4, capthick=1.5, linewidth=1.5,
                 elinewidth=1.2, label='Step-level')
    
    # Reference line: episode-level (h=full)
    if 'full' in curves:
        full_final = curves['full']['mean'][-1] * 100
        ax1.axhline(y=full_final, color='#9467bd', linestyle='--',
                    linewidth=1, alpha=0.7, label='Episode-level')
    
    # Highlight cliff region (shade between h=5 and h=10)
    ax1.axvspan(1.5, 2.5, alpha=0.08, color='red')
    
    # Collapse annotation (needs at least 4 non-NaN points)
    valid_points = [i for i in range(len(final_means)) if not np.isnan(final_means[i])]
    if len(valid_points) >= 4 and not np.isnan(final_means[2]):
        mid_y = (final_means[1] + final_means[2]) / 2
        ax1.annotate('', xy=(2, final_means[2]), xytext=(2, final_means[3]),
                     arrowprops=dict(arrowstyle='->', color='#d62728',
                                     lw=1.5, connectionstyle='arc3,rad=0.2'))
        ax1.text(2.3, mid_y, 'collapse', fontsize=7, color='#d62728',
                 fontstyle='italic', va='center')
    
    ax1.set_xticks(h_positions)
    ax1.set_xticklabels(H_LABELS)
    ax1.set_xlabel('Context Window Size ($h$)')
    ax1.set_ylabel('Final Success Rate (%)')
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, linestyle='--')
    handles, labels = ax1.get_legend_handles_labels()
    if labels:
        ax1.legend(loc='upper left', framealpha=0.8)
    
    # === Figure 1b: Training Curves ===
    for i, h_str in enumerate(H_LABELS):
        if h_str not in curves:
            continue
        curve = curves[h_str]
        steps = curve['steps']
        mean = curve['mean'] * 100
        std = curve['std'] * 100
        
        label = f'$h$={h_str}' if h_str != 'full' else '$h$=full'
        ax2.plot(steps, mean, color=COLORS[i], linewidth=1.5, label=label)
        ax2.fill_between(steps, mean - std, mean + std,
                         color=COLORS[i], alpha=0.15)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_ylim(bottom=0)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', framealpha=0.8, ncol=2)
    
    plt.tight_layout(pad=0.5)
    
    for f in fmt:
        out_path = os.path.join(output_dir, f'hsweep_cliff.{f}')
        dpi = 300 if f == 'png' else None
        fig.savefig(out_path, format=f, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot h-sweep cliff figure')
    parser.add_argument('--output_dir', type=str, default='./figures/',
                        help='Output directory for figures')
    parser.add_argument('--data_dir', type=str,
                        default='./RAGEN/',
                        help='Base directory containing wandb/ray data')
    parser.add_argument('--log_dir', type=str,
                        default='./runs/cost-verify-agent-rl',
                        help='Directory with tee\'d training logs')
    parser.add_argument('--format', nargs='+', default=['pdf', 'png'],
                        choices=['pdf', 'png', 'svg'],
                        help='Output formats')
    parser.add_argument('--width', choices=['single', 'double'],
                        default='double',
                        help='Figure width (single/double column)')
    parser.add_argument('--demo', action='store_true',
                        help='Use synthetic demo data')
    args = parser.parse_args()
    
    if args.demo:
        print("Using synthetic demo data...")
        results = generate_demo_data()
    else:
        print(f"Loading data from logs: {args.log_dir}")
        results = load_real_data(args.data_dir, args.log_dir)
        if not results:
            print("WARNING: No real data found. Use --demo for synthetic data.")
            return
    
    curves = extract_curves(results)
    print(f"Loaded curves for h values: {list(curves.keys())}")
    for h_str, c in curves.items():
        final = c['val_mean'][-1]*100 if len(c['val_mean']) > 0 else c['mean'][-1]*100
        print(f"  h={h_str}: {c['n_seeds']} seeds, "
              f"{len(c['steps'])} steps, "
              f"final={final:.1f}%")
    
    plot_cliff_figure(curves, args.output_dir, fmt=tuple(args.format),
                      width=args.width)


if __name__ == '__main__':
    main()
