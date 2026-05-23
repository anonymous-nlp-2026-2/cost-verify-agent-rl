"""
Advantage Sign Misattribution Analysis (plan_004)

Empirically tests: does advantage sign in step_independent GRPO
correlate more with episode outcome or with local action quality?

Key metrics:
  1. Point-biserial correlation: advantage_sign vs episode_outcome
  2. Point-biserial correlation: advantage_sign vs local_quality
  3. Magnitude position decay: gamma^k fit with R^2
  4. Episode-level within-episode variance
  5. Visualizations: heatmap + scatter plot
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse

DEFAULT_LOG = "./logs/hsweep_h2_seed2/advantage_log.jsonl"


def load_data(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def build_episodes(records):
    episodes = defaultdict(list)
    for r in records:
        episodes[r["episode_id"]].append(r)
    for eid in episodes:
        episodes[eid].sort(key=lambda x: x["window_position"])
    return episodes


def classify_episode_outcome(episodes):
    outcomes = {}
    for eid, windows in episodes.items():
        max_ret = max(w["step_return"] for w in windows)
        outcomes[eid] = 1 if max_ret > 1.0 else 0
    return outcomes


def analysis_1_sign_vs_outcome(records, episodes, outcomes):
    from scipy.stats import pointbiserialr

    adv_signs = []
    ep_outcomes = []
    for r in records:
        eid = r["episode_id"]
        if eid not in outcomes:
            continue
        adv_signs.append(1 if r["advantage"] > 0 else 0)
        ep_outcomes.append(outcomes[eid])

    adv_signs = np.array(adv_signs)
    ep_outcomes = np.array(ep_outcomes)

    sign_match = np.mean(adv_signs == ep_outcomes)
    r_pb, p_val = pointbiserialr(ep_outcomes, adv_signs)

    print("=" * 60)
    print("Analysis 1: Advantage Sign vs Episode Outcome")
    print("=" * 60)
    print(f"N samples: {len(adv_signs)}")
    print(f"Success episodes: {np.sum(list(outcomes.values()))}/{len(outcomes)}")
    print(f"Sign-outcome agreement: {sign_match:.4f}")
    print(f"Point-biserial r: {r_pb:.4f} (p={p_val:.2e})")
    print(f"  Prediction: r > 0.95 if sign fully determined by outcome")

    return {"sign_outcome_agreement": sign_match, "r_sign_outcome": r_pb, "p_sign_outcome": p_val}


def analysis_2_sign_vs_local_quality(records, episodes, outcomes):
    from scipy.stats import pointbiserialr, pearsonr

    adv_signs = []
    local_quality_binary = []
    step_returns = []
    advantages = []

    for r in records:
        adv_signs.append(1 if r["advantage"] > 0 else 0)
        local_quality_binary.append(1 if r["step_return"] > 0 else 0)
        step_returns.append(r["step_return"])
        advantages.append(r["advantage"])

    adv_signs = np.array(adv_signs)
    local_quality_binary = np.array(local_quality_binary)
    step_returns = np.array(step_returns)
    advantages = np.array(advantages)

    r_local_bin, p_local_bin = pointbiserialr(local_quality_binary, adv_signs)
    agreement_local = np.mean(adv_signs == local_quality_binary)
    r_cont, p_cont = pearsonr(advantages, step_returns)

    print("\n" + "=" * 60)
    print("Analysis 2: Advantage Sign vs Local Action Quality")
    print("=" * 60)
    print(f"Local quality proxy: step_return > 0")
    print(f"Sign-local_quality agreement: {agreement_local:.4f}")
    print(f"Point-biserial r (sign vs local): {r_local_bin:.4f} (p={p_local_bin:.2e})")
    print(f"Pearson r (advantage vs step_return): {r_cont:.4f} (p={p_cont:.2e})")
    print(f"  Prediction: r < 0.1 if sign misattribution is strong")
    print(f"  Alternative: r >> 0.1 means step_independent assigns credit locally")

    return {
        "r_sign_local": r_local_bin,
        "p_sign_local": p_local_bin,
        "agreement_local": agreement_local,
        "r_adv_return_continuous": r_cont,
    }


def analysis_3_magnitude_decay(records, episodes, outcomes, gamma=0.95):
    """Fit advantage magnitude vs distance-from-terminal to gamma^k.
    
    Uses pure-success episodes (all windows positive return) for clean signal.
    Terminal = highest window position in that episode.
    """
    from scipy.optimize import curve_fit

    print("\n" + "=" * 60)
    print("Analysis 3: Magnitude Position Decay (gamma^k fit)")
    print("=" * 60)

    # Pure-success: all windows have positive return AND multi-window
    pure_success_eids = []
    for eid, windows in episodes.items():
        if outcomes.get(eid) != 1:
            continue
        if all(w["step_return"] > 0 for w in windows) and len(windows) >= 2:
            pure_success_eids.append(eid)

    print(f"Pure-success multi-window episodes: {pure_success_eids}")

    positions = []
    magnitudes = []
    returns_normalized = []

    for eid in pure_success_eids:
        windows = episodes[eid]
        terminal_pos = max(w["window_position"] for w in windows)
        terminal_return = max(w["step_return"] for w in windows)
        for w in windows:
            dist = terminal_pos - w["window_position"]
            positions.append(dist)
            magnitudes.append(abs(w["advantage"]))
            returns_normalized.append(w["step_return"] / terminal_return)

    positions = np.array(positions, dtype=float)
    magnitudes = np.array(magnitudes, dtype=float)
    returns_normalized = np.array(returns_normalized)

    if len(positions) < 3:
        print(f"Insufficient data ({len(positions)} points). Need more multi-window success episodes.")
        return {"r_squared": None, "gamma_fit": None, "r_squared_return": None}

    def decay_model(k, A, g):
        return A * g ** k

    # Part A: step_return decay (theory verification)
    print(f"\n--- Step Return Decay (theory verification) ---")
    r2_ret = None
    gamma_ret = None
    try:
        popt_ret, _ = curve_fit(decay_model, positions, returns_normalized,
                                p0=[1.0, gamma], bounds=([0, 0], [2.0, 1.0]))
        A_ret, gamma_ret = popt_ret
        pred_ret = decay_model(positions, A_ret, gamma_ret)
        ss_res = np.sum((returns_normalized - pred_ret) ** 2)
        ss_tot = np.sum((returns_normalized - np.mean(returns_normalized)) ** 2)
        r2_ret = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Fitted: A={A_ret:.4f}, gamma={gamma_ret:.4f}")
        print(f"  R^2 = {r2_ret:.4f} (expect > 0.95 if returns follow gamma^k)")
    except Exception as e:
        print(f"  Fit failed: {e}")

    # Part B: advantage magnitude decay
    print(f"\n--- Advantage Magnitude Decay ---")
    r2_adv = None
    gamma_adv = None
    try:
        popt_adv, _ = curve_fit(decay_model, positions, magnitudes,
                                p0=[0.05, gamma], bounds=([0, 0], [np.inf, 1.0]))
        A_adv, gamma_adv = popt_adv
        pred_adv = decay_model(positions, A_adv, gamma_adv)
        ss_res = np.sum((magnitudes - pred_adv) ** 2)
        ss_tot = np.sum((magnitudes - np.mean(magnitudes)) ** 2)
        r2_adv = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Fitted: A={A_adv:.6f}, gamma={gamma_adv:.4f}")
        print(f"  R^2 = {r2_adv:.4f}")
    except Exception as e:
        print(f"  Fit failed: {e}")

    # Raw table
    print(f"\n{'Dist from terminal':>18} {'Mean |adv|':>12} {'Mean ret/R':>12} {'N':>5}")
    pos_groups_adv = defaultdict(list)
    pos_groups_ret = defaultdict(list)
    for p, m, r in zip(positions, magnitudes, returns_normalized):
        pos_groups_adv[int(p)].append(m)
        pos_groups_ret[int(p)].append(r)
    for p in sorted(pos_groups_adv.keys()):
        print(f"{p:>18} {np.mean(pos_groups_adv[p]):>12.6f} "
              f"{np.mean(pos_groups_ret[p]):>12.4f} {len(pos_groups_adv[p]):>5}")

    return {"r_squared": r2_adv, "gamma_fit": gamma_adv, "r_squared_return": r2_ret}


def analysis_4_episode_level_variance(records, episodes, outcomes):
    print("\n" + "=" * 60)
    print("Analysis 4: Within-Episode Advantage Variance")
    print("=" * 60)

    variances_success = []
    variances_fail = []

    for eid, windows in episodes.items():
        if len(windows) < 2:
            continue
        advs = [w["advantage"] for w in windows]
        var = np.var(advs)
        if outcomes.get(eid) == 1:
            variances_success.append(var)
        else:
            variances_fail.append(var)

    print(f"Multi-window episodes: {len(variances_success) + len(variances_fail)}")
    if variances_success:
        print(f"  Success episodes within-variance: {np.mean(variances_success):.8f} (n={len(variances_success)})")
    if variances_fail:
        print(f"  Failed episodes within-variance: {np.mean(variances_fail):.8f} (n={len(variances_fail)})")
    print(f"  Overall within-episode variance: {np.mean(variances_success + variances_fail):.8f}")
    print(f"  (Near 0 = episode-level mode; Non-zero = step-independent differentiates windows)")

    all_advs = [r["advantage"] for r in records]
    print(f"  Cross-sample variance (total): {np.var(all_advs):.8f}")

    return {
        "within_var_success": float(np.mean(variances_success)) if variances_success else 0,
        "within_var_fail": float(np.mean(variances_fail)) if variances_fail else 0,
    }


def analysis_5_visualizations(records, episodes, outcomes, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Advantage sign heatmap ---
    max_pos = max(r["window_position"] for r in records)
    sorted_eids = sorted(episodes.keys(),
                         key=lambda eid: (outcomes.get(eid, 0), max(w["step_return"] for w in episodes[eid])))

    heatmap = np.full((len(sorted_eids), max_pos + 1), np.nan)
    for i, eid in enumerate(sorted_eids):
        for w in episodes[eid]:
            heatmap[i, w["window_position"]] = np.sign(w["advantage"])

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.RdBu
    cmap.set_bad("white")
    im = ax.imshow(heatmap, aspect="auto", cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")

    n_success = sum(1 for o in outcomes.values() if o == 1)
    n_fail = len(outcomes) - n_success
    ax.axhline(n_fail - 0.5, color="black", linewidth=2, linestyle="--")
    ax.text(max_pos + 0.5, n_fail / 2, "fail", va="center", fontsize=10)
    ax.text(max_pos + 0.5, n_fail + n_success / 2, "success", va="center", fontsize=10)

    ax.set_xlabel("Window Position")
    ax.set_ylabel("Episodes (sorted by outcome)")
    ax.set_title("Advantage Sign Heatmap\n(blue=negative, red=positive)")
    plt.colorbar(im, ax=ax, label="sign(advantage)")
    plt.tight_layout()
    fig.savefig(output_dir / "advantage_sign_heatmap.png", dpi=150)
    plt.close()
    print(f"\nSaved: {output_dir / 'advantage_sign_heatmap.png'}")

    # --- Figure 2: Magnitude decay (pure-success episodes) ---
    pure_success_eids = [eid for eid, ws in episodes.items()
                         if outcomes.get(eid) == 1 and all(w["step_return"] > 0 for w in ws) and len(ws) >= 2]
    positions_arr = []
    magnitudes_arr = []
    returns_norm_arr = []
    for eid in pure_success_eids:
        windows = episodes[eid]
        terminal_pos = max(w["window_position"] for w in windows)
        terminal_ret = max(w["step_return"] for w in windows)
        for w in windows:
            dist = terminal_pos - w["window_position"]
            positions_arr.append(dist)
            magnitudes_arr.append(abs(w["advantage"]))
            returns_norm_arr.append(w["step_return"] / terminal_ret)
    positions_arr = np.array(positions_arr, dtype=float)
    magnitudes_arr = np.array(magnitudes_arr)
    returns_norm_arr = np.array(returns_norm_arr)

    if len(positions_arr) >= 3:
        def decay_model(k, A, g):
            return A * g ** k
        try:
            popt, _ = curve_fit(decay_model, positions_arr, magnitudes_arr,
                                p0=[0.05, 0.95], bounds=([0, 0], [np.inf, 1.0]))
            A_fit, gamma_fit = popt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(positions_arr, magnitudes_arr, alpha=0.7, label="Empirical")
            k_smooth = np.linspace(0, max(positions_arr), 50)
            ax1.plot(k_smooth, decay_model(k_smooth, A_fit, gamma_fit), "r-", linewidth=2,
                     label=f"Fit: {A_fit:.4f} * {gamma_fit:.3f}^k")
            ax1.set_xlabel("Distance from terminal (k)")
            ax1.set_ylabel("|advantage|")
            ax1.set_title("Advantage Magnitude Decay")
            ax1.legend()

            popt_r, _ = curve_fit(decay_model, positions_arr, returns_norm_arr,
                                  p0=[1.0, 0.95], bounds=([0, 0], [2.0, 1.0]))
            ax2.scatter(positions_arr, returns_norm_arr, alpha=0.7, color="green", label="Empirical")
            ax2.plot(k_smooth, decay_model(k_smooth, *popt_r), "r-", linewidth=2,
                     label=f"Fit: {popt_r[0]:.3f} * {popt_r[1]:.3f}^k")
            ax2.set_xlabel("Distance from terminal (k)")
            ax2.set_ylabel("step_return / R_terminal")
            ax2.set_title("Return Decay (theory verification)")
            ax2.legend()

            plt.tight_layout()
            fig.savefig(output_dir / "magnitude_decay_scatter.png", dpi=150)
            plt.close()
            print(f"Saved: {output_dir / 'magnitude_decay_scatter.png'}")
        except Exception as e:
            print(f"Decay plot failed: {e}")
    else:
        print("Insufficient pure-success episodes for magnitude decay plot.")

    # --- Figure 3: Advantage vs step_return scatter ---
    fig, ax = plt.subplots(figsize=(8, 5))
    advs = [r["advantage"] for r in records]
    rets = [r["step_return"] for r in records]
    colors = ["red" if outcomes.get(r["episode_id"], 0) == 1 else "blue" for r in records]
    ax.scatter(rets, advs, c=colors, alpha=0.5, s=20)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("step_return (local quality)")
    ax.set_ylabel("advantage")
    ax.set_title("Advantage vs Local Quality\n(red=success episode, blue=fail episode)")
    plt.tight_layout()
    fig.savefig(output_dir / "advantage_vs_local_quality.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'advantage_vs_local_quality.png'}")



def analysis_6_conditional_correlations(records, episodes, outcomes):
    from scipy.stats import pearsonr

    print("\n" + "=" * 60)
    print("Analysis 6: Conditional Correlations (controlling episode outcome)")
    print("=" * 60)

    advantages = []
    step_returns = []
    ep_outcomes = []
    for r in records:
        eid = r["episode_id"]
        if eid not in outcomes:
            continue
        advantages.append(r["advantage"])
        step_returns.append(r["step_return"])
        ep_outcomes.append(outcomes[eid])

    advantages = np.array(advantages)
    step_returns = np.array(step_returns)
    ep_outcomes = np.array(ep_outcomes)

    fail_mask = ep_outcomes == 0
    n_fail = int(np.sum(fail_mask))
    if n_fail > 2:
        r_fail, p_fail = pearsonr(advantages[fail_mask], step_returns[fail_mask])
        print(f"Within FAILED episodes:  r(adv, step_return) = {r_fail:.4f} (p={p_fail:.2e}, n={n_fail})")
    else:
        r_fail = None
        print(f"Within FAILED episodes:  insufficient data (n={n_fail})")

    succ_mask = ep_outcomes == 1
    n_succ = int(np.sum(succ_mask))
    if n_succ > 2:
        r_succ, p_succ = pearsonr(advantages[succ_mask], step_returns[succ_mask])
        print(f"Within SUCCESS episodes: r(adv, step_return) = {r_succ:.4f} (p={p_succ:.2e}, n={n_succ})")
    else:
        r_succ = None
        print(f"Within SUCCESS episodes: insufficient data (n={n_succ})")

    r_xy = float(np.corrcoef(advantages, step_returns)[0, 1])
    r_xz = float(np.corrcoef(advantages, ep_outcomes)[0, 1])
    r_yz = float(np.corrcoef(step_returns, ep_outcomes)[0, 1])

    denom = (1 - r_xz**2) * (1 - r_yz**2)
    if denom > 0:
        r_partial = (r_xy - r_xz * r_yz) / (denom ** 0.5)
    else:
        r_partial = float("nan")

    print(f"\nUnconditional:           r(adv, step_return) = {r_xy:.4f}")
    print(f"r(adv, outcome):         {r_xz:.4f}")
    print(f"r(step_return, outcome): {r_yz:.4f}")
    print(f"Partial correlation:     r(adv, step_return | outcome) = {r_partial:.4f}")

    print(f"\n--- Interpretation ---")
    if r_fail is not None and abs(r_fail) < 0.1:
        print(f"Within-failure r={r_fail:.4f} near 0: advantage does NOT track local quality in failed episodes.")
        print("Sign misattribution confirmed: advantage sign is episode-level, not step-level.")
    elif r_fail is not None:
        print(f"Within-failure r={r_fail:.4f}: some local credit assignment exists even in failed episodes.")

    return {
        "r_within_fail": float(r_fail) if r_fail is not None else None,
        "r_within_succ": float(r_succ) if r_succ is not None else None,
        "r_partial": float(r_partial),
        "r_unconditional": float(r_xy),
    }


def generate_summary(res1, res2, res3, res4, res6=None):
    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER (plan_004)")
    print("=" * 60)

    r_outcome = res1.get("r_sign_outcome", 0)
    r_local = res2.get("r_sign_local", 0)
    r_cont = res2.get("r_adv_return_continuous", 0)
    r_sq = res3.get("r_squared")
    r_sq_ret = res3.get("r_squared_return")

    print(f"\n1. Sign vs Episode Outcome:  r = {r_outcome:.4f}")
    print(f"2. Sign vs Local Quality:   r = {r_local:.4f}")
    print(f"3. Advantage vs Return:      r = {r_cont:.4f} (continuous)")
    if r_sq is not None:
        print(f"4. Adv decay fit R^2:        {r_sq:.4f}")
    if r_sq_ret is not None:
        print(f"   Return decay R^2:         {r_sq_ret:.4f}")
    print(f"5. Within-ep variance (fail): {res4.get('within_var_fail', 0):.8f}")

    if r_outcome > r_local:
        print("\n>> CONCLUSION: Sign misattribution confirmed.")
        print("   Advantage sign is primarily determined by episode outcome,")
        print("   not by local action quality within the window.")
    else:
        print("\n>> CONCLUSION: Step-independent mode provides local credit.")
        print("   Advantage sign correlates more with local quality than episode outcome.")
        print("   Sign misattribution is NOT the dominant pattern in this setting.")

    if res6:
        print(f"\n6. Conditional Correlations:")
        r_wf = res6.get('r_within_fail')
        r_ws = res6.get('r_within_succ')
        r_p = res6.get('r_partial')
        r_u = res6.get('r_unconditional')
        print(f"   Within FAILED:  r = {r_wf:.4f}" if r_wf is not None else "   Within FAILED:  r = N/A")
        print(f"   Within SUCCESS: r = {r_ws:.4f}" if r_ws is not None else "   Within SUCCESS: r = N/A")
        print(f"   Partial (|outcome): r = {r_p:.4f}" if r_p is not None else "   Partial: N/A")
        print(f"   Unconditional:      r = {r_u:.4f}" if r_u is not None else "   Unconditional: N/A")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advantage sign misattribution analysis (plan_004)")
    parser.add_argument("--input", default=DEFAULT_LOG, help="Path to advantage_log.jsonl")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor for decay fit")
    parser.add_argument("--output-dir", default=None, help="Directory for plots (default: same as input)")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Data not available yet: {path}")
        print("Run h=2 seed with LOG_PER_SAMPLE_ADV=1 first.")
        exit(0)

    output_dir = Path(args.output_dir) if args.output_dir else path.parent / "analysis"

    records = load_data(path)
    print(f"Loaded {len(records)} records from {path}")

    episodes = build_episodes(records)
    outcomes = classify_episode_outcome(episodes)
    n_success = sum(outcomes.values())
    print(f"Episodes: {len(episodes)} ({n_success} success, {len(episodes) - n_success} fail)")

    res1 = analysis_1_sign_vs_outcome(records, episodes, outcomes)
    res2 = analysis_2_sign_vs_local_quality(records, episodes, outcomes)
    res3 = analysis_3_magnitude_decay(records, episodes, outcomes, gamma=args.gamma)
    res4 = analysis_4_episode_level_variance(records, episodes, outcomes)
    res6 = analysis_6_conditional_correlations(records, episodes, outcomes)
    analysis_5_visualizations(records, episodes, outcomes, output_dir)
    generate_summary(res1, res2, res3, res4, res6)
