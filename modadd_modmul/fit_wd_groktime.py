#!/usr/bin/env python3
"""
Fit the relationship between weight decay and grokking time.

Models tested:
  1. Power law:    t_grok = a * WD^{-b}
  2. Exponential:  t_grok = a * exp(-b * WD)
  3. Log-linear:   log(t_grok) = a + b * log(WD)  (equivalent to power law)
  4. Reciprocal:   t_grok = a / WD + c

Uses data from all 3 seeds × 5 grokking WD values (0.1, 0.2, 0.3, 0.5, 1.0).

Figures:
  figMT_F1 — Power law fit with confidence band
  figMT_F2 — All fits comparison (lin-lin and log-log)
  figMT_F3 — Residuals
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = Path(__file__).parent / "plots"
RESULTS_DIR = Path(__file__).parent / "results"
SEEDS = [42, 137, 2024]


def load_grok_data():
    """Load grokking times from the aggregate results."""
    # Load from hessian_aggregate's load_all logic
    sys.path.insert(0, str(Path(__file__).parent))
    from hessian_aggregate import load_all
    _, meta = load_all()

    wd_vals = []
    t_add = []
    t_mul = []
    t_mean = []

    for wd in sorted(meta.keys()):
        if wd == 0.0:
            continue  # never groks
        for seed in SEEDS:
            ga = meta[wd][seed].get("grok_step_add")
            gm = meta[wd][seed].get("grok_step_mul")
            if ga is not None and gm is not None:
                wd_vals.append(wd)
                t_add.append(ga)
                t_mul.append(gm)
                t_mean.append((ga + gm) / 2)

    return np.array(wd_vals), np.array(t_add), np.array(t_mul), np.array(t_mean)


# ═══════════════════════════════════════════════════════════════════════════
# Fit models
# ═══════════════════════════════════════════════════════════════════════════

def power_law(x, a, b):
    return a * x ** (-b)

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def reciprocal(x, a, b, c):
    return a / (x + b) + c

def log_linear(x, a, b):
    """log(t) = a + b*log(wd) => t = exp(a) * wd^b"""
    return a + b * x


def fit_all(wd, t):
    """Fit all models, return dict of results."""
    results = {}

    # 1. Power law: t = a * WD^{-b}
    try:
        popt, pcov = curve_fit(power_law, wd, t, p0=[10000, 1.0], maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        t_pred = power_law(wd, *popt)
        ss_res = np.sum((t - t_pred) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results["power_law"] = {
            "params": {"a": popt[0], "b": popt[1]},
            "errors": {"a": perr[0], "b": perr[1]},
            "r2": r2,
            "formula": f"t = {popt[0]:.0f} · WD^{{-{popt[1]:.3f}}}",
            "pred_fn": lambda x, p=popt: power_law(x, *p),
        }
    except Exception as e:
        print(f"  Power law fit failed: {e}")

    # 2. Log-log linear (same as power law but fit in log space — more robust)
    log_wd = np.log(wd)
    log_t = np.log(t)
    popt_ll, pcov_ll = curve_fit(log_linear, log_wd, log_t, p0=[10, -1])
    perr_ll = np.sqrt(np.diag(pcov_ll))
    log_t_pred = log_linear(log_wd, *popt_ll)
    ss_res = np.sum((log_t - log_t_pred) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r2_ll = 1 - ss_res / ss_tot
    # Convert back: t = exp(a) * WD^b
    A = np.exp(popt_ll[0])
    b_exp = popt_ll[1]
    results["log_log"] = {
        "params": {"A": A, "b": b_exp, "intercept": popt_ll[0], "slope": popt_ll[1]},
        "errors": {"intercept": perr_ll[0], "slope": perr_ll[1]},
        "r2": r2_ll,
        "formula": f"t = {A:.0f} · WD^{{{b_exp:.3f}}}  [R²={r2_ll:.6f} in log-log]",
        "pred_fn": lambda x, a=popt_ll[0], b=popt_ll[1]: np.exp(a + b * np.log(x)),
    }

    # 3. Exponential: t = a * exp(-b * WD)
    try:
        popt_e, pcov_e = curve_fit(exponential_decay, wd, t, p0=[100000, 3.0], maxfev=10000)
        perr_e = np.sqrt(np.diag(pcov_e))
        t_pred_e = exponential_decay(wd, *popt_e)
        ss_res = np.sum((t - t_pred_e) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2_e = 1 - ss_res / ss_tot
        results["exponential"] = {
            "params": {"a": popt_e[0], "b": popt_e[1]},
            "errors": {"a": perr_e[0], "b": perr_e[1]},
            "r2": r2_e,
            "formula": f"t = {popt_e[0]:.0f} · exp(-{popt_e[1]:.3f} · WD)",
            "pred_fn": lambda x, p=popt_e: exponential_decay(x, *p),
        }
    except Exception as e:
        print(f"  Exponential fit failed: {e}")

    # 4. Reciprocal: t = a / (WD + b) + c
    try:
        popt_r, pcov_r = curve_fit(reciprocal, wd, t, p0=[10000, 0.01, 5000], maxfev=10000)
        perr_r = np.sqrt(np.diag(pcov_r))
        t_pred_r = reciprocal(wd, *popt_r)
        ss_res = np.sum((t - t_pred_r) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        r2_r = 1 - ss_res / ss_tot
        results["reciprocal"] = {
            "params": {"a": popt_r[0], "b": popt_r[1], "c": popt_r[2]},
            "errors": {"a": perr_r[0], "b": perr_r[1], "c": perr_r[2]},
            "r2": r2_r,
            "formula": f"t = {popt_r[0]:.0f} / (WD + {popt_r[1]:.4f}) + {popt_r[2]:.0f}",
            "pred_fn": lambda x, p=popt_r: reciprocal(x, *p),
        }
    except Exception as e:
        print(f"  Reciprocal fit failed: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def fig_f1_power_law(wd, t_add, t_mul, t_mean, fits):
    """Power law fit with per-seed data and confidence band."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    wd_smooth = np.linspace(0.05, 1.1, 200)

    # Left: log-log
    ax = axes[0]
    # Individual points
    for seed, marker in zip(SEEDS, ["o", "s", "^"]):
        mask = np.arange(len(wd)) % 3  # seeds repeat in order
        # Just plot all points
    ax.scatter(wd, t_add, c="#1f77b4", marker="o", s=60, alpha=0.6, label="ADD (all seeds)")
    ax.scatter(wd, t_mul, c="#d62728", marker="s", s=60, alpha=0.6, label="MUL (all seeds)")

    # Best fit (log-log)
    ll = fits["log_log"]
    t_fit = ll["pred_fn"](wd_smooth)
    ax.plot(wd_smooth, t_fit, "k-", lw=2.5, label=f"Fit: {ll['formula']}")

    # Confidence band (bootstrap-style from seed variability)
    unique_wds = np.unique(wd)
    means = []
    stds = []
    for w in unique_wds:
        mask = wd == w
        means.append(np.mean(t_mean[mask]))
        stds.append(np.std(t_mean[mask]))
    means = np.array(means)
    stds = np.array(stds)
    ax.fill_between(unique_wds, means - 2*stds, means + 2*stds, color="gray", alpha=0.15, label="±2σ (seeds)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Weight Decay", fontsize=13)
    ax.set_ylabel("Grokking Step", fontsize=13)
    ax.set_title("Log-Log: Power Law Fit", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # Right: lin-lin
    ax = axes[1]
    ax.scatter(wd, t_add, c="#1f77b4", marker="o", s=60, alpha=0.6, label="ADD")
    ax.scatter(wd, t_mul, c="#d62728", marker="s", s=60, alpha=0.6, label="MUL")
    ax.plot(wd_smooth, t_fit, "k-", lw=2.5, label="Power law fit")

    # Also plot other fits for comparison
    for name, color, ls in [("exponential", "#2ca02c", "--"),
                              ("reciprocal", "#9467bd", "-.")]:
        if name in fits:
            ax.plot(wd_smooth, fits[name]["pred_fn"](wd_smooth), color=color, ls=ls,
                     lw=2, label=f"{name} (R²={fits[name]['r2']:.4f})")

    ax.set_xlabel("Weight Decay", fontsize=13)
    ax.set_ylabel("Grokking Step", fontsize=13)
    ax.set_title("Linear Scale: All Fits", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Grokking Time vs Weight Decay — Curve Fits\n(3 seeds × 5 WD values, mean task)",
                  fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_F1_powerlaw_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_F1_powerlaw_fit.png")


def fig_f2_residuals(wd, t_mean, fits):
    """Residual plots for each fit."""
    unique_wds = np.unique(wd)
    # Compute per-WD means
    wd_means = []
    t_means = []
    for w in unique_wds:
        mask = wd == w
        wd_means.append(w)
        t_means.append(np.mean(t_mean[mask]))
    wd_means = np.array(wd_means)
    t_means = np.array(t_means)

    fig, axes = plt.subplots(1, len(fits), figsize=(5 * len(fits), 5))
    if len(fits) == 1:
        axes = [axes]

    for idx, (name, fit) in enumerate(fits.items()):
        ax = axes[idx]
        t_pred = fit["pred_fn"](wd_means)
        residuals = t_means - t_pred
        pct_residuals = 100 * residuals / t_means

        ax.bar(range(len(wd_means)), pct_residuals, color="#3498db", alpha=0.7, edgecolor="black")
        ax.set_xticks(range(len(wd_means)))
        ax.set_xticklabels([f"{w:.1f}" for w in wd_means])
        ax.set_xlabel("Weight Decay")
        ax.set_ylabel("Residual (%)")
        ax.set_title(f"{name}\nR²={fit['r2']:.6f}")
        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Fit Residuals (% of actual)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_F2_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_F2_residuals.png")


def fig_f3_loglog_detail(wd, t_add, t_mul, t_mean, fits):
    """Detailed log-log with separate ADD/MUL fits."""
    fig, ax = plt.subplots(figsize=(10, 7))

    wd_smooth = np.logspace(np.log10(0.05), np.log10(1.5), 200)

    # Fit ADD and MUL separately in log-log
    log_wd = np.log(wd)
    for t_arr, label, color, marker in [(t_add, "ADD", "#1f77b4", "o"),
                                         (t_mul, "MUL", "#d62728", "s")]:
        log_t = np.log(t_arr)
        popt, pcov = curve_fit(log_linear, log_wd, log_t)
        A = np.exp(popt[0])
        b = popt[1]

        # R² in log space
        log_pred = log_linear(log_wd, *popt)
        ss_res = np.sum((log_t - log_pred) ** 2)
        ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
        r2 = 1 - ss_res / ss_tot

        ax.scatter(wd, t_arr, c=color, marker=marker, s=80, alpha=0.6, zorder=5)
        t_fit = np.exp(popt[0] + popt[1] * np.log(wd_smooth))
        ax.plot(wd_smooth, t_fit, color=color, lw=2.5,
                label=f"{label}: t = {A:.0f}·WD^{{{b:.3f}}}  [R²={r2:.6f}]")

    # Combined fit
    ll = fits["log_log"]
    t_fit_all = ll["pred_fn"](wd_smooth)
    ax.plot(wd_smooth, t_fit_all, "k--", lw=2, alpha=0.5,
            label=f"Combined: {ll['formula']}")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Weight Decay", fontsize=13)
    ax.set_ylabel("Grokking Step", fontsize=13)
    ax.set_title("Power Law: t_grok ∝ WD^α  (separate ADD/MUL fits)", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # Add annotation for exponent
    b_val = ll["params"]["b"]
    ax.text(0.05, 0.05, f"Combined exponent: α = {b_val:.3f}",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "figMT_F3_loglog_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved figMT_F3_loglog_detail.png")


def main():
    PLOT_DIR.mkdir(exist_ok=True)

    print("Loading grokking data...")
    wd, t_add, t_mul, t_mean = load_grok_data()
    print(f"  {len(wd)} data points: WD ∈ {{{', '.join(f'{w:.1f}' for w in np.unique(wd))}}}")
    print(f"  t_grok range: [{t_mean.min():.0f}, {t_mean.max():.0f}]")

    print(f"\nFitting models to mean(t_add, t_mul)...")
    fits = fit_all(wd, t_mean)

    print(f"\n{'='*70}")
    print("  FIT RESULTS")
    print(f"{'='*70}")
    for name, fit in sorted(fits.items(), key=lambda x: -x[1]["r2"]):
        print(f"\n  {name}:")
        print(f"    Formula:  {fit['formula']}")
        print(f"    R²:       {fit['r2']:.8f}")
        for k, v in fit["params"].items():
            err = fit["errors"].get(k, float("nan"))
            print(f"    {k:>12s} = {v:.6f} ± {err:.6f}")

    # Pearson r in log-log space
    log_wd = np.log(wd)
    log_t = np.log(t_mean)
    r_pearson, p_pearson = pearsonr(log_wd, log_t)
    print(f"\n  Pearson r (log-log): {r_pearson:.6f}  (p = {p_pearson:.2e})")

    print(f"\n  Generating figures...")
    fig_f1_power_law(wd, t_add, t_mul, t_mean, fits)
    fig_f2_residuals(wd, t_mean, fits)
    fig_f3_loglog_detail(wd, t_add, t_mul, t_mean, fits)

    print("\nDone.")


if __name__ == "__main__":
    main()
