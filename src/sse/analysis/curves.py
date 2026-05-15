"""Scaling curve analysis — plotting, curve fitting, and emergence detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t as t_dist

matplotlib.use("Agg")

PYTHIA_PARAMS: dict[str, float] = {
    "70m": 70e6,
    "160m": 160e6,
    "410m": 410e6,
    "1b": 1e9,
    "1.4b": 1.4e9,
    "2.8b": 2.8e9,
    "6.9b": 6.9e9,
    "12b": 12e9,
}


def _sigmoid(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0)))


def _is_accuracy_metric(metric: str) -> bool:
    return "accuracy" in metric or "acc" in metric


def plot_scaling_curve(
    df: pd.DataFrame,
    task: str,
    metric: str,
    ax: plt.Axes | None = None,
    output_dir: Path | None = None,
) -> plt.Axes:
    subset = df[(df["task_name"] == task) & (df["metric_name"] == metric)]

    x_params = np.array([PYTHIA_PARAMS[s] for s in subset["model_size"]])
    y_values = np.array(subset["metric_value"].values, dtype=float)

    sort_idx = np.argsort(x_params)
    x_params = x_params[sort_idx]
    y_values = y_values[sort_idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    is_acc = _is_accuracy_metric(metric)
    flat_zero = is_acc and y_values.max() < 0.01

    if is_acc and not flat_zero:
        n_vals = subset["n_examples"].values[sort_idx] if "n_examples" in subset else None
        if n_vals is not None:
            se = np.sqrt(y_values * (1 - y_values) / n_vals)
            ax.errorbar(
                x_params,
                y_values,
                yerr=1.96 * se,
                fmt="none",
                color="gray",
                alpha=0.5,
                capsize=3,
            )

    ax.scatter(x_params, y_values, zorder=5, s=60)

    if flat_zero:
        ax.set_ylim(-0.05, 1.0)
    elif len(x_params) >= 3:
        log_x = np.log10(x_params)
        x_smooth = np.logspace(np.log10(x_params.min()), np.log10(x_params.max()), 100)
        log_x_smooth = np.log10(x_smooth)

        try:
            if is_acc:
                p0 = [max(y_values.max(), 0.5), 2.0, np.median(log_x)]
                bounds = ([0, 0.01, log_x.min() - 2], [1.5, 20, log_x.max() + 2])
                popt, _ = curve_fit(
                    _sigmoid,
                    log_x,
                    y_values,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000,
                )
                y_fit = _sigmoid(log_x_smooth, *popt)
            else:
                coeffs = np.polyfit(log_x, y_values, 1)
                y_fit = np.polyval(coeffs, log_x_smooth)
                y_pred = np.polyval(coeffs, log_x)
                ss_res = np.sum((y_values - y_pred) ** 2)
                ss_tot = np.sum((y_values - y_values.mean()) ** 2)
                r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                ax.text(
                    0.05,
                    0.95,
                    f"slope={coeffs[0]:.3f}\nR²={r_sq:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                    family="monospace",
                )

            ax.plot(x_smooth, y_fit, "--", alpha=0.7, linewidth=2)
        except RuntimeError:
            pass

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel(metric)
    ax.set_title(f"{task} — {metric}")
    ax.grid(True, alpha=0.3)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig = ax.get_figure()
        fig.savefig(output_dir / f"{task}_{metric}.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"{task}_{metric}.pdf", bbox_inches="tight")

    return ax


def plot_emergence_comparison(
    df: pd.DataFrame,
    task: str,
    accuracy_metric: str = "accuracy",
    logprob_metric: str = "mean_logprob_correct",
    output_dir: Path | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_scaling_curve(df, task, accuracy_metric, ax=axes[0])
    axes[0].set_title("Discontinuous (Accuracy)")

    plot_scaling_curve(df, task, logprob_metric, ax=axes[1])
    axes[1].set_title("Continuous (Log-prob)")

    fig.suptitle(f"Emergence Comparison — {task}", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{task}_emergence.png", dpi=150, bbox_inches="tight")
        fig.savefig(output_dir / f"{task}_emergence.pdf", bbox_inches="tight")

    return fig


def fit_emergence_threshold(
    df: pd.DataFrame,
    task: str,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    subset = df[(df["task_name"] == task) & (df["metric_name"] == "accuracy")]

    x_params = np.array([PYTHIA_PARAMS[s] for s in subset["model_size"]])
    y_values = np.array(subset["metric_value"].values, dtype=float)

    log_x = np.log10(x_params)

    p0 = [max(y_values.max(), 0.5), 2.0, np.median(log_x)]
    bounds = ([0, 0.01, log_x.min() - 2], [1.5, 20, log_x.max() + 2])

    try:
        popt, _ = curve_fit(_sigmoid, log_x, y_values, p0=p0, bounds=bounds, maxfev=5000)
    except RuntimeError:
        return {
            "threshold_params": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }

    def _find_threshold_crossing(params):
        L, k, x0 = params
        if threshold >= L:
            return float("nan")
        return 10 ** (x0 - np.log(L / threshold - 1) / k)

    threshold_params = _find_threshold_crossing(popt)

    rng = np.random.default_rng(seed)
    bootstrap_thresholds = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(log_x), size=len(log_x), replace=True)
        bx = log_x[idx]
        by = y_values[idx]
        try:
            bpopt, _ = curve_fit(_sigmoid, bx, by, p0=popt, bounds=bounds, maxfev=2000)
            bt = _find_threshold_crossing(bpopt)
            if np.isfinite(bt):
                bootstrap_thresholds.append(bt)
        except RuntimeError:
            continue

    if bootstrap_thresholds:
        ci_lower = float(np.percentile(bootstrap_thresholds, 2.5))
        ci_upper = float(np.percentile(bootstrap_thresholds, 97.5))
    else:
        ci_lower = float("nan")
        ci_upper = float("nan")

    return {
        "threshold_params": float(threshold_params),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def plot_emergence_with_extrapolation(
    df: pd.DataFrame,
    task: str,
    accuracy_metric: str = "accuracy",
    logprob_metric: str = "mean_logprob_correct",
    target_scale: float = 10e9,
    output_dir: Path | None = None,
) -> dict[str, float | None]:
    """Two-panel plot with extrapolation beyond observed model sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    acc_subset = df[(df["task_name"] == task) & (df["metric_name"] == accuracy_metric)]
    lp_subset = df[(df["task_name"] == task) & (df["metric_name"] == logprob_metric)]

    predicted_crossing: float | None = None
    predicted_logprob: float | None = None
    predicted_logprob_ci_lower: float | None = None
    predicted_logprob_ci_upper: float | None = None

    # Left panel: accuracy with extrapolation
    if not acc_subset.empty:
        x_params = np.array([PYTHIA_PARAMS[s] for s in acc_subset["model_size"]])
        y_values = np.array(acc_subset["metric_value"].values, dtype=float)
        sort_idx = np.argsort(x_params)
        x_params, y_values = x_params[sort_idx], y_values[sort_idx]

        max_observed = x_params.max()
        log_x = np.log10(x_params)

        axes[0].scatter(x_params, y_values, zorder=5, s=60, label="Observed")

        if y_values.max() < 0.01:
            axes[0].set_ylim(-0.05, 1.0)
            axes[0].text(
                0.5,
                0.5,
                "Accuracy pinned at 0%\n(no meaningful fit)",
                transform=axes[0].transAxes,
                ha="center",
                va="center",
                fontsize=11,
                alpha=0.5,
            )
        else:
            x_extrap = np.logspace(log_x.min(), np.log10(target_scale), 200)
            log_x_extrap = np.log10(x_extrap)
            try:
                p0 = [max(y_values.max(), 0.5), 2.0, np.median(log_x)]
                bounds = ([0, 0.01, log_x.min() - 2], [1.5, 20, log_x.max() + 4])
                popt, _ = curve_fit(
                    _sigmoid,
                    log_x,
                    y_values,
                    p0=p0,
                    bounds=bounds,
                    maxfev=5000,
                )
                y_fit = _sigmoid(log_x_extrap, *popt)

                obs_mask = x_extrap <= max_observed
                ext_mask = x_extrap >= max_observed
                axes[0].plot(
                    x_extrap[obs_mask],
                    y_fit[obs_mask],
                    "--",
                    color="C0",
                    alpha=0.7,
                    linewidth=2,
                )
                axes[0].plot(
                    x_extrap[ext_mask],
                    y_fit[ext_mask],
                    "--",
                    color="C3",
                    alpha=0.7,
                    linewidth=2,
                    label="Extrapolated",
                )

                L, k, x0 = popt
                if 0.5 < L:
                    crossing_log = x0 - np.log(L / 0.5 - 1) / k
                    predicted_crossing = float(10**crossing_log)
                    axes[0].axhline(0.5, linestyle=":", color="gray", alpha=0.4)
                    if predicted_crossing <= target_scale:
                        axes[0].plot(
                            predicted_crossing,
                            0.5,
                            "*",
                            color="C3",
                            markersize=15,
                            zorder=10,
                            label=f"50% at {predicted_crossing:.1e}",
                        )
            except RuntimeError:
                pass

        axes[0].axvline(
            max_observed,
            linestyle="--",
            color="gray",
            alpha=0.5,
            linewidth=1,
        )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Parameters")
        axes[0].set_ylabel(accuracy_metric)
        axes[0].set_title("Accuracy (with extrapolation)")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)

    # Right panel: logprob with extrapolation
    if not lp_subset.empty:
        x_params_lp = np.array([PYTHIA_PARAMS[s] for s in lp_subset["model_size"]])
        y_lp = np.array(lp_subset["metric_value"].values, dtype=float)
        sort_idx = np.argsort(x_params_lp)
        x_params_lp, y_lp = x_params_lp[sort_idx], y_lp[sort_idx]

        max_obs_lp = x_params_lp.max()
        log_x_lp = np.log10(x_params_lp)
        x_extrap_lp = np.logspace(log_x_lp.min(), np.log10(target_scale), 200)
        log_x_extrap_lp = np.log10(x_extrap_lp)

        axes[1].scatter(x_params_lp, y_lp, zorder=5, s=60, label="Observed")

        coeffs = np.polyfit(log_x_lp, y_lp, 1)
        y_fit_lp = np.polyval(coeffs, log_x_extrap_lp)
        predicted_logprob = float(np.polyval(coeffs, np.log10(target_scale)))

        y_pred_lp = np.polyval(coeffs, log_x_lp)
        ss_res = np.sum((y_lp - y_pred_lp) ** 2)
        ss_tot = np.sum((y_lp - y_lp.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        n_pts = len(log_x_lp)
        predicted_logprob_ci_lower: float | None = None
        predicted_logprob_ci_upper: float | None = None
        if n_pts > 2:
            s_err = np.sqrt(ss_res / (n_pts - 2))
            x_mean = log_x_lp.mean()
            sx2 = np.sum((log_x_lp - x_mean) ** 2)
            se_pred = s_err * np.sqrt(
                1 + 1 / n_pts + (log_x_extrap_lp - x_mean) ** 2 / sx2,
            )
            t_val = t_dist.ppf(0.975, df=n_pts - 2)
            upper = y_fit_lp + t_val * se_pred
            lower = y_fit_lp - t_val * se_pred
            axes[1].fill_between(
                x_extrap_lp,
                lower,
                upper,
                alpha=0.12,
                color="C0",
                label="95% PI",
            )
            target_se = s_err * np.sqrt(
                1 + 1 / n_pts + (np.log10(target_scale) - x_mean) ** 2 / sx2,
            )
            predicted_logprob_ci_lower = float(predicted_logprob - t_val * target_se)
            predicted_logprob_ci_upper = float(predicted_logprob + t_val * target_se)

        obs_mask = x_extrap_lp <= max_obs_lp
        ext_mask = x_extrap_lp >= max_obs_lp
        axes[1].plot(
            x_extrap_lp[obs_mask],
            y_fit_lp[obs_mask],
            "--",
            color="C0",
            alpha=0.7,
            linewidth=2,
        )
        axes[1].plot(
            x_extrap_lp[ext_mask],
            y_fit_lp[ext_mask],
            "--",
            color="C3",
            alpha=0.7,
            linewidth=2,
            label="Extrapolated",
        )
        axes[1].axvline(
            max_obs_lp,
            linestyle="--",
            color="gray",
            alpha=0.5,
            linewidth=1,
        )
        axes[1].text(
            0.05,
            0.95,
            f"slope={coeffs[0]:.3f}\nR²={r_sq:.3f}",
            transform=axes[1].transAxes,
            va="top",
            fontsize=9,
            family="monospace",
        )
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Parameters")
        axes[1].set_ylabel(logprob_metric)
        axes[1].set_title("Log-probability (with extrapolation)")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Extrapolated Scaling — {task}", fontsize=14, y=1.02)
    fig.tight_layout()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"{task}_extrapolation.png",
            dpi=150,
            bbox_inches="tight",
        )
        fig.savefig(output_dir / f"{task}_extrapolation.pdf", bbox_inches="tight")

    plt.close(fig)
    return {
        "predicted_crossing_params": predicted_crossing,
        "predicted_logprob_at_target": predicted_logprob,
        "predicted_logprob_ci_lower": predicted_logprob_ci_lower,
        "predicted_logprob_ci_upper": predicted_logprob_ci_upper,
    }
