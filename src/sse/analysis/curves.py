"""Scaling curve analysis — plotting, curve fitting, and emergence detection."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

    ax.scatter(x_params, y_values, zorder=5, s=60)

    if len(x_params) >= 3:
        log_x = np.log10(x_params)
        x_smooth = np.logspace(np.log10(x_params.min()), np.log10(x_params.max()), 100)
        log_x_smooth = np.log10(x_smooth)

        try:
            if _is_accuracy_metric(metric):
                p0 = [max(y_values.max(), 0.5), 2.0, np.median(log_x)]
                bounds = ([0, 0.01, log_x.min() - 2], [1.5, 20, log_x.max() + 2])
                popt, _ = curve_fit(_sigmoid, log_x, y_values, p0=p0, bounds=bounds, maxfev=5000)
                y_fit = _sigmoid(log_x_smooth, *popt)
            else:
                coeffs = np.polyfit(log_x, y_values, 1)
                y_fit = np.polyval(coeffs, log_x_smooth)

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
            "threshold_params": float("nan"), "ci_lower": float("nan"),
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
