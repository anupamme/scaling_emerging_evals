from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from sse.analysis.curves import (
    fit_emergence_threshold,
    plot_emergence_comparison,
    plot_scaling_curve,
)


def _make_synthetic_df() -> pd.DataFrame:
    """Create synthetic results mimicking sigmoid emergence for accuracy
    and linear improvement for logprob across 6 model sizes."""
    sizes = ["70m", "160m", "410m", "1b", "1.4b", "2.8b"]
    params = [70e6, 160e6, 410e6, 1e9, 1.4e9, 2.8e9]
    log_params = np.log10(params)

    # Sigmoid accuracy: near 0 for small, rising to ~0.8 for large
    accuracy = 0.85 / (1 + np.exp(-3.0 * (log_params - 8.8)))
    # Linear logprob: improves (less negative) with scale
    logprob = -12 + 1.5 * (log_params - 7.8)

    rows = []
    for i, size in enumerate(sizes):
        base = {
            "run_id": f"run-{i}",
            "timestamp": "2025-01-01T00:00:00Z",
            "model_size": size,
            "model_revision": "main",
            "task_name": "arithmetic",
            "task_version": "v1",
            "n_examples": 200,
            "schema_version": "1.0",
        }
        rows.append({**base, "metric_name": "accuracy", "metric_value": float(accuracy[i])})
        rows.append({
            **base, "metric_name": "mean_logprob_correct", "metric_value": float(logprob[i])
        })

    return pd.DataFrame(rows)


class TestPlotScalingCurve:
    def test_returns_axes(self):
        df = _make_synthetic_df()
        ax = plot_scaling_curve(df, "arithmetic", "accuracy")
        assert ax is not None
        assert ax.get_xscale() == "log"

    def test_saves_files(self, tmp_path: Path):
        df = _make_synthetic_df()
        plot_scaling_curve(df, "arithmetic", "accuracy", output_dir=tmp_path)
        assert (tmp_path / "arithmetic_accuracy.png").exists()
        assert (tmp_path / "arithmetic_accuracy.pdf").exists()

    def test_logprob_metric(self, tmp_path: Path):
        df = _make_synthetic_df()
        ax = plot_scaling_curve(df, "arithmetic", "mean_logprob_correct", output_dir=tmp_path)
        assert ax is not None
        assert (tmp_path / "arithmetic_mean_logprob_correct.png").exists()


class TestPlotEmergenceComparison:
    def test_returns_figure_with_two_panels(self, tmp_path: Path):
        df = _make_synthetic_df()
        fig = plot_emergence_comparison(df, "arithmetic", output_dir=tmp_path)
        assert fig is not None
        assert len(fig.axes) == 2
        assert (tmp_path / "arithmetic_emergence.png").exists()
        assert (tmp_path / "arithmetic_emergence.pdf").exists()


class TestFitEmergenceThreshold:
    def test_returns_threshold_with_ci(self):
        df = _make_synthetic_df()
        result = fit_emergence_threshold(df, "arithmetic", threshold=0.4, n_bootstrap=200)
        assert "threshold_params" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert np.isfinite(result["threshold_params"])
        assert result["ci_lower"] <= result["threshold_params"] <= result["ci_upper"]
