"""Regenerate all figures from results/ directory."""

import sys
from pathlib import Path

from sse.analysis import (
    plot_emergence_comparison,
    plot_emergence_with_extrapolation,
    plot_scaling_curve,
)
from sse.results import load_all_results


def _find_logprob_metric(metrics):
    for m in metrics:
        if "logprob" in m:
            return m
    return None


def main(results_dir: Path | None = None) -> None:
    results_dir = results_dir or Path("results")

    if not results_dir.exists() or not list(results_dir.glob("*.jsonl")):
        print(f"No results found in {results_dir}/ — run a sweep first.")
        return

    df = load_all_results(results_dir, deduplicate=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    tasks = df["task_name"].unique().tolist()
    print(f"Generating figures for {len(tasks)} task(s): {tasks}")

    for task in tasks:
        metrics = df[df["task_name"] == task]["metric_name"].unique()

        for metric in metrics:
            plot_scaling_curve(df, task, metric, output_dir=figures_dir)

        logprob_metric = _find_logprob_metric(metrics)
        if "accuracy" in metrics and logprob_metric:
            plot_emergence_comparison(
                df,
                task,
                logprob_metric=logprob_metric,
                output_dir=figures_dir,
            )
            plot_emergence_with_extrapolation(
                df,
                task,
                logprob_metric=logprob_metric,
                output_dir=figures_dir,
            )

    generated = list(figures_dir.glob("*"))
    print(f"Done: {len(generated)} files written to {figures_dir}/")


if __name__ == "__main__":
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(results_path)
