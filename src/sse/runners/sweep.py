"""Sweep orchestration — run all (model, task) combinations from a YAML config."""

import argparse
import logging
import resource
import time
from pathlib import Path

import yaml

from sse.models.pythia import PythiaModel
from sse.results import EvalResult, load_results, write_result
from sse.tasks import get_task

logger = logging.getLogger(__name__)

TOKENS_PER_EXAMPLE_ESTIMATE = 50


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_completed_pairs(output_dir: Path) -> set[tuple[str, str]]:
    completed = set()
    if not output_dir.exists():
        return completed
    for path in output_dir.glob("*.jsonl"):
        try:
            results = load_results(path)
            for r in results:
                task_key = r.config.get("registry_task_name", r.task_name)
                completed.add((r.model_size, task_key))
        except Exception:
            continue
    return completed


def estimate_token_budget(tasks_config: list[dict], models: list[str]) -> int:
    total_examples = sum(t.get("n_examples", 100) for t in tasks_config)
    return total_examples * TOKENS_PER_EXAMPLE_ESTIMATE * len(models)


def run_sweep(
    config: dict,
    models_override: list[str] | None = None,
    dry_run: bool = False,
    resume: bool = False,
    output_override: str | None = None,
) -> list[EvalResult]:
    models = models_override or config["models"]
    tasks_config = config["tasks"]
    seed = config.get("seed", 42)
    dtype = config.get("dtype", "float32")
    revision = config.get("revision", "main")
    output_dir = Path(output_override or config.get("output", "results/"))

    token_budget = estimate_token_budget(tasks_config, models)
    logger.info("Estimated token budget: %d tokens", token_budget)
    print(f"Estimated token budget: {token_budget:,} tokens")

    completed = get_completed_pairs(output_dir) if resume else set()
    if completed:
        logger.info("Found %d completed (model, task) pairs", len(completed))

    pairs = []
    for model_size in models:
        for task_cfg in tasks_config:
            task_name = task_cfg["name"]
            if (model_size, task_name) in completed:
                status = "SKIP"
            else:
                status = "RUN"
            pairs.append((model_size, task_name, task_cfg.get("n_examples", 100), status))

    print(f"\nSweep matrix ({len(models)} models × {len(tasks_config)} tasks):")
    for model_size, task_name, n, status in pairs:
        print(f"  [{status}] model={model_size} task={task_name} n={n}")

    if dry_run:
        print("\n--dry-run: exiting without running.")
        return []

    all_results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_size in models:
        model_tasks = [
            (name, n) for m, name, n, status in pairs
            if m == model_size and status == "RUN"
        ]
        if not model_tasks:
            logger.info("Skipping model %s (all tasks completed)", model_size)
            continue

        logger.info("Loading model pythia-%s", model_size)
        model = PythiaModel(size=model_size, revision=revision, dtype=dtype)

        for task_name, n_examples in model_tasks:
            logger.info("Running task %s (n=%d) on model %s", task_name, n_examples, model_size)
            start = time.time()

            task = get_task(task_name, n_examples=n_examples, seed=seed)
            examples = task.load_examples()
            result = task.evaluate(model, examples)

            wall_clock = time.time() - start
            peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            result.config["wall_clock_seconds"] = round(wall_clock, 2)
            result.config["peak_memory_mb"] = round(peak_mem, 1)
            result.config["registry_task_name"] = task_name

            output_path = output_dir / f"{task_name}.jsonl"
            write_result(result, output_path)
            all_results.append(result)

            logger.info(
                "Done: %s/%s accuracy=%.3f (%.1fs)",
                model_size, task_name, result.metrics.get("accuracy", 0), wall_clock,
            )

        model.unload()
        logger.info("Unloaded model %s", model_size)

    print(f"\nSweep complete: {len(all_results)} results written to {output_dir}")
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sweep of (model, task) evaluations")
    parser.add_argument(
        "--config", default="configs/main_sweep.yaml", help="Path to sweep YAML config"
    )
    parser.add_argument("--models", help="Override model list (comma-separated)")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    parser.add_argument("--resume", action="store_true", help="Skip completed pairs")
    parser.add_argument("--output", help="Override output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(Path(args.config))
    models_override = args.models.split(",") if args.models else None

    run_sweep(
        config=config,
        models_override=models_override,
        dry_run=args.dry_run,
        resume=args.resume,
        output_override=args.output,
    )


if __name__ == "__main__":
    main()
