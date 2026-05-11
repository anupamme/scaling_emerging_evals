"""CLI entry point for running evaluations."""

import argparse
import logging
import resource
import time
from pathlib import Path

from sse.models.pythia import PythiaModel
from sse.results import write_result
from sse.tasks import get_task

logger = logging.getLogger(__name__)


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(stdout_handler)

    file_handler = logging.FileHandler(output_dir / "debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    root.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an SSE evaluation")
    parser.add_argument("--model", required=True, help="Pythia model size (e.g. 70m, 1b)")
    parser.add_argument("--task", required=True, help="Task name from registry")
    parser.add_argument("--n", type=int, default=100, help="Number of examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--revision", default="main", help="Model revision")
    parser.add_argument("--dtype", default="float32", help="Model dtype")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    configure_logging(output_dir)

    logger.info(
        "Starting eval: model=%s task=%s n=%d seed=%d",
        args.model, args.task, args.n, args.seed,
    )

    start_time = time.time()

    logger.info(
        "Loading model pythia-%s (revision=%s, dtype=%s)",
        args.model, args.revision, args.dtype,
    )
    model = PythiaModel(size=args.model, revision=args.revision, dtype=args.dtype)

    task = get_task(args.task, n_examples=args.n, seed=args.seed)
    examples = task.load_examples()
    logger.info("Loaded %d examples for task %s", len(examples), args.task)

    result = task.evaluate(model, examples)

    wall_clock = time.time() - start_time
    peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    result.config["wall_clock_seconds"] = round(wall_clock, 2)
    result.config["peak_memory_mb"] = round(peak_memory_mb, 1)

    output_path = output_dir / f"{task.name}.jsonl"
    write_result(result, output_path)
    logger.info("Result written to %s", output_path)

    model.unload()

    logger.info(
        "Done. accuracy=%.3f wall_clock=%.1fs peak_mem=%.0fMB",
        result.metrics.get("accuracy", 0),
        wall_clock,
        peak_memory_mb,
    )


if __name__ == "__main__":
    main()
