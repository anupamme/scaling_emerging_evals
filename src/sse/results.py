from pathlib import Path

import pandas as pd
from pydantic import BaseModel


class EvalResult(BaseModel):
    run_id: str
    timestamp: str
    model_size: str
    model_revision: str
    task_name: str
    task_version: str
    n_examples: int
    metrics: dict[str, float]
    per_example: list[dict]
    config: dict
    schema_version: str = "1.0"


def write_result(result: EvalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(result.model_dump_json() + "\n")


def load_results(path: Path) -> list[EvalResult]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(EvalResult.model_validate_json(line))
    return results


def load_all_results(results_dir: Path) -> pd.DataFrame:
    all_results = []
    for path in sorted(results_dir.glob("*.jsonl")):
        all_results.extend(load_results(path))

    rows = []
    for result in all_results:
        base = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "model_size": result.model_size,
            "model_revision": result.model_revision,
            "task_name": result.task_name,
            "task_version": result.task_version,
            "n_examples": result.n_examples,
            "schema_version": result.schema_version,
        }
        for metric_name, metric_value in result.metrics.items():
            rows.append({**base, "metric_name": metric_name, "metric_value": metric_value})

    return pd.DataFrame(rows)
