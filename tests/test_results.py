from pathlib import Path

from sse.results import EvalResult, load_all_results, load_results, write_result


def _make_result(run_id: str = "abc-123", metrics: dict | None = None) -> EvalResult:
    return EvalResult(
        run_id=run_id,
        timestamp="2025-05-11T12:00:00Z",
        model_size="70m",
        model_revision="main",
        task_name="arithmetic_2digit",
        task_version="v1",
        n_examples=100,
        metrics=metrics or {"accuracy": 0.42},
        per_example=[{"input": "2+2", "target": "4", "output": "4", "correct": True}],
        config={"batch_size": 8},
    )


def test_round_trip(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    original = _make_result()
    write_result(original, path)
    loaded = load_results(path)
    assert len(loaded) == 1
    assert loaded[0] == original


def test_load_all_results_dataframe(tmp_path: Path):
    result1 = _make_result("run-1", {"accuracy": 0.5, "mean_logprob": -2.0})
    write_result(result1, tmp_path / "a.jsonl")
    write_result(_make_result("run-2", {"accuracy": 0.6}), tmp_path / "a.jsonl")
    write_result(_make_result("run-3", {"accuracy": 0.7, "f1": 0.8}), tmp_path / "b.jsonl")

    df = load_all_results(tmp_path)

    # run-1 has 2 metrics, run-2 has 1, run-3 has 2 => 5 rows total
    assert len(df) == 5
    assert set(df.columns) == {
        "run_id",
        "timestamp",
        "model_size",
        "model_revision",
        "task_name",
        "task_version",
        "n_examples",
        "schema_version",
        "metric_name",
        "metric_value",
    }
    assert set(df["run_id"]) == {"run-1", "run-2", "run-3"}
    assert set(df["metric_name"]) == {"accuracy", "mean_logprob", "f1"}


def test_deduplication(tmp_path: Path):
    for i in range(3):
        write_result(_make_result(f"run-{i}", {"accuracy": 0.5 + i * 0.01}), tmp_path / "t.jsonl")
    df = load_all_results(tmp_path, deduplicate=True)
    assert len(df) == 1
    assert df.iloc[0]["metric_value"] == 0.52


def test_no_dedup_by_default(tmp_path: Path):
    for i in range(3):
        write_result(_make_result(f"run-{i}"), tmp_path / "t.jsonl")
    df = load_all_results(tmp_path)
    assert len(df) == 3
