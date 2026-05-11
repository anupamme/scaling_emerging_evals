import json
import subprocess
import sys
from pathlib import Path

import pytest


def run_cli(*args: str, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sse.runners.run_eval", *args],
        capture_output=True,
        text=True,
        cwd=cwd or "/Users/mediratta/code/hiring/scaling_emerging_evals",
    )


class TestCLI:
    def test_produces_jsonl(self, tmp_path: Path):
        result = run_cli(
            "--model", "70m", "--task", "arithmetic_2digit",
            "--n", "3", "--output", str(tmp_path),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        jsonl_path = tmp_path / "arithmetic.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["task_name"] == "arithmetic"
        assert data["n_examples"] == 3
        assert "accuracy" in data["metrics"]
        assert "wall_clock_seconds" in data["config"]
        assert "peak_memory_mb" in data["config"]

    def test_appends_on_rerun(self, tmp_path: Path):
        args = [
            "--model", "70m", "--task", "arithmetic_2digit",
            "--n", "2", "--output", str(tmp_path),
        ]
        run_cli(*args)
        run_cli(*args)

        jsonl_path = tmp_path / "arithmetic.jsonl"
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 2

        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["run_id"] != r2["run_id"]

    def test_registry_second_task(self, tmp_path: Path):
        result = run_cli(
            "--model", "70m", "--task", "arithmetic_3digit",
            "--n", "2", "--output", str(tmp_path),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

        jsonl_path = tmp_path / "arithmetic.jsonl"
        assert jsonl_path.exists()
        data = json.loads(jsonl_path.read_text().strip().split("\n")[0])
        assert data["task_name"] == "arithmetic"
        assert data["config"]["n_digits"] == 3


class TestRegistry:
    def test_get_task_unknown(self):
        from sse.tasks import get_task

        with pytest.raises(ValueError, match="Unknown task"):
            get_task("nonexistent_task")

    def test_get_task_known(self):
        from sse.tasks import get_task

        task = get_task("arithmetic_2digit", n_examples=5, seed=1)
        examples = task.load_examples()
        assert len(examples) == 5
