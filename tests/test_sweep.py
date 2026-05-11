import json
import subprocess
import sys
from pathlib import Path

import yaml

from sse.runners.sweep import get_completed_pairs, load_config


def _make_config(tmp_path: Path, models: list[str], tasks: list[dict]) -> Path:
    config = {
        "models": models,
        "tasks": tasks,
        "seed": 42,
        "dtype": "float32",
        "revision": "main",
        "output": str(tmp_path / "results"),
    }
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


def _run_sweep_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "sse.runners.sweep", *args],
        capture_output=True,
        text=True,
        cwd="/Users/mediratta/code/hiring/scaling_emerging_evals",
    )


class TestDryRun:
    def test_prints_matrix(self, tmp_path: Path):
        config_path = _make_config(
            tmp_path,
            models=["70m", "160m"],
            tasks=[
                {"name": "arithmetic_2digit", "n_examples": 5},
                {"name": "arithmetic_3digit", "n_examples": 5},
            ],
        )
        result = _run_sweep_cli("--config", str(config_path), "--dry-run")
        assert result.returncode == 0, f"stderr: {result.stderr}"

        out = result.stdout
        assert "70m" in out
        assert "160m" in out
        assert "arithmetic_2digit" in out
        assert "arithmetic_3digit" in out
        assert "dry-run" in out.lower()
        assert "[RUN]" in out


class TestRealRun:
    def test_writes_4_results(self, tmp_path: Path):
        config_path = _make_config(
            tmp_path,
            models=["70m", "160m"],
            tasks=[
                {"name": "arithmetic_2digit", "n_examples": 3},
                {"name": "arithmetic_3digit", "n_examples": 3},
            ],
        )
        result = _run_sweep_cli("--config", str(config_path))
        assert result.returncode == 0, f"stderr: {result.stderr}"

        results_dir = tmp_path / "results"
        all_lines = []
        for jsonl in results_dir.glob("*.jsonl"):
            lines = jsonl.read_text().strip().split("\n")
            all_lines.extend(lines)

        assert len(all_lines) == 4

        models_seen = set()
        tasks_seen = set()
        for line in all_lines:
            data = json.loads(line)
            models_seen.add(data["model_size"])
            tasks_seen.add(data["task_name"])

        assert models_seen == {"70m", "160m"}
        assert tasks_seen == {"arithmetic"}


class TestResume:
    def test_skips_completed(self, tmp_path: Path):
        config_path = _make_config(
            tmp_path,
            models=["70m"],
            tasks=[
                {"name": "arithmetic_2digit", "n_examples": 3},
                {"name": "arithmetic_3digit", "n_examples": 3},
            ],
        )

        # First run
        r1 = _run_sweep_cli("--config", str(config_path))
        assert r1.returncode == 0, f"stderr: {r1.stderr}"

        results_dir = tmp_path / "results"
        lines_after_first = 0
        for jsonl in results_dir.glob("*.jsonl"):
            lines_after_first += len(jsonl.read_text().strip().split("\n"))
        assert lines_after_first == 2

        # Second run with --resume
        r2 = _run_sweep_cli("--config", str(config_path), "--resume")
        assert r2.returncode == 0, f"stderr: {r2.stderr}"
        assert "SKIP" in r2.stdout

        lines_after_second = 0
        for jsonl in results_dir.glob("*.jsonl"):
            lines_after_second += len(jsonl.read_text().strip().split("\n"))
        assert lines_after_second == 2  # No new rows


class TestHelpers:
    def test_load_config(self, tmp_path: Path):
        config_path = _make_config(tmp_path, ["70m"], [{"name": "test", "n_examples": 10}])
        config = load_config(config_path)
        assert config["models"] == ["70m"]
        assert config["tasks"][0]["name"] == "test"

    def test_get_completed_pairs_empty(self, tmp_path: Path):
        assert get_completed_pairs(tmp_path) == set()

    def test_models_override(self, tmp_path: Path):
        config_path = _make_config(
            tmp_path,
            models=["70m", "160m", "410m"],
            tasks=[{"name": "arithmetic_2digit", "n_examples": 3}],
        )
        result = _run_sweep_cli(
            "--config", str(config_path), "--models", "70m", "--dry-run"
        )
        assert result.returncode == 0
        assert "70m" in result.stdout
        assert "160m" not in result.stdout
