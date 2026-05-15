import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestWriteupExists:
    def test_writeup_file_exists(self):
        assert (PROJECT_ROOT / "WRITEUP.md").exists()

    def test_writeup_has_all_sections(self):
        content = (PROJECT_ROOT / "WRITEUP.md").read_text()
        required_sections = [
            "Abstract",
            "Setup",
            "Per-Task Scaling Curves",
            "Discrete vs Continuous Metric Comparison",
            "Limitations",
            "Reproduction Instructions",
            "References",
        ]
        for section in required_sections:
            assert section in content, f"Missing section: {section}"


class TestGenerateFigures:
    def test_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "generate_figures.py").exists()

    def test_runs_with_empty_results(self, tmp_path: Path):
        result = subprocess.run(
            [sys.executable, "scripts/generate_figures.py", str(tmp_path)],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "No results found" in result.stdout


class TestMakefileTargets:
    def test_has_figures_target(self):
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "figures:" in content

    def test_has_writeup_target(self):
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "writeup:" in content
