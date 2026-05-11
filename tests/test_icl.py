import math

import pytest

from sse.analysis.icl import compute_icl_delta
from sse.models.pythia import PythiaModel
from sse.tasks.icl import ICLArithmeticTask, run_icl_sweep


@pytest.fixture(scope="module")
def model():
    m = PythiaModel(size="70m", dtype="float32")
    yield m
    m.unload()


class TestExampleStability:
    def test_same_ids_across_shots(self):
        examples_0 = ICLArithmeticTask(n_shots=0, seed=42).load_examples(10)
        examples_1 = ICLArithmeticTask(n_shots=1, seed=42).load_examples(10)
        examples_4 = ICLArithmeticTask(n_shots=4, seed=42).load_examples(10)

        ids_0 = [e.id for e in examples_0]
        ids_1 = [e.id for e in examples_1]
        ids_4 = [e.id for e in examples_4]
        assert ids_0 == ids_1 == ids_4

        targets_0 = [e.target for e in examples_0]
        targets_1 = [e.target for e in examples_1]
        targets_4 = [e.target for e in examples_4]
        assert targets_0 == targets_1 == targets_4

    def test_reproducibility(self):
        t1 = ICLArithmeticTask(n_shots=4, seed=99).load_examples(5)
        t2 = ICLArithmeticTask(n_shots=4, seed=99).load_examples(5)
        assert t1 == t2


class TestSingleRun:
    def test_zero_shot_produces_result(self, model):
        task = ICLArithmeticTask(n_examples=5, n_shots=0, seed=42)
        examples = task.load_examples()
        result = task.evaluate(model, examples)

        assert result.task_name == "icl_arithmetic"
        assert result.n_examples == 5
        assert "accuracy" in result.metrics
        assert "mean_logprob_correct" in result.metrics
        assert result.config["n_shots"] == 0
        assert math.isfinite(result.metrics["mean_logprob_correct"])


class TestSweep:
    def test_produces_three_results(self, model):
        results = run_icl_sweep(model, n_shots_list=[0, 1, 4], n_examples=5, seed=42)

        assert len(results) == 3
        assert results[0].config["n_shots"] == 0
        assert results[1].config["n_shots"] == 1
        assert results[2].config["n_shots"] == 4

        for r in results:
            assert r.task_name == "icl_arithmetic"
            assert r.n_examples == 5
            assert "accuracy" in r.metrics

    def test_identical_example_ids(self, model):
        results = run_icl_sweep(model, n_shots_list=[0, 1, 4], n_examples=5, seed=42)

        ids_0 = [e["id"] for e in results[0].per_example]
        ids_1 = [e["id"] for e in results[1].per_example]
        ids_4 = [e["id"] for e in results[2].per_example]
        assert ids_0 == ids_1 == ids_4


class TestAnalysis:
    def test_compute_icl_delta(self, model):
        results = run_icl_sweep(model, n_shots_list=[0, 1, 4], n_examples=5, seed=42)
        delta = compute_icl_delta(results)

        assert "icl_delta_accuracy" in delta
        assert "accuracy_0shot" in delta
        assert "accuracy_4shot" in delta
        assert "accuracy_1shot" in delta
        assert math.isfinite(delta["icl_delta_accuracy"])
        assert delta["icl_delta_accuracy"] == delta["accuracy_4shot"] - delta["accuracy_0shot"]
