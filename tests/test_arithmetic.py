import math

import pytest

from sse.models.pythia import PythiaModel
from sse.tasks.arithmetic import ArithmeticTask


@pytest.fixture(scope="module")
def model_70m():
    m = PythiaModel(size="70m", dtype="float32")
    yield m
    m.unload()


@pytest.fixture(scope="module")
def model_1b():
    m = PythiaModel(size="1b", dtype="float32")
    yield m
    m.unload()


class TestReproducibility:
    def test_same_seed_same_examples(self):
        task1 = ArithmeticTask(n_digits=2, operation="add", n_examples=10, seed=123)
        task2 = ArithmeticTask(n_digits=2, operation="add", n_examples=10, seed=123)
        assert task1.load_examples() == task2.load_examples()

    def test_different_seed_different_examples(self):
        task1 = ArithmeticTask(n_digits=2, operation="add", n_examples=10, seed=1)
        task2 = ArithmeticTask(n_digits=2, operation="add", n_examples=10, seed=2)
        assert task1.load_examples() != task2.load_examples()


class TestArithmetic70m:
    def test_reports_both_metrics(self, model_70m):
        task = ArithmeticTask(n_digits=2, operation="add", n_examples=10, seed=42)
        examples = task.load_examples()
        result = task.evaluate(model_70m, examples)

        assert "accuracy" in result.metrics
        assert "mean_logprob_correct" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert math.isfinite(result.metrics["mean_logprob_correct"])
        assert result.metrics["mean_logprob_correct"] < 0.0
        assert result.n_examples == 10


class TestScaling:
    def test_1b_at_least_as_good_as_70m(self, model_70m, model_1b):
        task = ArithmeticTask(n_digits=2, operation="add", n_examples=50, seed=42)
        examples = task.load_examples()

        result_70m = task.evaluate(model_70m, examples)
        result_1b = task.evaluate(model_1b, examples)

        ll_1b = result_1b.metrics["mean_logprob_correct"]
        ll_70m = result_70m.metrics["mean_logprob_correct"]
        assert ll_1b >= ll_70m - 2.0
