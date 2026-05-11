import math

import pytest

from sse.models.pythia import PythiaModel
from sse.tasks.multiple_choice import TruthfulQATask


@pytest.fixture(scope="module")
def model_1b():
    m = PythiaModel(size="1b", dtype="float32")
    yield m
    m.unload()


class TestReproducibility:
    def test_same_seed_same_examples(self):
        task1 = TruthfulQATask(n_examples=10, seed=123)
        task2 = TruthfulQATask(n_examples=10, seed=123)
        assert task1.load_examples() == task2.load_examples()

    def test_different_seed_different_examples(self):
        task1 = TruthfulQATask(n_examples=10, seed=1)
        task2 = TruthfulQATask(n_examples=10, seed=2)
        assert task1.load_examples() != task2.load_examples()


class TestTruthfulQA1b:
    def test_end_to_end(self, model_1b):
        task = TruthfulQATask(n_examples=200, seed=42)
        examples = task.load_examples()
        result = task.evaluate(model_1b, examples)

        assert result.task_name == "truthfulqa_mc1"
        assert result.n_examples == 200

        assert "accuracy" in result.metrics
        assert "mean_logprob_correct" in result.metrics
        assert "mean_logprob_margin" in result.metrics

        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert math.isfinite(result.metrics["mean_logprob_correct"])
        assert result.metrics["mean_logprob_correct"] < 0.0
        assert math.isfinite(result.metrics["mean_logprob_margin"])

        assert len(result.per_example) == 200
