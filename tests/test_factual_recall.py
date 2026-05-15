import math

import pytest

from sse.models.pythia import PythiaModel
from sse.tasks.factual_recall import FactualRecallTask


@pytest.fixture(scope="module")
def model_410m():
    m = PythiaModel(size="410m", dtype="float32")
    yield m
    m.unload()


class TestReproducibility:
    def test_same_seed_same_examples(self):
        task1 = FactualRecallTask(n_examples=10, seed=123)
        task2 = FactualRecallTask(n_examples=10, seed=123)
        assert task1.load_examples() == task2.load_examples()

    def test_different_seed_different_examples(self):
        task1 = FactualRecallTask(n_examples=10, seed=1)
        task2 = FactualRecallTask(n_examples=10, seed=2)
        assert task1.load_examples() != task2.load_examples()


class TestSubstringMatching:
    def setup_method(self):
        self.task = FactualRecallTask(n_examples=5, seed=42)

    def test_exact_match(self):
        assert self.task._match_any_alias("Paris", ["Paris"])

    def test_substring_match(self):
        assert self.task._match_any_alias("Paris, France", ["Paris"])

    def test_substring_normalized(self):
        assert self.task._match_any_alias("Paris, the capital of France", ["Paris"])

    def test_no_match_when_absent(self):
        assert not self.task._match_any_alias("London is great", ["Paris"])

    def test_multiple_aliases(self):
        assert self.task._match_any_alias("The Big Apple is NYC", ["New York City", "NYC"])

    def test_empty_alias_no_match(self):
        assert not self.task._match_any_alias("anything", [""])


class TestFactualRecall410m:
    def test_end_to_end(self, model_410m):
        task = FactualRecallTask(n_examples=100, seed=42)
        examples = task.load_examples()
        result = task.evaluate(model_410m, examples)

        assert result.task_name == "factual_recall"
        assert result.n_examples == 100
        assert "accuracy" in result.metrics
        assert "mean_logprob_first_alias" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert math.isfinite(result.metrics["mean_logprob_first_alias"])
        assert result.metrics["mean_logprob_first_alias"] < 0.0
        assert len(result.per_example) == 100
