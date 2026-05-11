import pytest

from sse.models.pythia import PythiaModel
from sse.results import EvalResult
from sse.tasks.base import Example, GenerationTask, MultipleChoiceTask


@pytest.fixture(scope="module")
def model():
    m = PythiaModel(size="70m", dtype="float32")
    yield m
    m.unload()


class DummyGenerationTask(GenerationTask):
    name = "dummy_generation"
    version = "v1"
    max_new_tokens = 10

    def load_examples(self, n: int | None = None) -> list[Example]:
        examples = [
            Example(id="1", prompt="2+2=", target="4"),
            Example(id="2", prompt="The color of the sky is", target="blue"),
            Example(id="3", prompt="Hello, my name is", target="Alice"),
        ]
        return examples[:n] if n is not None else examples


class DummyMultipleChoiceTask(MultipleChoiceTask):
    name = "dummy_mc"
    version = "v1"

    def load_examples(self, n: int | None = None) -> list[Example]:
        examples = [
            Example(
                id="1",
                prompt="The capital of France is",
                target=" Paris",
                choices=[" Paris", " London", " Berlin"],
            ),
            Example(
                id="2",
                prompt="Water is",
                target=" wet",
                choices=[" dry", " wet", " solid"],
            ),
            Example(
                id="3",
                prompt="The sun is a",
                target=" star",
                choices=[" planet", " star", " moon"],
            ),
        ]
        return examples[:n] if n is not None else examples


class TestGenerationTask:
    def test_produces_valid_result(self, model):
        task = DummyGenerationTask()
        examples = task.load_examples()
        result = task.evaluate(model, examples)

        assert isinstance(result, EvalResult)
        assert result.task_name == "dummy_generation"
        assert result.task_version == "v1"
        assert result.n_examples == 3
        assert "accuracy" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert len(result.per_example) == 3
        for ex in result.per_example:
            assert "output" in ex
            assert "correct" in ex


class TestMultipleChoiceTask:
    def test_produces_valid_result(self, model):
        task = DummyMultipleChoiceTask()
        examples = task.load_examples()
        result = task.evaluate(model, examples)

        assert isinstance(result, EvalResult)
        assert result.task_name == "dummy_mc"
        assert result.task_version == "v1"
        assert result.n_examples == 3
        assert "accuracy" in result.metrics
        assert "mean_logprob_correct" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0
        assert result.metrics["mean_logprob_correct"] < 0.0
        assert len(result.per_example) == 3
        for ex in result.per_example:
            assert "scores" in ex
            assert "predicted" in ex
            assert "correct" in ex
