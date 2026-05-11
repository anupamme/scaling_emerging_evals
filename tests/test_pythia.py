import pytest
import torch

from sse.models.pythia import PythiaModel


@pytest.fixture(scope="module")
def model():
    m = PythiaModel(size="70m", dtype="float32")
    yield m
    m.unload()


class TestValidation:
    def test_invalid_size(self):
        with pytest.raises(ValueError, match="Invalid size"):
            PythiaModel(size="50m")

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="Invalid dtype"):
            PythiaModel(size="70m", dtype="int8")


class TestGenerate:
    def test_generate_nonempty(self, model):
        output = model.generate("The capital of France is", max_new_tokens=20)
        assert isinstance(output, str)
        assert len(output) > 0


class TestLoglikelihood:
    def test_ranking(self, model):
        ll_paris = model.loglikelihood("The capital of France is", " Paris")
        ll_tokyo = model.loglikelihood("The capital of France is", " Tokyo")
        assert ll_paris > ll_tokyo

    def test_batch_consistency(self, model):
        pairs = [
            ("The capital of France is", " Paris"),
            ("The capital of Japan is", " Tokyo"),
            ("Water freezes at", " zero degrees"),
            ("The sun is a", " star"),
            ("Python is a", " programming language"),
        ]
        individual = [model.loglikelihood(ctx, cont) for ctx, cont in pairs]
        batched = model.loglikelihood_batch(pairs, batch_size=8)

        for i, (ind, bat) in enumerate(zip(individual, batched)):
            assert abs(ind - bat) < 1e-4, f"Pair {i}: individual={ind}, batched={bat}"


class TestMemory:
    def test_unload_reduces_memory(self):
        m = PythiaModel(size="70m")

        if m.device.type == "mps":
            mem_before = torch.mps.current_allocated_memory()
            m.unload()
            mem_after = torch.mps.current_allocated_memory()
            assert mem_after < mem_before
        else:
            m.unload()
            assert m.model is None
            assert m.tokenizer is None
