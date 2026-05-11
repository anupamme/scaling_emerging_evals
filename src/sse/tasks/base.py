from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel

from sse.results import EvalResult

if TYPE_CHECKING:
    from sse.models.pythia import PythiaModel


class Example(BaseModel):
    id: str
    prompt: str
    target: str
    choices: list[str] | None = None
    metadata: dict = {}


class Task(ABC):
    name: str
    version: str

    @abstractmethod
    def load_examples(self, n: int | None = None) -> list[Example]: ...

    @abstractmethod
    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult: ...


class GenerationTask(Task):
    max_new_tokens: int = 32

    def match(self, output: str, target: str) -> bool:
        return output.strip().lower() == target.strip().lower()

    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult:
        per_example = []
        correct = 0

        for ex in examples:
            output = model.generate(ex.prompt, max_new_tokens=self.max_new_tokens)
            is_correct = self.match(output, ex.target)
            correct += int(is_correct)
            per_example.append({
                "id": ex.id,
                "prompt": ex.prompt,
                "target": ex.target,
                "output": output,
                "correct": is_correct,
            })

        accuracy = correct / len(examples) if examples else 0.0

        return EvalResult(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_size=model.size,
            model_revision=model.revision,
            task_name=self.name,
            task_version=self.version,
            n_examples=len(examples),
            metrics={"accuracy": accuracy},
            per_example=per_example,
            config={"max_new_tokens": self.max_new_tokens},
        )


class MultipleChoiceTask(Task):
    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult:
        per_example = []
        correct = 0
        logprobs_correct = []

        for ex in examples:
            assert ex.choices is not None, f"Example {ex.id} has no choices"
            scores = [model.loglikelihood(ex.prompt, choice) for choice in ex.choices]
            predicted_idx = max(range(len(scores)), key=lambda i: scores[i])
            predicted = ex.choices[predicted_idx]
            is_correct = predicted == ex.target
            correct += int(is_correct)

            correct_idx = ex.choices.index(ex.target)
            logprobs_correct.append(scores[correct_idx])

            per_example.append({
                "id": ex.id,
                "prompt": ex.prompt,
                "target": ex.target,
                "choices": ex.choices,
                "scores": scores,
                "predicted": predicted,
                "correct": is_correct,
            })

        accuracy = correct / len(examples) if examples else 0.0
        mean_logprob = sum(logprobs_correct) / len(logprobs_correct) if logprobs_correct else 0.0

        return EvalResult(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_size=model.size,
            model_revision=model.revision,
            task_name=self.name,
            task_version=self.version,
            n_examples=len(examples),
            metrics={"accuracy": accuracy, "mean_logprob_correct": mean_logprob},
            per_example=per_example,
            config={},
        )
