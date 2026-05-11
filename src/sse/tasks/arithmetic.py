"""Arithmetic eval task — synthetic, deterministic, known to show sharp emergence."""

from __future__ import annotations

import random
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from sse.results import EvalResult
from sse.tasks.base import Example, GenerationTask

if TYPE_CHECKING:
    from sse.models.pythia import PythiaModel


class ArithmeticTask(GenerationTask):
    """Synthetic arithmetic task.

    Prompt template: "Q: What is {a} {op} {b}?\nA:"
    Target: " {answer}" (leading space for proper tokenization)
    """

    name = "arithmetic"
    version = "v1"
    max_new_tokens = 10

    def __init__(
        self,
        n_digits: int = 2,
        operation: Literal["add", "subtract"] = "add",
        n_examples: int = 100,
        seed: int = 42,
    ):
        self.n_digits = n_digits
        self.operation = operation
        self.n_examples = n_examples
        self.seed = seed

    def load_examples(self, n: int | None = None) -> list[Example]:
        rng = random.Random(self.seed)
        lo = 10 ** (self.n_digits - 1)
        hi = 10**self.n_digits - 1
        count = n if n is not None else self.n_examples

        op_symbol = "+" if self.operation == "add" else "-"
        examples = []
        for i in range(count):
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            if self.operation == "add":
                answer = a + b
            else:
                answer = a - b
            prompt = f"Q: What is {a} {op_symbol} {b}?\nA:"
            examples.append(Example(
                id=str(i),
                prompt=prompt,
                target=f" {answer}",
                metadata={"a": a, "b": b, "operation": self.operation},
            ))
        return examples

    def match(self, output: str, target: str) -> bool:
        match = re.search(r"-?\d+", output)
        if match is None:
            return False
        return int(match.group()) == int(target.strip())

    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult:
        per_example = []
        correct = 0
        logprobs = []

        for ex in examples:
            output = model.generate(ex.prompt, max_new_tokens=self.max_new_tokens)
            is_correct = self.match(output, ex.target)
            correct += int(is_correct)

            ll = model.loglikelihood(ex.prompt, ex.target)
            logprobs.append(ll)

            per_example.append({
                "id": ex.id,
                "prompt": ex.prompt,
                "target": ex.target,
                "output": output,
                "correct": is_correct,
                "logprob": ll,
            })

        accuracy = correct / len(examples) if examples else 0.0
        mean_logprob = sum(logprobs) / len(logprobs) if logprobs else 0.0

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
            config={
                "n_digits": self.n_digits,
                "operation": self.operation,
                "seed": self.seed,
                "max_new_tokens": self.max_new_tokens,
            },
        )
