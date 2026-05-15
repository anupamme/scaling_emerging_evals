"""In-context learning task — measures few-shot vs zero-shot delta on arithmetic."""

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


class ICLArithmeticTask(GenerationTask):
    """Arithmetic with in-context demonstrations.

    Same eval examples regardless of n_shots (IDs are stable).
    Demonstrations come from a separate RNG stream to avoid overlap.
    """

    name = "icl_arithmetic"
    version = "v2"
    max_new_tokens = 10

    def __init__(
        self,
        n_digits: int = 2,
        operation: Literal["add", "subtract"] = "add",
        n_examples: int = 100,
        n_shots: int = 0,
        seed: int = 42,
        prompt_format: Literal["equation", "qa"] = "equation",
    ):
        self.n_digits = n_digits
        self.operation = operation
        self.n_examples = n_examples
        self.n_shots = n_shots
        self.seed = seed
        self.prompt_format = prompt_format

    def _generate_pairs(self, rng: random.Random, count: int) -> list[tuple[int, int, int]]:
        lo = 10 ** (self.n_digits - 1)
        hi = 10**self.n_digits - 1
        pairs = []
        for _ in range(count):
            a = rng.randint(lo, hi)
            b = rng.randint(lo, hi)
            answer = a + b if self.operation == "add" else a - b
            pairs.append((a, b, answer))
        return pairs

    def _format_qa(self, a: int, b: int, answer: int) -> str:
        op_symbol = "+" if self.operation == "add" else "-"
        if self.prompt_format == "equation":
            return f"{a} {op_symbol} {b} = {answer}"
        return f"Q: What is {a} {op_symbol} {b}?\nA: {answer}"

    def load_examples(self, n: int | None = None) -> list[Example]:
        rng = random.Random(self.seed)
        count = n if n is not None else self.n_examples
        op_symbol = "+" if self.operation == "add" else "-"

        pairs = self._generate_pairs(rng, count)
        examples = []
        for i, (a, b, answer) in enumerate(pairs):
            if self.prompt_format == "equation":
                prompt = f"{a} {op_symbol} {b} ="
            else:
                prompt = f"Q: What is {a} {op_symbol} {b}?\nA:"
            examples.append(Example(
                id=str(i),
                prompt=prompt,
                target=f" {answer}",
                metadata={"a": a, "b": b, "operation": self.operation},
            ))
        return examples

    def _get_demonstrations(self) -> list[str]:
        demo_rng = random.Random(self.seed + 1000)
        pairs = self._generate_pairs(demo_rng, self.n_shots)
        return [self._format_qa(a, b, ans) for a, b, ans in pairs]

    def _build_prompt(self, example_prompt: str, demos: list[str]) -> str:
        if not demos:
            return example_prompt
        return "\n\n".join(demos) + "\n\n" + example_prompt

    def match(self, output: str, target: str) -> bool:
        m = re.search(r"-?\d+", output)
        if m is None:
            return False
        return int(m.group()) == int(target.strip())

    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult:
        demos = self._get_demonstrations()
        per_example = []
        correct = 0
        logprobs = []

        for ex in examples:
            full_prompt = self._build_prompt(ex.prompt, demos)
            output = model.generate(full_prompt, max_new_tokens=self.max_new_tokens)
            is_correct = self.match(output, ex.target)
            correct += int(is_correct)

            ll = model.loglikelihood(full_prompt, ex.target)
            logprobs.append(ll)

            per_example.append({
                "id": ex.id,
                "prompt": full_prompt,
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
                "n_shots": self.n_shots,
                "seed": self.seed,
                "max_new_tokens": self.max_new_tokens,
            },
        )


def run_icl_sweep(
    model: PythiaModel,
    n_shots_list: list[int] | None = None,
    n_digits: int = 2,
    operation: Literal["add", "subtract"] = "add",
    n_examples: int = 100,
    seed: int = 42,
) -> list[EvalResult]:
    """Run ICL arithmetic at multiple shot counts, returning one result per count."""
    if n_shots_list is None:
        n_shots_list = [0, 1, 4]

    results = []
    for n_shots in n_shots_list:
        task = ICLArithmeticTask(
            n_digits=n_digits,
            operation=operation,
            n_examples=n_examples,
            n_shots=n_shots,
            seed=seed,
        )
        examples = task.load_examples()
        result = task.evaluate(model, examples)
        results.append(result)
    return results
