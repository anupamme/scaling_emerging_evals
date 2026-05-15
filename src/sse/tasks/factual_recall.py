"""Factual recall task using TriviaQA (closed-book, no context)."""

from __future__ import annotations

import random
import re
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import datasets

from sse.results import EvalResult
from sse.tasks.base import Example, GenerationTask

if TYPE_CHECKING:
    from sse.models.pythia import PythiaModel


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


class FactualRecallTask(GenerationTask):
    """Closed-book factual recall using TriviaQA (rc.nocontext).

    Prompt template: "Question: {question}\\nAnswer:"
    Target: " {first_alias}" (leading space for proper tokenization)
    Match: normalized output compared against all answer aliases.
    """

    name = "factual_recall"
    version = "v1"
    max_new_tokens = 20

    def __init__(self, n_examples: int = 100, seed: int = 42):
        self.n_examples = n_examples
        self.seed = seed

    def load_examples(self, n: int | None = None) -> list[Example]:
        ds = datasets.load_dataset(
            "mandarjoshi/trivia_qa", "rc.nocontext", split="validation"
        )
        count = n if n is not None else self.n_examples

        indices = list(range(len(ds)))
        random.Random(self.seed).shuffle(indices)
        indices = indices[:count]

        examples = []
        for i, idx in enumerate(indices):
            item = ds[idx]
            question = item["question"]
            aliases = item["answer"]["aliases"]
            normalized_value = item["answer"]["normalized_aliases"]

            first_alias = aliases[0] if aliases else item["answer"]["value"]
            prompt = f"Question: {question}\nAnswer:"
            examples.append(Example(
                id=str(i),
                prompt=prompt,
                target=f" {first_alias}",
                metadata={
                    "aliases": aliases,
                    "normalized_aliases": normalized_value,
                    "question_id": item.get("question_id", str(idx)),
                },
            ))
        return examples

    def match(self, output: str, target: str) -> bool:
        return _normalize(output) == _normalize(target)

    def _match_any_alias(self, output: str, aliases: list[str]) -> bool:
        normalized_output = _normalize(output)
        for alias in aliases:
            normalized_alias = _normalize(alias)
            if normalized_alias and normalized_alias in normalized_output:
                return True
        return False

    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult:
        per_example = []
        correct = 0
        logprobs = []

        for ex in examples:
            output = model.generate(ex.prompt, max_new_tokens=self.max_new_tokens)
            aliases = ex.metadata["aliases"]
            is_correct = self._match_any_alias(output, aliases)
            correct += int(is_correct)

            ll = model.loglikelihood(ex.prompt, ex.target)
            logprobs.append(ll)

            per_example.append({
                "id": ex.id,
                "prompt": ex.prompt,
                "target": ex.target,
                "output": output,
                "correct": is_correct,
                "logprob_first_alias": ll,
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
            metrics={"accuracy": accuracy, "mean_logprob_first_alias": mean_logprob},
            per_example=per_example,
            config={
                "seed": self.seed,
                "max_new_tokens": self.max_new_tokens,
            },
        )
