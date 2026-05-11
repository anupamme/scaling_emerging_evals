"""Multiple-choice task using TruthfulQA MC1 with length-normalized scoring."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import datasets

from sse.results import EvalResult
from sse.tasks.base import Example, MultipleChoiceTask

if TYPE_CHECKING:
    from sse.models.pythia import PythiaModel


class TruthfulQATask(MultipleChoiceTask):
    """TruthfulQA MC1 — multiple choice with length-normalized log-likelihood scoring.

    Each question has 4-5 choices. Scoring uses per-token normalized log-likelihood
    to avoid bias toward shorter completions.
    """

    name = "truthfulqa_mc1"
    version = "v1"

    def __init__(self, n_examples: int = 200, seed: int = 42):
        self.n_examples = n_examples
        self.seed = seed

    def load_examples(self, n: int | None = None) -> list[Example]:
        ds = datasets.load_dataset("truthful_qa", "multiple_choice", split="validation")
        count = n if n is not None else self.n_examples

        indices = list(range(len(ds)))
        random.Random(self.seed).shuffle(indices)
        indices = indices[:count]

        examples = []
        for i, idx in enumerate(indices):
            item = ds[idx]
            question = item["question"]
            mc1 = item["mc1_targets"]
            choices_raw = mc1["choices"]
            labels = mc1["labels"]

            choices = [f" {c}" for c in choices_raw]
            correct_idx = labels.index(1)
            target = choices[correct_idx]

            examples.append(Example(
                id=str(i),
                prompt=f"Question: {question}\nAnswer:",
                target=target,
                choices=choices,
                metadata={"correct_idx": correct_idx},
            ))
        return examples

    def evaluate(self, model: PythiaModel, examples: list[Example]) -> EvalResult:
        per_example = []
        correct = 0
        logprobs_correct = []
        margins = []

        for ex in examples:
            assert ex.choices is not None

            raw_scores = [
                model.loglikelihood(ex.prompt, choice) for choice in ex.choices
            ]
            token_counts = [
                len(model.tokenizer(choice)["input_ids"]) for choice in ex.choices
            ]
            norm_scores = [s / t for s, t in zip(raw_scores, token_counts)]

            predicted_idx = max(range(len(norm_scores)), key=lambda i: norm_scores[i])
            predicted = ex.choices[predicted_idx]
            is_correct = predicted == ex.target
            correct += int(is_correct)

            correct_idx = ex.choices.index(ex.target)
            logprobs_correct.append(norm_scores[correct_idx])

            incorrect_scores = [
                s for i, s in enumerate(norm_scores) if i != correct_idx
            ]
            margin = norm_scores[correct_idx] - max(incorrect_scores)
            margins.append(margin)

            per_example.append({
                "id": ex.id,
                "prompt": ex.prompt,
                "target": ex.target,
                "choices": ex.choices,
                "raw_scores": raw_scores,
                "norm_scores": norm_scores,
                "predicted": predicted,
                "correct": is_correct,
                "margin": margin,
            })

        n = len(examples)
        accuracy = correct / n if n else 0.0
        mean_logprob = sum(logprobs_correct) / n if n else 0.0
        mean_margin = sum(margins) / n if n else 0.0

        return EvalResult(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_size=model.size,
            model_revision=model.revision,
            task_name=self.name,
            task_version=self.version,
            n_examples=n,
            metrics={
                "accuracy": accuracy,
                "mean_logprob_correct": mean_logprob,
                "mean_logprob_margin": mean_margin,
            },
            per_example=per_example,
            config={"seed": self.seed, "length_normalized": True},
        )
