from collections.abc import Callable

from sse.tasks.arithmetic import ArithmeticTask
from sse.tasks.base import Example, GenerationTask, MultipleChoiceTask, Task
from sse.tasks.factual_recall import FactualRecallTask
from sse.tasks.icl import ICLArithmeticTask
from sse.tasks.multiple_choice import TruthfulQATask

__all__ = ["Example", "GenerationTask", "MultipleChoiceTask", "Task", "get_task"]

TASK_REGISTRY: dict[str, Callable[..., Task]] = {
    "arithmetic_2digit": lambda n, seed: ArithmeticTask(
        n_digits=2, operation="add", n_examples=n, seed=seed
    ),
    "arithmetic_3digit": lambda n, seed: ArithmeticTask(
        n_digits=3, operation="add", n_examples=n, seed=seed
    ),
    "factual_recall": lambda n, seed: FactualRecallTask(n_examples=n, seed=seed),
    "truthfulqa_mc1": lambda n, seed: TruthfulQATask(n_examples=n, seed=seed),
    "icl_arithmetic_2digit": lambda n, seed: ICLArithmeticTask(
        n_digits=2, operation="add", n_examples=n, n_shots=0, seed=seed
    ),
}


def get_task(name: str, n_examples: int = 100, seed: int = 42) -> Task:
    if name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{name}'. Available: {available}")
    return TASK_REGISTRY[name](n_examples, seed)
