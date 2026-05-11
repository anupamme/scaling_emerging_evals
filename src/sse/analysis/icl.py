"""Analysis utilities for in-context learning experiments."""

from __future__ import annotations

from sse.results import EvalResult


def compute_icl_delta(results: list[EvalResult]) -> dict[str, float]:
    """Compute ICL delta from results at different shot counts.

    Expects results from the same model/seed with different n_shots in config.
    Returns metrics including the accuracy delta from 0-shot to max-shot.
    """
    by_shots: dict[int, EvalResult] = {}
    for r in results:
        n_shots = r.config.get("n_shots", 0)
        by_shots[n_shots] = r

    if 0 not in by_shots:
        raise ValueError("No 0-shot result found")

    max_shots = max(by_shots.keys())
    if max_shots == 0:
        raise ValueError("Need results at more than one shot count")

    acc_0 = by_shots[0].metrics["accuracy"]
    acc_max = by_shots[max_shots].metrics["accuracy"]

    delta = acc_max - acc_0

    out = {
        "icl_delta_accuracy": delta,
        "accuracy_0shot": acc_0,
        f"accuracy_{max_shots}shot": acc_max,
    }

    if 1 in by_shots:
        out["accuracy_1shot"] = by_shots[1].metrics["accuracy"]

    return out
