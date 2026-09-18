from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MockNoulAnswer:
    noul: float


@dataclass
class MockChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass
class MockScoreAnswer:
    score: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    confidence: float


class MockBackend:
    """Deterministic, offline stand-in for `TypeSafeBackend` - no network, no API key.

    Useful for developing and testing `actionreflex` itself (or anything built on
    it) without early access to the real Jev API. By default every Noul question
    resolves to `noul=0.0` (a low/"no" probability), every Choice resolves to its
    first option, and every Score resolves to its first (index 0) level. Whether
    that "triggers" a given policy depends on the policy's `judge` - it won't
    trigger a `noul_above()` judge (like `destructive_action`'s), but it *will*
    trigger a `noul_below()` judge (like `intent_mismatch`'s, where a low
    probability means "does not match"). Script specific scenarios explicitly
    with `responses`, keyed by policy/question id, when you need a scenario
    to come out a specific way:

        MockBackend(responses={"destructive": 0.9, "risk_tier": "high", "urgency": 2})

    A Noul override is a float (the probability of "yes"). A Choice override is
    the option string to select. A Score override is the index into that
    question's `criteria` list.
    """

    def __init__(
        self,
        responses: dict[str, float | int | str] | None = None,
        default_noul: float = 0.0,
    ):
        self.responses = responses or {}
        self.default_noul = default_noul

    def evaluate(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        return {qid: self._answer_for(qid, question) for qid, question in questions.items()}

    async def aevaluate(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        return self.evaluate(state, questions, model)

    def _answer_for(self, qid: str, question: Any) -> Any:
        qtype = type(question).__name__.lower()
        override = self.responses.get(qid)

        if qtype == "noul":
            value = override if isinstance(override, (int, float)) else self.default_noul
            return MockNoulAnswer(noul=float(value))

        if qtype == "choice":
            options = list(getattr(question, "criteria", {}) or {})
            if isinstance(override, str) and override in options:
                choice = override
            else:
                choice = options[0] if options else "unknown"
            probabilities = {opt: (1.0 if opt == choice else 0.0) for opt in options}
            return MockChoiceAnswer(choice=choice, probabilities=probabilities, confidence=1.0)

        if qtype == "score":
            criteria = list(getattr(question, "criteria", []) or [])
            legend = {i: level for i, level in enumerate(criteria)}
            index = int(override) if isinstance(override, (int, float)) else 0
            probabilities = {i: (1.0 if i == index else 0.0) for i in range(len(criteria))}
            return MockScoreAnswer(
                score=float(index), legend=legend, probabilities=probabilities, confidence=1.0
            )

        raise ValueError(f"MockBackend: unrecognized question type for '{qid}': {question!r}")
