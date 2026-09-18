from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

OnTrigger = Literal["block", "escalate", "warn"]


@dataclass
class Policy:
    """One judgment to run against a proposed action before it executes.

    `question` is a `typesafe_sdk.Noul | Choice | Score` instance - the thing
    actually sent to Jev. `judge` turns its raw answer into a triggered/not-triggered
    bool (e.g. "noul probability >= 0.6", "choice in {'high', 'critical'}").
    `on_trigger` decides what a trigger does to the overall verdict: "block" always
    wins, "escalate" wins unless something else blocks, "warn" never changes the
    decision but still shows up in `Verdict.triggered_results` for logging/audit.

    Most policies won't need to build this by hand - see `actionreflex.policies`
    for ready-made ones, and the `noul_above` / `noul_below` / `choice_in` /
    `score_at_least` helpers below for common judge shapes.
    """

    id: str
    question: Any
    judge: Callable[[Any], bool]
    on_trigger: OnTrigger = "block"
    reason: str | None = None

    def reason_for(self, answer: Any) -> str:
        if self.reason:
            return self.reason
        return f"policy '{self.id}' triggered (answer: {answer!r})"


def noul_above(threshold: float) -> Callable[[Any], bool]:
    """Judge: triggers when the Noul (yes-probability) answer is >= threshold.

    Use for questions phrased so "yes" is the bad outcome, e.g.
    "Would this action be destructive or irreversible?".
    """

    def _judge(answer: Any) -> bool:
        return answer.noul >= threshold

    return _judge


def noul_below(threshold: float) -> Callable[[Any], bool]:
    """Judge: triggers when the Noul answer is < threshold.

    Use for questions phrased so "yes" is the good outcome, e.g.
    "Does this action match what the user asked for?" - triggers on low match.
    """

    def _judge(answer: Any) -> bool:
        return answer.noul < threshold

    return _judge


def choice_in(blocked_options: set[str] | list[str]) -> Callable[[Any], bool]:
    """Judge: triggers when the selected Choice option is one of `blocked_options`."""
    blocked = set(blocked_options)

    def _judge(answer: Any) -> bool:
        return answer.choice in blocked

    return _judge


def score_at_least(threshold: float) -> Callable[[Any], bool]:
    """Judge: triggers when the Score answer's numeric value is >= threshold
    (threshold is an index into the question's `criteria` levels)."""

    def _judge(answer: Any) -> bool:
        return answer.score >= threshold

    return _judge
