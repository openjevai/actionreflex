from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .action import Action
from .exceptions import MissingAnswer

if TYPE_CHECKING:
    from typesafe_sdk import Usage

    from .policy import Policy

Decision = Literal["allow", "block", "escalate"]


@dataclass
class PolicyResult:
    """The outcome of one policy's check against one action.

    `raw_answer` is the answer object Jev returned (`NoulAnswer`, `ChoiceAnswer`
    or `ScoreAnswer`) - keep it around if you want the underlying probabilities
    for logging, threshold tuning, or offline analysis of your own traffic.
    """

    policy_id: str
    raw_answer: Any
    triggered: bool
    on_trigger: Literal["block", "escalate", "warn"]
    reason: str

    @property
    def probability(self) -> float | None:
        """The probability behind this result, when the answer type has a single
        obvious one: a Noul's yes-probability, or a Choice/Score's confidence."""
        answer = self.raw_answer
        if hasattr(answer, "noul"):
            return answer.noul
        return getattr(answer, "confidence", None)


@dataclass
class Verdict:
    """The gate's overall decision for one action, plus every policy's result."""

    action: Action
    decision: Decision
    results: list[PolicyResult] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: Usage | None = None
    error: Exception | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def triggered_results(self) -> list[PolicyResult]:
        return [r for r in self.results if r.triggered]

    @property
    def reasons(self) -> list[str]:
        return [r.reason for r in self.triggered_results]

    @classmethod
    def from_answers(
        cls,
        action: Action,
        policies: list[Policy],
        answers: dict[str, Any],
        latency_ms: float = 0.0,
        usage: Usage | None = None,
    ) -> Verdict:
        """Fold Jev's answers into a decision.

        "block" always wins; "escalate" wins over "allow"; "warn" never changes
        the decision but still lands in `triggered_results` for logging/audit.
        """
        results: list[PolicyResult] = []
        decision: Decision = "allow"

        for policy in policies:
            if policy.id not in answers:
                raise MissingAnswer(
                    f"No answer returned for policy '{policy.id}'. Answers received: "
                    f"{sorted(answers)}"
                )
            answer = answers[policy.id]
            triggered = policy.judge(answer)
            results.append(
                PolicyResult(
                    policy_id=policy.id,
                    raw_answer=answer,
                    triggered=triggered,
                    on_trigger=policy.on_trigger,
                    reason=policy.reason_for(answer) if triggered else "",
                )
            )
            if triggered:
                if policy.on_trigger == "block":
                    decision = "block"
                elif policy.on_trigger == "escalate" and decision == "allow":
                    decision = "escalate"

        return cls(
            action=action,
            decision=decision,
            results=results,
            latency_ms=latency_ms,
            usage=usage,
        )

    def __repr__(self) -> str:
        triggered = ", ".join(r.policy_id for r in self.triggered_results) or "-"
        return (
            f"Verdict(action={self.action.name!r}, decision={self.decision!r}, "
            f"triggered=[{triggered}], latency_ms={self.latency_ms:.1f})"
        )
