from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .action import Action

Decision = Literal["allow", "block", "escalate"]


@dataclass
class PolicyResult:
    """The outcome of one policy's check against one action."""

    policy_id: str
    raw_answer: Any
    triggered: bool
    on_trigger: Literal["block", "escalate", "warn"]
    reason: str


@dataclass
class Verdict:
    """The gate's overall decision for one action, plus every policy's result."""

    action: Action
    decision: Decision
    results: list[PolicyResult]
    latency_ms: float

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @property
    def triggered_results(self) -> list[PolicyResult]:
        return [r for r in self.results if r.triggered]

    def __repr__(self) -> str:
        reasons = ", ".join(r.policy_id for r in self.triggered_results) or "-"
        return (
            f"Verdict(action={self.action.name!r}, decision={self.decision!r}, "
            f"triggered=[{reasons}], latency_ms={self.latency_ms:.1f})"
        )
