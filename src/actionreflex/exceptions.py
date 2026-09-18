from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .verdict import Verdict


class ActionReflexError(Exception):
    """Base class for actionreflex errors."""


class ActionBlocked(ActionReflexError):
    """Raised by `Gate.guard` (default behavior) when a policy blocks the action."""

    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        reasons = "; ".join(r.reason for r in verdict.triggered_results) or "no reason given"
        super().__init__(f"Action '{verdict.action.name}' blocked: {reasons}")


class ActionEscalated(ActionReflexError):
    """Raised by `Gate.guard` (default behavior) when a policy escalates the action
    and no `on_escalate` handler was provided."""

    def __init__(self, verdict: Verdict):
        self.verdict = verdict
        reasons = "; ".join(r.reason for r in verdict.triggered_results) or "no reason given"
        super().__init__(f"Action '{verdict.action.name}' needs review: {reasons}")
