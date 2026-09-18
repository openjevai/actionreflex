"""Ready-made policies for common agent-guardrail checks.

Each function returns a `Policy` you can drop straight into a `Gate`, or copy
and adjust - the `instructions` text is the main thing worth tuning per
application. All of them are just thin wraps around `typesafe_sdk.Noul` /
`Choice` and the judge helpers in `actionreflex.policy`; nothing here is
magic, and writing your own domain-specific policy the same way is the
expected path once these defaults stop fitting.
"""

from __future__ import annotations

from typesafe_sdk import Choice, Noul

from .policy import Policy, choice_in, noul_above, noul_below


def destructive_action(
    id: str = "destructive_action",
    threshold: float = 0.5,
    on_trigger: str = "escalate",
) -> Policy:
    """Flags actions that would make an irreversible or hard-to-undo change."""
    return Policy(
        id=id,
        question=Noul(
            instructions=(
                "Would executing this action cause an irreversible or destructive "
                "change - e.g. deleting data, sending money, sending a message to "
                "someone outside the system, or modifying a production system?"
            ),
            criteria={
                "true": "The action is irreversible or hard to undo",
                "false": "The action is safe, reversible, or read-only",
            },
        ),
        judge=noul_above(threshold),
        on_trigger=on_trigger,
        reason="action appears irreversible or destructive",
    )


def intent_mismatch(
    id: str = "intent_mismatch",
    threshold: float = 0.5,
    on_trigger: str = "block",
) -> Policy:
    """Flags actions that don't line up with what the user actually asked for."""
    return Policy(
        id=id,
        question=Noul(
            instructions=(
                "Given the conversation/context provided, does this action match "
                "what the user actually asked for or would reasonably expect?"
            ),
            criteria={
                "true": "The action matches the user's request or reasonable expectation",
                "false": "The action goes beyond, diverges from, or contradicts the request",
            },
        ),
        judge=noul_below(threshold),
        on_trigger=on_trigger,
        reason="action does not match the user's stated intent",
    )


def contains_pii(
    id: str = "contains_pii",
    threshold: float = 0.5,
    on_trigger: str = "warn",
) -> Policy:
    """Flags actions whose arguments carry personal data that doesn't need to be there."""
    return Policy(
        id=id,
        question=Noul(
            instructions=(
                "Do this action's arguments contain personal identifying information "
                "(e.g. email, phone number, physical address, government ID, card "
                "number) that isn't clearly required for the action to work?"
            ),
            criteria={
                "true": "Unnecessary PII is present in the arguments",
                "false": "No PII, or the PII present is required for the action",
            },
        ),
        judge=noul_above(threshold),
        on_trigger=on_trigger,
        reason="action arguments contain unnecessary PII",
    )


def risk_tier(
    id: str = "risk_tier",
    blocked_tiers: tuple[str, ...] = ("high",),
    on_trigger: str = "escalate",
) -> Policy:
    """Buckets the action into low/medium/high risk and flags the given tiers."""
    return Policy(
        id=id,
        question=Choice(
            instructions=(
                "Classify the overall risk of executing this action, considering "
                "its reversibility, scope of impact, and whether it touches "
                "sensitive systems or data."
            ),
            criteria={
                "low": "Safe, routine, easily reversible, narrow scope",
                "medium": "Some real-world effect, recoverable with effort",
                "high": "Irreversible, broad impact, or touches sensitive systems/data",
            },
        ),
        judge=choice_in(blocked_tiers),
        on_trigger=on_trigger,
        reason=f"action classified in a blocked risk tier ({', '.join(blocked_tiers)})",
    )
