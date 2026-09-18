from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterable
from typing import Any

from .action import Action
from .backends.mock import MockBackend
from .exceptions import ActionBlocked, ActionEscalated
from .policy import Policy
from .verdict import PolicyResult, Verdict


class Gate:
    """Runs a set of policies against a proposed `Action` in a single batched
    TypeSafe call, and returns an allow/block/escalate `Verdict`.

    Because every policy question is asked in parallel in one `system_one`
    request, adding policies costs latency in the tens-of-ms range, not a
    multiplied number of round trips - the whole point is that this is cheap
    and fast enough to run on every action an agent proposes, not a sampled
    subset of them.

        gate = Gate(policies=[destructive_action(), intent_mismatch()])
        verdict = gate.check(Action(name="delete_record", arguments={"id": 42}))
        if not verdict.allowed:
            ...

    No backend is required for local development: `Gate()` defaults to
    `MockBackend`, which returns a fixed "no"/first-option answer for every
    question unless you script one (see `MockBackend`'s docstring - whether
    that default triggers a given policy depends on the policy's judge).
    Pass a real `TypeSafeBackend()` (or your own `Backend` implementation) to
    actually call Jev.
    """

    def __init__(
        self,
        policies: Iterable[Policy] = (),
        backend: Any | None = None,
        model: str = "jev-latest",
    ):
        self.policies: list[Policy] = list(policies)
        self.backend = backend if backend is not None else MockBackend()
        self.model = model

    def add(self, policy: Policy) -> Gate:
        self.policies.append(policy)
        return self

    def check(self, action: Action, policies: Iterable[Policy] | None = None) -> Verdict:
        active = list(policies) if policies is not None else self.policies
        started = time.perf_counter()
        answers = self.backend.evaluate(
            state=action.to_state(),
            questions={p.id: p.question for p in active},
            model=self.model,
        )
        return self._build_verdict(action, active, answers, started)

    async def acheck(self, action: Action, policies: Iterable[Policy] | None = None) -> Verdict:
        active = list(policies) if policies is not None else self.policies
        started = time.perf_counter()
        answers = await self.backend.aevaluate(
            state=action.to_state(),
            questions={p.id: p.question for p in active},
            model=self.model,
        )
        return self._build_verdict(action, active, answers, started)

    def _build_verdict(
        self,
        action: Action,
        active_policies: list[Policy],
        answers: dict[str, Any],
        started: float,
    ) -> Verdict:
        results: list[PolicyResult] = []
        decision = "allow"
        for policy in active_policies:
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
        latency_ms = (time.perf_counter() - started) * 1000
        return Verdict(action=action, decision=decision, results=results, latency_ms=latency_ms)

    def guard(
        self,
        func: Callable | None = None,
        *,
        action_builder: Callable[..., Action] | None = None,
        on_block: Callable[[Verdict], Any] | None = None,
        on_escalate: Callable[[Verdict], Any] | None = None,
    ):
        """Decorator sugar: check an action before running the wrapped function.

        By default the `Action` is built as `Action(name=func.__name__, arguments=kwargs)`;
        pass `action_builder(*args, **kwargs) -> Action` for anything more specific
        (e.g. pulling conversation context into `Action.context`).

        On block: raises `ActionBlocked`, or calls `on_block(verdict)` if given.
        On escalate: raises `ActionEscalated`, or calls `on_escalate(verdict)` if given -
        escalation is meant for a human-in-the-loop or a slower-path review, so there's
        no automatic "proceed anyway" default.
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                action = (
                    action_builder(*args, **kwargs)
                    if action_builder
                    else Action(name=fn.__name__, arguments=kwargs)
                )
                verdict = self.check(action)
                if verdict.decision == "block":
                    return on_block(verdict) if on_block else _raise(ActionBlocked(verdict))
                if verdict.decision == "escalate":
                    return (
                        on_escalate(verdict) if on_escalate else _raise(ActionEscalated(verdict))
                    )
                return fn(*args, **kwargs)

            return wrapper

        return decorator(func) if func is not None else decorator


def _raise(exc: Exception):
    raise exc
