from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeClient, TypeSafeError

from .action import Action
from .exceptions import ActionBlocked, ActionEscalated, GateUnavailable
from .policy import Policy
from .verdict import Verdict

if TYPE_CHECKING:
    from typing_extensions import Self

ErrorMode = Literal["raise", "fail_closed", "fail_open"]

DEFAULT_MODEL = "jev-latest"


class _GateBase:
    """Shared policy handling and failure semantics for `Gate` / `AsyncGate`."""

    def __init__(
        self,
        policies: Iterable[Policy] = (),
        *,
        model: str | None = DEFAULT_MODEL,
        on_error: ErrorMode = "raise",
    ):
        self.policies: list[Policy] = list(policies)
        self.model = model
        if on_error not in ("raise", "fail_closed", "fail_open"):
            raise ValueError(
                f"on_error must be 'raise', 'fail_closed' or 'fail_open', got {on_error!r}"
            )
        self.on_error: ErrorMode = on_error

    def add(self, policy: Policy):
        self.policies.append(policy)
        return self

    def _active(self, policies: Iterable[Policy] | None) -> list[Policy]:
        active = list(policies) if policies is not None else self.policies
        if not active:
            raise ValueError("A gate needs at least one policy to check an action against.")
        return active

    def _request_kwargs(self, action: Action, active: list[Policy]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "state": action.to_state(),
            "questions": {p.id: p.question for p in active},
        }
        if self.model is not None:
            kwargs["model"] = self.model
        return kwargs

    def _handle_failure(self, action: Action, exc: TypeSafeError, started: float) -> Verdict:
        """Apply the configured failure mode when the TypeSafe call doesn't succeed."""
        latency_ms = (time.perf_counter() - started) * 1000
        if self.on_error == "raise":
            raise GateUnavailable(
                f"Could not check action '{action.name}': {exc}", cause=exc
            ) from exc
        return Verdict(
            action=action,
            decision="block" if self.on_error == "fail_closed" else "allow",
            results=[],
            latency_ms=latency_ms,
            error=exc,
        )


class Gate(_GateBase):
    """Checks a proposed `Action` against a set of policies using TypeSafe's Jev model.

    Every policy is asked in the *same* `system_one` request, so they're evaluated
    in parallel: adding policies costs tokens, not extra round trips. A check is a
    single Jev call - typically 70-500ms and a fraction of a cent - which is the
    point: it's affordable to check every action an agent proposes.

        from actionreflex import Action, Gate
        from actionreflex.policies import destructive_action, intent_mismatch

        with Gate([destructive_action(), intent_mismatch()]) as gate:
            verdict = gate.check(Action(
                name="delete_customer_account",
                arguments={"customer_id": "cust_8123"},
                context="User asked: 'why didn't my last order ship?'",
            ))
            if verdict.allowed:
                ...

    Auth comes from `api_key`, or from `TYPESAFE_API_KEY` in the environment.

    `on_error` decides what happens when the TypeSafe call itself fails (network
    error, rate limit, auth problem) - a real decision you should make deliberately:

    - "raise" (default): raise `GateUnavailable`. The action does not execute and
      you find out immediately.
    - "fail_closed": return a `Verdict` with decision "block" and `.error` set.
      Safe, but a TypeSafe outage stops your agent acting at all.
    - "fail_open": return a `Verdict` with decision "allow" and `.error` set.
      Keeps the agent running, but the action executes unchecked - only pick this
      if an unchecked action is genuinely less bad than a halted agent, and log
      `verdict.error` if you do.
    """

    def __init__(
        self,
        policies: Iterable[Policy] = (),
        *,
        api_key: str | None = None,
        model: str | None = DEFAULT_MODEL,
        on_error: ErrorMode = "raise",
        client: TypeSafeClient | None = None,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
    ):
        super().__init__(policies, model=model, on_error=on_error)
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = TypeSafeClient(api_key=api_key, timeout=timeout, retry=retry)
            self._owns_client = True

    @property
    def client(self) -> TypeSafeClient:
        return self._client

    def check(self, action: Action, policies: Iterable[Policy] | None = None) -> Verdict:
        """Ask Jev about `action` and return an allow/block/escalate `Verdict`."""
        active = self._active(policies)
        started = time.perf_counter()
        try:
            response = self._client.system_one(**self._request_kwargs(action, active))
        except TypeSafeError as exc:
            return self._handle_failure(action, exc, started)
        latency_ms = (time.perf_counter() - started) * 1000
        return Verdict.from_answers(
            action=action,
            policies=active,
            answers=response.answers,
            latency_ms=latency_ms,
            usage=response.usage,
        )

    def guard(
        self,
        func: Callable | None = None,
        *,
        policies: Iterable[Policy] | None = None,
        action_builder: Callable[..., Action] | None = None,
        on_block: Callable[[Verdict], Any] | None = None,
        on_escalate: Callable[[Verdict], Any] | None = None,
    ):
        """Decorator: check before running the wrapped function.

        `policies` overrides the gate's own policies for this function only - useful
        because a read-only lookup and a payment tool rarely need the same checks.

        By default the `Action` is built from the function's name and its bound
        arguments, e.g. `delete_record(42)` -> `Action("delete_record", {"record_id": 42})`;
        pass `action_builder(*args, **kwargs) -> Action` for anything more specific
        (e.g. pulling conversation context into `Action.context`).

        On block: raises `ActionBlocked`, or calls `on_block(verdict)` if given.
        On escalate: raises `ActionEscalated`, or calls `on_escalate(verdict)` if given -
        escalation means "a human or a slower path should decide", so there is no
        automatic "proceed anyway" default.
        """

        if func is not None and not callable(func):
            raise TypeError("guard() takes policies as a keyword: @gate.guard(policies=[...])")
        guard_policies = list(policies) if policies is not None else None

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                action = _build_action(fn, action_builder, args, kwargs)
                verdict = self.check(action, policies=guard_policies)
                outcome = _apply_verdict(verdict, on_block, on_escalate)
                if outcome is not _PROCEED:
                    return outcome
                return fn(*args, **kwargs)

            return wrapper

        return decorator(func) if func is not None else decorator

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class AsyncGate(_GateBase):
    """Async counterpart to `Gate`, backed by `AsyncTypeSafeClient`.

    Same policies, same verdicts, same failure modes:

        async with AsyncGate([destructive_action()]) as gate:
            verdict = await gate.check(action)
    """

    def __init__(
        self,
        policies: Iterable[Policy] = (),
        *,
        api_key: str | None = None,
        model: str | None = DEFAULT_MODEL,
        on_error: ErrorMode = "raise",
        client: AsyncTypeSafeClient | None = None,
        timeout: float | None = None,
        retry: RetryPolicy | None = None,
    ):
        super().__init__(policies, model=model, on_error=on_error)
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = AsyncTypeSafeClient(api_key=api_key, timeout=timeout, retry=retry)
            self._owns_client = True

    @property
    def client(self) -> AsyncTypeSafeClient:
        return self._client

    async def check(self, action: Action, policies: Iterable[Policy] | None = None) -> Verdict:
        """Ask Jev about `action` and return an allow/block/escalate `Verdict`."""
        active = self._active(policies)
        started = time.perf_counter()
        try:
            response = await self._client.system_one(**self._request_kwargs(action, active))
        except TypeSafeError as exc:
            return self._handle_failure(action, exc, started)
        latency_ms = (time.perf_counter() - started) * 1000
        return Verdict.from_answers(
            action=action,
            policies=active,
            answers=response.answers,
            latency_ms=latency_ms,
            usage=response.usage,
        )

    def guard(
        self,
        func: Callable | None = None,
        *,
        policies: Iterable[Policy] | None = None,
        action_builder: Callable[..., Action] | None = None,
        on_block: Callable[[Verdict], Any] | None = None,
        on_escalate: Callable[[Verdict], Any] | None = None,
    ):
        """Decorator for async tools. See `Gate.guard`."""

        if func is not None and not callable(func):
            raise TypeError("guard() takes policies as a keyword: @gate.guard(policies=[...])")
        guard_policies = list(policies) if policies is not None else None

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                action = _build_action(fn, action_builder, args, kwargs)
                verdict = await self.check(action, policies=guard_policies)
                outcome = _apply_verdict(verdict, on_block, on_escalate)
                if outcome is not _PROCEED:
                    return outcome
                return await fn(*args, **kwargs)

            return wrapper

        return decorator(func) if func is not None else decorator

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()


_PROCEED = object()
"""Sentinel: the verdict allows the wrapped function to run."""


def _build_action(
    fn: Callable,
    action_builder: Callable[..., Action] | None,
    args: tuple,
    kwargs: dict,
) -> Action:
    if action_builder is not None:
        return action_builder(*args, **kwargs)
    return Action(name=fn.__name__, arguments=_bind_arguments(fn, args, kwargs))


def _bind_arguments(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Map positional and keyword arguments onto the function's parameter names,
    so `delete_record(42)` is checked as `{"record_id": 42}` rather than `{}`."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
    except (TypeError, ValueError):
        return {"args": list(args), **kwargs}
    bound.apply_defaults()
    return {name: value for name, value in bound.arguments.items() if name not in ("self", "cls")}


def _apply_verdict(
    verdict: Verdict,
    on_block: Callable[[Verdict], Any] | None,
    on_escalate: Callable[[Verdict], Any] | None,
) -> Any:
    if verdict.decision == "block":
        if on_block is not None:
            return on_block(verdict)
        raise ActionBlocked(verdict)
    if verdict.decision == "escalate":
        if on_escalate is not None:
            return on_escalate(verdict)
        raise ActionEscalated(verdict)
    return _PROCEED
