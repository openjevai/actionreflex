"""Gate configuration, lifecycle, and failure handling.

Failure modes are exercised against the real `typesafe_sdk` client pointed at a
closed local port, so the errors are the SDK's own `TypeSafeAPIConnectionError` -
nothing is faked. Successful calls to Jev live in `test_integration.py`.
"""

import pytest
from typesafe_sdk import (
    AsyncTypeSafeClient,
    NoulAnswer,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeClient,
    TypeSafeError,
)

from actionreflex import (
    Action,
    ActionBlocked,
    ActionEscalated,
    AsyncGate,
    Gate,
    GateUnavailable,
    Verdict,
)
from actionreflex.gate import _PROCEED, _apply_verdict
from actionreflex.policies import destructive_action, intent_mismatch

UNREACHABLE = "http://127.0.0.1:9"
ACTION = Action(name="delete_customer_account", arguments={"customer_id": "cust_8123"})


def unreachable_client() -> TypeSafeClient:
    return TypeSafeClient(api_key="test", base_url=UNREACHABLE, retry=RetryPolicy(max_retries=0))


def unreachable_async_client() -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(
        api_key="test", base_url=UNREACHABLE, retry=RetryPolicy(max_retries=0)
    )


# --- configuration ---------------------------------------------------------


def test_rejects_unknown_error_mode():
    with pytest.raises(ValueError, match="on_error"):
        Gate([destructive_action()], client=unreachable_client(), on_error="ignore")


def test_check_requires_at_least_one_policy():
    with Gate(client=unreachable_client()) as gate, pytest.raises(ValueError, match="policy"):
        gate.check(ACTION)


def test_missing_api_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeError, match="API key"):
        Gate([destructive_action()])


def test_add_is_chainable():
    gate = Gate(client=unreachable_client())
    gate.add(destructive_action()).add(intent_mismatch())
    assert [p.id for p in gate.policies] == ["destructive_action", "intent_mismatch"]


# --- failure modes -----------------------------------------------------------


def test_raise_mode_wraps_sdk_error():
    with (
        Gate([destructive_action()], client=unreachable_client()) as gate,
        pytest.raises(GateUnavailable) as excinfo,
    ):
        gate.check(ACTION)
    assert isinstance(excinfo.value.cause, TypeSafeAPIConnectionError)
    assert excinfo.value.__cause__ is excinfo.value.cause


def test_fail_closed_blocks_and_records_error():
    with Gate([destructive_action()], client=unreachable_client(), on_error="fail_closed") as gate:
        verdict = gate.check(ACTION)
    assert verdict.decision == "block"
    assert isinstance(verdict.error, TypeSafeAPIConnectionError)
    assert verdict.results == []


def test_fail_open_allows_and_records_error():
    with Gate([destructive_action()], client=unreachable_client(), on_error="fail_open") as gate:
        verdict = gate.check(ACTION)
    assert verdict.decision == "allow"
    assert isinstance(verdict.error, TypeSafeAPIConnectionError)


async def test_async_gate_failure_modes():
    async with AsyncGate(
        [destructive_action()], client=unreachable_async_client(), on_error="fail_closed"
    ) as gate:
        verdict = await gate.check(ACTION)
    assert verdict.decision == "block"
    assert isinstance(verdict.error, TypeSafeAPIConnectionError)


# --- client lifecycle --------------------------------------------------------


def test_closing_gate_leaves_a_supplied_client_open():
    client = unreachable_client()
    Gate([destructive_action()], client=client).close()
    # Still open: we get a connection error, not "client has been closed".
    with pytest.raises(TypeSafeAPIConnectionError):
        client.system_one(state={}, questions={"q": destructive_action().question})
    client.close()


def test_closing_gate_closes_the_client_it_created(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    gate = Gate([destructive_action()])
    gate.close()
    with pytest.raises(RuntimeError, match="closed"):
        gate.client.system_one(state={}, questions={"q": destructive_action().question})


# --- guard decorator ---------------------------------------------------------


def test_guard_raises_gate_unavailable_by_default():
    gate = Gate([destructive_action()], client=unreachable_client())
    calls = []

    @gate.guard
    def delete_customer_account(customer_id: str):
        calls.append(customer_id)

    with pytest.raises(GateUnavailable):
        delete_customer_account("cust_8123")
    assert calls == []


def test_guard_does_not_run_function_when_blocked():
    gate = Gate([destructive_action()], client=unreachable_client(), on_error="fail_closed")
    calls = []

    @gate.guard
    def delete_customer_account(customer_id: str):
        calls.append(customer_id)

    with pytest.raises(ActionBlocked):
        delete_customer_account("cust_8123")
    assert calls == []


def test_guard_on_block_handler_replaces_the_call():
    gate = Gate([destructive_action()], client=unreachable_client(), on_error="fail_closed")
    seen = []

    def handle_block(verdict):
        seen.append(verdict)
        return "blocked"

    @gate.guard(on_block=handle_block)
    def delete_customer_account(customer_id: str):
        return "deleted"

    assert delete_customer_account("cust_8123") == "blocked"
    assert seen[0].action.arguments == {"customer_id": "cust_8123"}


def test_guard_runs_function_when_allowed():
    gate = Gate([destructive_action()], client=unreachable_client(), on_error="fail_open")

    @gate.guard
    def read_order(order_id: str):
        return f"order {order_id}"

    assert read_order("ord_1") == "order ord_1"


async def test_async_guard():
    gate = AsyncGate(
        [destructive_action()], client=unreachable_async_client(), on_error="fail_open"
    )

    @gate.guard
    async def read_order(order_id: str):
        return f"order {order_id}"

    assert await read_order("ord_1") == "order ord_1"
    await gate.aclose()


def test_escalate_raises_without_handler_and_uses_handler_when_given():
    verdict = Verdict.from_answers(
        action=ACTION,
        policies=[destructive_action()],
        answers={"destructive_action": NoulAnswer(noul=0.9)},
    )
    with pytest.raises(ActionEscalated):
        _apply_verdict(verdict, on_block=None, on_escalate=None)
    assert _apply_verdict(verdict, on_block=None, on_escalate=lambda v: "queued") == "queued"


def test_allow_proceeds():
    verdict = Verdict.from_answers(
        action=ACTION,
        policies=[destructive_action()],
        answers={"destructive_action": NoulAnswer(noul=0.05)},
    )
    assert _apply_verdict(verdict, on_block=None, on_escalate=None) is _PROCEED


def test_guard_policies_override_the_gates_own():
    gate = Gate(client=unreachable_client())  # no policies of its own

    @gate.guard
    def uses_gate_policies():
        return "ran"

    @gate.guard(policies=[destructive_action()])
    def uses_own_policies():
        return "ran"

    with pytest.raises(ValueError, match="policy"):
        uses_gate_policies()
    # Got past the policy check and actually tried to call TypeSafe.
    with pytest.raises(GateUnavailable):
        uses_own_policies()


def test_guard_rejects_positional_policies():
    gate = Gate(client=unreachable_client())
    with pytest.raises(TypeError, match="policies="):
        gate.guard([destructive_action()])


def test_every_builtin_policy_builds_a_request_the_sdk_accepts():
    """The SDK serializes and validates the request body before connecting, so
    reaching a connection error means every question, and a state holding
    non-JSON argument types, made it through to the wire format."""
    from datetime import datetime, timezone
    from decimal import Decimal

    from actionreflex.policies import default_policies, risk_tier, scope_of_impact

    action = Action(
        name="refund_order",
        arguments={"amount": Decimal("39.90"), "at": datetime.now(timezone.utc)},
        context={"user_request": "refund my broken item", "turns": ["hi", "it broke"]},
    )
    with (
        Gate(
            default_policies() + [scope_of_impact(), risk_tier()], client=unreachable_client()
        ) as gate,
        pytest.raises(GateUnavailable) as excinfo,
    ):
        gate.check(action)
    assert isinstance(excinfo.value.cause, TypeSafeAPIConnectionError)
