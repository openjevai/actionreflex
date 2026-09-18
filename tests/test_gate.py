import pytest

from actionreflex import Action, ActionBlocked, ActionEscalated, Gate
from actionreflex.backends import MockBackend
from actionreflex.policies import destructive_action, intent_mismatch, risk_tier


def test_allows_when_nothing_triggers():
    # MockBackend defaults every Noul answer to 0.0. That's "safe" for an
    # above-threshold judge like destructive_action's, but intent_mismatch uses
    # noul_below() (low match = bad), so it needs an explicit high-match
    # response to represent "matches intent" - see test_blocks_on_intent_mismatch
    # for the low-match/triggers case.
    gate = Gate(
        policies=[destructive_action(), intent_mismatch()],
        backend=MockBackend(responses={"intent_mismatch": 0.95}),
    )
    verdict = gate.check(Action(name="read_record", arguments={"id": 1}))
    assert verdict.allowed
    assert verdict.decision == "allow"
    assert verdict.triggered_results == []


def test_blocks_on_intent_mismatch():
    gate = Gate(
        policies=[intent_mismatch(on_trigger="block")],
        backend=MockBackend(responses={"intent_mismatch": 0.1}),  # low match -> triggers
    )
    verdict = gate.check(Action(name="wire_transfer", arguments={"amount": 10000}))
    assert not verdict.allowed
    assert verdict.decision == "block"
    assert verdict.triggered_results[0].policy_id == "intent_mismatch"


def test_escalates_on_destructive_action():
    gate = Gate(
        policies=[destructive_action(on_trigger="escalate")],
        backend=MockBackend(responses={"destructive_action": 0.9}),
    )
    verdict = gate.check(Action(name="delete_account", arguments={"id": 1}))
    assert verdict.decision == "escalate"
    assert not verdict.allowed


def test_block_wins_over_escalate():
    gate = Gate(
        policies=[
            destructive_action(id="destructive", on_trigger="escalate"),
            intent_mismatch(id="mismatch", on_trigger="block"),
        ],
        backend=MockBackend(responses={"destructive": 0.9, "mismatch": 0.1}),
    )
    verdict = gate.check(Action(name="delete_account", arguments={"id": 1}))
    assert verdict.decision == "block"


def test_choice_policy_risk_tier():
    gate = Gate(
        policies=[risk_tier(blocked_tiers=("high", "medium"))],
        backend=MockBackend(responses={"risk_tier": "high"}),
    )
    verdict = gate.check(Action(name="drop_table", arguments={}))
    assert verdict.decision == "escalate"


def test_guard_decorator_blocks_by_default():
    gate = Gate(
        policies=[destructive_action(id="destructive", on_trigger="block")],
        backend=MockBackend(responses={"destructive": 0.99}),
    )

    @gate.guard()
    def delete_thing(id: int) -> str:
        return f"deleted {id}"

    with pytest.raises(ActionBlocked):
        delete_thing(id=1)


def test_guard_decorator_calls_on_block_handler_instead_of_raising():
    gate = Gate(
        policies=[destructive_action(id="destructive", on_trigger="block")],
        backend=MockBackend(responses={"destructive": 0.99}),
    )
    captured = []

    def handle_block(verdict):
        captured.append(verdict)
        return "blocked"

    @gate.guard(on_block=handle_block)
    def delete_thing(id: int) -> str:
        return f"deleted {id}"

    result = delete_thing(id=1)
    assert result == "blocked"
    assert captured[0].decision == "block"


def test_guard_decorator_allows_when_safe():
    gate = Gate(
        policies=[destructive_action(id="destructive", on_trigger="block")],
        backend=MockBackend(),  # default_noul=0.0 -> never triggers
    )

    @gate.guard()
    def read_thing(id: int) -> str:
        return f"read {id}"

    assert read_thing(id=1) == "read 1"


async def test_acheck_matches_check():
    gate = Gate(
        policies=[destructive_action(id="destructive")],
        backend=MockBackend(responses={"destructive": 0.9}),
    )
    verdict = await gate.acheck(Action(name="delete_account", arguments={"id": 1}))
    assert verdict.decision == "escalate"


def test_escalate_without_handler_raises():
    gate = Gate(
        policies=[destructive_action(id="destructive", on_trigger="escalate")],
        backend=MockBackend(responses={"destructive": 0.9}),
    )

    @gate.guard()
    def delete_thing(id: int) -> str:
        return f"deleted {id}"

    with pytest.raises(ActionEscalated):
        delete_thing(id=1)
