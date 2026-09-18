"""Decision logic, exercised with real `typesafe_sdk` answer objects - the same
types `TypeSafeClient.system_one` returns in `response.answers`."""

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, Usage

from actionreflex import Action, MissingAnswer, Verdict
from actionreflex.policies import contains_pii, destructive_action, intent_mismatch, risk_tier

ACTION = Action(name="delete_customer_account", arguments={"customer_id": "cust_8123"})


def verdict_for(policies, answers, **kwargs):
    return Verdict.from_answers(action=ACTION, policies=policies, answers=answers, **kwargs)


def test_allow_when_nothing_triggers():
    verdict = verdict_for(
        [destructive_action(), intent_mismatch()],
        {"destructive_action": NoulAnswer(noul=0.1), "intent_mismatch": NoulAnswer(noul=0.9)},
    )
    assert verdict.allowed
    assert verdict.decision == "allow"
    assert verdict.triggered_results == []
    assert verdict.reasons == []


def test_intent_mismatch_triggers_on_low_match_probability():
    verdict = verdict_for([intent_mismatch()], {"intent_mismatch": NoulAnswer(noul=0.2)})
    assert verdict.decision == "block"
    assert verdict.reasons == ["action does not follow from the user's request"]


def test_escalate():
    verdict = verdict_for([destructive_action()], {"destructive_action": NoulAnswer(noul=0.9)})
    assert verdict.decision == "escalate"
    assert not verdict.allowed


def test_block_wins_over_escalate_regardless_of_order():
    policies = [destructive_action(on_trigger="escalate"), intent_mismatch(on_trigger="block")]
    answers = {"destructive_action": NoulAnswer(noul=0.9), "intent_mismatch": NoulAnswer(noul=0.1)}
    assert verdict_for(policies, answers).decision == "block"
    assert verdict_for(list(reversed(policies)), answers).decision == "block"


def test_warn_is_recorded_but_does_not_change_decision():
    verdict = verdict_for([contains_pii()], {"contains_pii": NoulAnswer(noul=0.95)})
    assert verdict.decision == "allow"
    assert [r.policy_id for r in verdict.triggered_results] == ["contains_pii"]


def test_choice_policy():
    answer = ChoiceAnswer(
        choice="high", confidence=0.72, probabilities={"low": 0.08, "medium": 0.2, "high": 0.72}
    )
    verdict = verdict_for([risk_tier()], {"risk_tier": answer})
    assert verdict.decision == "escalate"
    assert verdict.results[0].probability == pytest.approx(0.72)


def test_probability_exposes_noul_value():
    verdict = verdict_for([destructive_action()], {"destructive_action": NoulAnswer(noul=0.83)})
    assert verdict.results[0].probability == pytest.approx(0.83)


def test_usage_and_latency_are_carried_through():
    verdict = verdict_for(
        [destructive_action()],
        {"destructive_action": NoulAnswer(noul=0.1)},
        latency_ms=112.5,
        usage=Usage(input_tokens=240, output_tokens=0),
        model="jev-1.13.0",
    )
    assert verdict.latency_ms == 112.5
    assert verdict.usage.input_tokens == 240
    assert verdict.model == "jev-1.13.0"


def test_missing_answer_is_an_explicit_error():
    with pytest.raises(MissingAnswer, match="destructive_action"):
        verdict_for([destructive_action()], {"something_else": NoulAnswer(noul=0.1)})
