from typesafe_sdk import Choice, Noul, Score

from actionreflex.backends.mock import MockBackend


def test_default_noul_never_triggers_high():
    backend = MockBackend()
    answers = backend.evaluate(
        state={}, questions={"q": Noul(instructions="risky?", criteria={"true": "y", "false": "n"})}
    )
    assert answers["q"].noul == 0.0


def test_noul_override():
    backend = MockBackend(responses={"q": 0.75})
    answers = backend.evaluate(
        state={}, questions={"q": Noul(instructions="risky?", criteria={"true": "y", "false": "n"})}
    )
    assert answers["q"].noul == 0.75


def test_choice_override_and_probabilities():
    backend = MockBackend(responses={"q": "high"})
    question = Choice(instructions="risk?", criteria={"low": "l", "high": "h"})
    answers = backend.evaluate(state={}, questions={"q": question})
    assert answers["q"].choice == "high"
    assert answers["q"].probabilities == {"low": 0.0, "high": 1.0}


def test_score_override_uses_int_keys():
    backend = MockBackend(responses={"q": 1})
    question = Score(instructions="rate", criteria=["low", "medium", "high"])
    answers = backend.evaluate(state={}, questions={"q": question})
    assert answers["q"].score == 1.0
    assert answers["q"].legend == {0: "low", 1: "medium", 2: "high"}
    assert answers["q"].probabilities == {0: 0.0, 1: 1.0, 2: 0.0}


async def test_aevaluate_matches_evaluate():
    backend = MockBackend(responses={"q": 0.5})
    question = Noul(instructions="risky?", criteria={"true": "y", "false": "n"})
    sync_answer = backend.evaluate(state={}, questions={"q": question})["q"]
    async_answer = (await backend.aevaluate(state={}, questions={"q": question}))["q"]
    assert sync_answer.noul == async_answer.noul
