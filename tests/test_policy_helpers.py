from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from actionreflex.policy import choice_in, noul_above, noul_below, score_at_least


def test_noul_above():
    judge = noul_above(0.5)
    assert judge(NoulAnswer(noul=0.6)) is True
    assert judge(NoulAnswer(noul=0.5)) is True
    assert judge(NoulAnswer(noul=0.4)) is False


def test_noul_below():
    judge = noul_below(0.5)
    assert judge(NoulAnswer(noul=0.4)) is True
    assert judge(NoulAnswer(noul=0.5)) is False
    assert judge(NoulAnswer(noul=0.6)) is False


def test_choice_in():
    judge = choice_in({"high", "critical"})
    high = ChoiceAnswer(choice="high", confidence=0.9, probabilities={"low": 0.1, "high": 0.9})
    low = ChoiceAnswer(choice="low", confidence=0.8, probabilities={"low": 0.8, "high": 0.2})
    assert judge(high) is True
    assert judge(low) is False


def test_score_at_least():
    judge = score_at_least(1.5)
    legend = {0: "narrow", 1: "bounded", 2: "wide"}
    wide = ScoreAnswer(
        score=1.8, confidence=0.7, legend=legend, probabilities={0: 0.05, 1: 0.1, 2: 0.85}
    )
    bounded = ScoreAnswer(
        score=1.1, confidence=0.6, legend=legend, probabilities={0: 0.1, 1: 0.7, 2: 0.2}
    )
    assert judge(wide) is True
    assert judge(bounded) is False
