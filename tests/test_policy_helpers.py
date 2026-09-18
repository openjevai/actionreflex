from actionreflex.backends.mock import MockChoiceAnswer, MockNoulAnswer, MockScoreAnswer
from actionreflex.policy import choice_in, noul_above, noul_below, score_at_least


def test_noul_above():
    judge = noul_above(0.5)
    assert judge(MockNoulAnswer(noul=0.6)) is True
    assert judge(MockNoulAnswer(noul=0.4)) is False


def test_noul_below():
    judge = noul_below(0.5)
    assert judge(MockNoulAnswer(noul=0.4)) is True
    assert judge(MockNoulAnswer(noul=0.6)) is False


def test_choice_in():
    judge = choice_in({"high", "critical"})
    assert judge(MockChoiceAnswer(choice="high", probabilities={}, confidence=1.0)) is True
    assert judge(MockChoiceAnswer(choice="low", probabilities={}, confidence=1.0)) is False


def test_score_at_least():
    judge = score_at_least(2)
    assert judge(MockScoreAnswer(score=2.0, legend={}, probabilities={}, confidence=1.0)) is True
    assert judge(MockScoreAnswer(score=1.0, legend={}, probabilities={}, confidence=1.0)) is False
