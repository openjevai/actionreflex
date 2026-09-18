"""The eval harness's reading of R-Judge agent actions - it decides what the gate is
asked about, so the formats found in the dataset are pinned here."""

import pytest

from eval.datasets import parse_action


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GmailReadEmail{'email_id': 'email001'}", ("GmailReadEmail", {"email_id": "email001"})),
        (
            'GmailSendEmail\nAction Input: {"to": "a@b.c", "subject": "hi"}',
            ("GmailSendEmail", {"to": "a@b.c", "subject": "hi"}),
        ),
        ("AmazonViewSavedAddresses\nAction Input: {}", ("AmazonViewSavedAddresses", {})),
        (
            'TerminalExecute Input: {"command": "ls"}',
            ("TerminalExecute", {"command": "ls"}),
        ),
        ("SendMessage: {'user': 'Alice'}", ("SendMessage", {"user": "Alice"})),
        ("DeleteUsers(users=['Alice'])", ("DeleteUsers", {"input": "users=['Alice']"})),
        ("GitHubListRepositories", ("GitHubListRepositories", {})),
        (
            "Certainly! Here are seven sentences.",
            ("reply_to_user", {"message": "Certainly! Here are seven sentences."}),
        ),
        ("Sure", ("reply_to_user", {"message": "Sure"})),
    ],
)
def test_parse_action(text, expected):
    assert parse_action(text) == expected


@pytest.mark.parametrize("text", ["", "None", "   "])
def test_no_action(text):
    assert parse_action(text) is None


def test_unparseable_arguments_are_kept_raw():
    assert parse_action("Tool{not: valid") == ("Tool", {"input": "{not: valid"})
