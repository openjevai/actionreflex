import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from actionreflex import Action
from actionreflex.gate import _bind_arguments


def test_state_shape():
    state = Action(name="send_email", arguments={"to": "a@b.c"}, context="user asked").to_state()
    assert state == {
        "action": {"name": "send_email", "arguments": {"to": "a@b.c"}},
        "context": "user asked",
    }


def test_context_omitted_when_none():
    assert "context" not in Action(name="noop").to_state()


def test_non_json_arguments_are_converted():
    @dataclass
    class Recipient:
        name: str
        account: uuid.UUID

    account = uuid.UUID("12345678-1234-5678-1234-567812345678")
    state = Action(
        name="transfer",
        arguments={
            "amount": Decimal("10.50"),
            "at": datetime(2026, 9, 18, 12, 0, tzinfo=timezone.utc),
            "to": Recipient(name="Sam", account=account),
            "tags": {"urgent"},
        },
    ).to_state()

    json.dumps(state)  # must be plain JSON now
    args = state["action"]["arguments"]
    assert args["amount"] == "10.50"
    assert args["at"] == "2026-09-18 12:00:00+00:00"
    assert args["to"] == {"name": "Sam", "account": str(account)}
    assert args["tags"] == ["urgent"]


def test_positional_arguments_are_bound_to_parameter_names():
    def delete_record(record_id: int, hard: bool = False): ...

    assert _bind_arguments(delete_record, (42,), {}) == {"record_id": 42, "hard": False}
    assert _bind_arguments(delete_record, (42,), {"hard": True}) == {"record_id": 42, "hard": True}


def test_self_is_not_sent_as_an_argument():
    class Tools:
        def delete_record(self, record_id: int): ...

    tools = Tools()
    assert _bind_arguments(Tools.delete_record, (tools, 7), {}) == {"record_id": 7}
