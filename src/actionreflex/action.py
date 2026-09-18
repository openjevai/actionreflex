from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """A proposed agent action (typically a tool/function call) awaiting a decision.

    `name` and `arguments` describe the call itself. `context` carries whatever the
    policies need to judge it: the user's original request, recent conversation
    turns, the agent's stated plan, session metadata. It's sent to Jev as part of
    the evaluation `state`, so include what a judgment actually needs and no more -
    Jev is billed per input token.

    `context` may be a string or any JSON-shaped value. Structured context with
    named fields tends to work better than one long string when it has several
    parts, e.g. `{"user_request": ..., "recent_turns": [...], "user_role": ...}`.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    context: Any = None

    def to_state(self) -> dict[str, Any]:
        """Build the JSON `state` payload sent to TypeSafe for this action.

        Values that aren't natively JSON (datetimes, UUIDs, Decimals, dataclasses,
        pydantic models, ...) are converted rather than failing the request.
        """
        state: dict[str, Any] = {"action": {"name": self.name, "arguments": self.arguments}}
        if self.context is not None:
            state["context"] = self.context
        return _to_json(state)


def _to_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_fallback))


def _fallback(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return str(obj)
