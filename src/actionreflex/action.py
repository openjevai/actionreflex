from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Action:
    """A proposed agent action (typically a tool/function call) awaiting a gate decision.

    `name` and `arguments` describe the call itself. `context` carries whatever the
    policies need to judge it against: the user's original request, recent conversation
    turns, the agent's stated plan, session metadata, etc. It is passed to TypeSafe as
    part of the evaluation `state`, so keep it to what a judgment actually needs -
    Jev's cost scales with input tokens.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    context: Any = None

    def to_state(self) -> dict[str, Any]:
        """Build the `state` payload sent to TypeSafe for this action."""
        return {
            "action": {"name": self.name, "arguments": self.arguments},
            "context": self.context,
        }
