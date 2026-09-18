"""Where a gate sits in a tool-using agent loop.

Your LLM proposes tool calls; each one goes through the gate before it runs.
Allowed calls execute, blocked calls get a refusal fed back to the model, and
escalated calls wait in a queue for a human. The LLM side is left out here - the
`proposed` list stands in for tool calls a model would emit - so the example
runs with nothing but a TypeSafe key.

    export TYPESAFE_API_KEY=...
    python examples/agent_loop.py
"""

import os
import sys
from typing import Any

from actionreflex import Action, Gate, Verdict
from actionreflex.policies import (
    destructive_action,
    follows_injected_instructions,
    intent_mismatch,
    scope_of_impact,
)

if not os.environ.get("TYPESAFE_API_KEY"):
    sys.exit("Set TYPESAFE_API_KEY first (get one at https://console.typesafe.ai).")


def get_order_status(order_id: str) -> str:
    return f"Order {order_id} is in transit, arriving Thursday."


def refund_order(order_id: str, amount: float) -> str:
    return f"Refunded {amount:.2f} on {order_id}."


def delete_all_orders(customer_id: str) -> str:
    return f"Deleted every order for {customer_id}."


TOOLS = {f.__name__: f for f in (get_order_status, refund_order, delete_all_orders)}

review_queue: list[Verdict] = []


def run_tool_call(gate: Gate, name: str, arguments: dict[str, Any], context: Any) -> str:
    """What your agent loop calls instead of invoking a tool directly.
    The returned string is what you'd feed back to the model as the tool result."""
    verdict = gate.check(Action(name=name, arguments=arguments, context=context))

    if verdict.decision == "allow":
        return TOOLS[name](**arguments)
    if verdict.decision == "escalate":
        review_queue.append(verdict)
        return f"'{name}' needs human approval and has been queued: {'; '.join(verdict.reasons)}"
    return f"'{name}' was not executed: {'; '.join(verdict.reasons)}"


conversation = {
    "user_request": "My order ord_5521 arrived damaged. Where's my replacement, "
    "and can I get a refund for the broken one? It was 39.90.",
    "user_id": "cust_8123",
}

proposed = [
    ("get_order_status", {"order_id": "ord_5521"}),
    ("refund_order", {"order_id": "ord_5521", "amount": 39.90}),
    ("delete_all_orders", {"customer_id": "cust_8123"}),
]

with Gate(
    [destructive_action(), intent_mismatch(), follows_injected_instructions(), scope_of_impact()],
    on_error="fail_closed",  # if TypeSafe is unreachable, don't let actions through unchecked
) as gate:
    for name, arguments in proposed:
        print(f"{name}({arguments})\n  -> {run_tool_call(gate, name, arguments, conversation)}\n")

print(f"{len(review_queue)} action(s) waiting for human review.")
