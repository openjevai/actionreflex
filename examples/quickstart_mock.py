"""Runs with zero setup - no API key, no network. `python examples/quickstart_mock.py`

Shows the explicit `Gate.check()` flow, which is the shape you'd wire into a
LangGraph node, a custom agent loop, or anywhere else you intercept a tool
call before it executes.
"""

from actionreflex import Action, Gate
from actionreflex.backends import MockBackend
from actionreflex.policies import destructive_action, intent_mismatch

gate = Gate(
    policies=[destructive_action(), intent_mismatch()],
    # Script the mock so this demo actually triggers something. Swap
    # MockBackend() for actionreflex.backends.TypeSafeBackend() to use the
    # real Jev model instead.
    backend=MockBackend(responses={"destructive_action": 0.93, "intent_mismatch": 0.1}),
)

action = Action(
    name="delete_customer_account",
    arguments={"customer_id": "cust_8123"},
    context="User asked: 'can you check why my last order didn't ship?'",
)

verdict = gate.check(action)

print(verdict)
for result in verdict.results:
    status = "TRIGGERED" if result.triggered else "ok"
    print(f"  [{status}] {result.policy_id}: {result.reason or '-'}")

if not verdict.allowed:
    print(f"\n-> {verdict.decision.upper()}: not executing '{action.name}'")
