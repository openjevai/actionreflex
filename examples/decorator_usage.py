"""The `Gate.guard` decorator, for the common case where a tool is a plain
Python function and you want the gate check to happen transparently on every
call. Runs with zero setup - no API key, no network.
"""

from actionreflex import Action, ActionBlocked, Gate
from actionreflex.backends import MockBackend
from actionreflex.policies import destructive_action

gate = Gate(
    policies=[destructive_action(on_trigger="block")],
    backend=MockBackend(responses={"destructive_action": 0.8}),
)


def build_action(customer_id: str) -> Action:
    return Action(name="delete_customer_account", arguments={"customer_id": customer_id})


@gate.guard(action_builder=lambda customer_id: build_action(customer_id))
def delete_customer_account(customer_id: str) -> str:
    return f"deleted {customer_id}"


if __name__ == "__main__":
    try:
        delete_customer_account(customer_id="cust_8123")
    except ActionBlocked as exc:
        print(f"Blocked as expected: {exc}")
