"""`Gate.guard`: check a plain Python tool on every call.

export TYPESAFE_API_KEY=...
python examples/decorator_usage.py
"""

import os
import sys

from actionreflex import Action, ActionBlocked, Gate
from actionreflex.policies import destructive_action, intent_mismatch

if not os.environ.get("TYPESAFE_API_KEY"):
    sys.exit("Set TYPESAFE_API_KEY first (get one at https://console.typesafe.ai).")

gate = Gate([destructive_action(on_trigger="block"), intent_mismatch()])

# Whatever your agent knows about the current turn. In a real app this would come
# from your conversation state rather than a module-level variable.
current_user_request = "Why hasn't my last order shipped yet?"


def as_action(customer_id: str, hard_delete: bool = False) -> Action:
    return Action(
        name="delete_customer_account",
        arguments={"customer_id": customer_id, "hard_delete": hard_delete},
        context={"user_request": current_user_request},
    )


@gate.guard(action_builder=as_action)
def delete_customer_account(customer_id: str, hard_delete: bool = False) -> str:
    return f"deleted {customer_id}"


# Without an action_builder, the Action is built from the function's own name and
# its bound arguments - here: Action("lookup_order", {"order_id": "ord_5521"}).
# That action carries no user request, so an intent check would have nothing to
# compare against; `policies=` narrows this tool to the checks that make sense.
@gate.guard(policies=[destructive_action()])
def lookup_order(order_id: str) -> str:
    return f"order {order_id}: in transit"


try:
    print(lookup_order("ord_5521"))
    delete_customer_account("cust_8123", hard_delete=True)
except ActionBlocked as exc:
    print(f"Blocked: {exc}")
finally:
    gate.close()
