# actionreflex

A pre-execution gate for AI agent actions, powered by [TypeSafe](https://typesafe.ai)'s
System One model, **Jev**.

## The idea

Most agent guardrails today are built out of LLM calls: ask a model "is this tool call
safe?", wait a few hundred milliseconds to a couple of seconds, pay a real per-check
cost, and hope it's cheap enough to run on more than a sampled subset of what your
agent actually does.

Jev is a different kind of model. It doesn't generate text - it answers narrow, typed
questions (yes/no, pick-one, rate-on-a-scale) with a calibrated probability, in
**70-500ms**, for **$0.042 per million input tokens** with free output (it's a single
forward pass, not a generation loop). That's roughly 1/60th the input cost of GPT-4o
for a call an order of magnitude faster.

That profile is a good match for a specific job: checking *every* action an agent
proposes, not just the ones you can afford to check. `actionreflex` wraps that
capability as a small, framework-agnostic gate you drop in front of tool execution:

```python
from actionreflex import Action, Gate
from actionreflex.backends import TypeSafeBackend
from actionreflex.policies import destructive_action, intent_mismatch

gate = Gate(
    policies=[destructive_action(), intent_mismatch()],
    backend=TypeSafeBackend(),  # reads TYPESAFE_API_KEY from the environment
)

verdict = gate.check(Action(
    name="delete_customer_account",
    arguments={"customer_id": "cust_8123"},
    context="User asked: 'can you check why my last order didn't ship?'",
))

if not verdict.allowed:
    print(verdict.decision, [r.reason for r in verdict.triggered_results])
    # -> 'block', ['action does not match the user's stated intent']
```

Every policy in a `Gate` is asked in the *same* TypeSafe request - they run in
parallel, so adding more checks costs latency in the tens-of-ms range, not a
multiplied number of round trips.

> **Status:** Jev is in early access behind a waitlist as of September 2026 (see
> [docs.typesafe.ai](https://docs.typesafe.ai)). This library is written against the
> documented HTTP API and the `typesafe-sdk` package's actual installed types, but the
> `TypeSafeBackend` integration hasn't been exercised against a live key/response yet -
> only the offline `MockBackend` path is verified end-to-end so far. Issues and PRs
> from anyone with access are very welcome.

## Install

```bash
pip install actionreflex
```

(Not published to PyPI yet - for now, install from source: `pip install -e .`)

## Quickstart (no API key needed)

Everything below runs fully offline against `MockBackend`, so you can try the shape
of the library before you have Jev access:

```bash
python examples/quickstart_mock.py
python examples/decorator_usage.py
```

## Two ways to use it

**1. Explicit check** - call this wherever you already intercept a tool call: a
LangGraph node, a custom agent loop, a middleware layer.

```python
verdict = gate.check(action)
if verdict.decision == "block":
    ...
elif verdict.decision == "escalate":
    ...  # hand off to a human/slower review path
else:
    execute(action)
```

`gate.acheck(action)` is the async equivalent.

**2. Decorator** - for the common case where a tool is a plain Python function:

```python
@gate.guard()
def delete_customer_account(customer_id: str) -> str:
    ...
```

By default this raises `ActionBlocked` / `ActionEscalated`; pass `on_block=` /
`on_escalate=` handlers to do something else instead (log it, return a fixed
response, queue it for human review).

## Policies

A `Policy` is one typed question sent to Jev, plus a `judge` function that turns its
raw answer into triggered/not-triggered, plus what a trigger does to the verdict:

```python
from actionreflex import Policy, noul_above
from typesafe_sdk import Noul

Policy(
    id="destructive_action",
    question=Noul(
        instructions="Would executing this action be irreversible or destructive?",
        criteria={"true": "irreversible or destructive", "false": "safe or reversible"},
    ),
    judge=noul_above(0.5),
    on_trigger="escalate",  # "block" | "escalate" | "warn"
)
```

`actionreflex.policies` ships a handful of ready-made ones to start from or copy:

| Policy | Checks | Default |
| --- | --- | --- |
| `destructive_action()` | Is this irreversible/destructive? | escalate |
| `intent_mismatch()` | Does this match what the user actually asked for? | block |
| `contains_pii()` | Do the arguments carry unneeded personal data? | warn |
| `risk_tier()` | Low/medium/high risk classification | escalate on `high` |

Writing a new one is meant to be this cheap - the built-ins are a starting point, not
a fixed taxonomy.

## Testing without API access

`MockBackend` is a deterministic, offline stand-in for `TypeSafeBackend` - no network,
no key. It returns a fixed "no"/first-option answer for every question by default -
whether that trips a given policy depends on the policy's judge (e.g. it won't trigger
`destructive_action`'s above-threshold check, but it *will* trigger `intent_mismatch`'s
below-threshold one). Script specific scenarios explicitly with `responses`, keyed by
policy id:

```python
from actionreflex.backends import MockBackend

MockBackend(responses={"destructive_action": 0.9, "risk_tier": "high"})
```

This is what the whole test suite (`pytest`) runs against, and it's a reasonable way
to unit-test policy logic in your own app without spending real Jev calls on CI.

## Development

```bash
pip install -e ".[dev]"
pytest -q         # runs entirely offline against MockBackend
ruff check .
```

## Why not just use an LLM for this?

You can, and plenty of guardrail tools do. The tradeoff `actionreflex` is built around:
Jev is much cheaper and much faster than a generative LLM call, and it gives you a
calibrated probability instead of a free-text verdict you have to parse - but it can't
write anything (no explanations beyond what you template from the typed answer, no
open-ended reasoning). That's a good trade for "should this specific action execute,"
asked on every action, and a bad trade for anything that needs the model to actually
write or reason at length. Use it for the narrow, high-volume judgment calls; keep an
LLM (or a human) in the loop for the rest.

## License

MIT - see [LICENSE](LICENSE).
