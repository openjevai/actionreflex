# OpenJEV support

This fork of [actionreflex](https://github.com/eyenpi/actionreflex) adds **optional** support for
[OpenJEV](https://openjev.sh), a free community gateway to the same Jev model built by
[TypeSafe](https://typesafe.ai). TypeSafe remains the default; OpenJEV is purely additive.

## What changed

- `src/actionreflex/gate.py`
  - Added a `provider` keyword argument to `Gate` and `AsyncGate` (`"typesafe"` or `"openjev"`).
  - Added OpenJEV constants and helper functions (`_resolve_provider`, `_client_kwargs`,
    `_default_model_for`).
  - When OpenJEV is selected, the gate builds its `typesafe_sdk` client with
    `base_url="https://api.openjev.sh"`, sends model id `"openjev"`, and reads the key from
    `OPENJEV_API_KEY`.
  - TypeSafe behaviour is unchanged when its key is present or `provider` is unset.
- `README.md` — short OpenJEV note after the intro and in the install/keys section.
- `pyproject.toml` — added `openjev` keyword.
- `tests/test_gate.py` — `test_missing_api_key_fails_at_construction` now also clears
  `OPENJEV_API_KEY` and `JEV_PROVIDER`, so it still asserts "no key at all → fail"
  in environments where an OpenJEV key happens to be set.

## Provider selection rule

1. An explicit `provider` argument (or `JEV_PROVIDER` env var) always wins.
2. Otherwise, if a TypeSafe key is available (`TYPESAFE_API_KEY` or a passed `api_key`),
   TypeSafe is used — the original default, unchanged.
3. Otherwise, if only `OPENJEV_API_KEY` is set, OpenJEV is used.

Anyone with a TypeSafe key sees zero behaviour change.

## How to configure

TypeSafe (default, unchanged):

```bash
export TYPESAFE_API_KEY=...
```

OpenJEV (explicit):

```bash
export OPENJEV_API_KEY=...
export JEV_PROVIDER=openjev   # optional when TYPESAFE_API_KEY is unset
```

Or in code:

```python
from actionreflex import Gate

gate = Gate(policies, provider="openjev")
```

The `typesafe_sdk` default retry policy already covers HTTP 500–599 (including 503 and 529),
so no retry-status change was needed.

## How it was verified

A live `POST https://api.openjev.sh/v1/systemone` request was made with the OpenJEV key,
model `openjev`, state `ping`, and one `noul` question; it returned HTTP 200. A grep confirms
no hardcoded `api.typesafe.ai` default was introduced — TypeSafe stays the SDK's default base
URL, and OpenJEV is only selected by explicit choice or key fallback.

## Upstream

Original project: https://github.com/eyenpi/actionreflex by @eyenpi (Ali Nabipour), MIT license.
