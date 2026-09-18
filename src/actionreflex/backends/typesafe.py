from __future__ import annotations

from typing import Any

try:
    from typesafe_sdk import AsyncTypeSafeClient, TypeSafeClient
except ImportError as exc:  # pragma: no cover - exercised only when the dep is missing
    raise ImportError(
        "TypeSafeBackend requires the 'typesafe-sdk' package. Install it with "
        "`pip install typesafe-sdk` (it ships with actionreflex's default "
        "dependencies) and set a TYPESAFE_API_KEY."
    ) from exc


class TypeSafeBackend:
    """Evaluates policies against the real TypeSafe Jev API.

    Jev is in early access behind a waitlist as of writing this backend (see
    https://docs.typesafe.ai). This client is verified against `typesafe-sdk`'s
    actual installed types (`TypeSafeClient.system_one`, `Noul`/`Choice`/`Score`,
    `SystemOneResponse.answers`), not just the docs prose - but has not been
    exercised against a live key/response, since Jev access itself is gated.
    Please open an issue (or a PR) with anything that doesn't match once you
    have access.

    `api_key` is optional; if omitted, the SDK falls back to the
    `TYPESAFE_API_KEY` environment variable itself.
    """

    def __init__(self, api_key: str | None = None, model: str = "jev-latest"):
        self._api_key = api_key
        self.model = model

    def evaluate(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        with TypeSafeClient(api_key=self._api_key) as client:
            response = client.system_one(state=state, questions=questions, model=model or self.model)
        return response.answers

    async def aevaluate(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]:
        async with AsyncTypeSafeClient(api_key=self._api_key) as client:
            response = await client.system_one(
                state=state, questions=questions, model=model or self.model
            )
        return response.answers
