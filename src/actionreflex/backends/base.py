from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    """A pluggable evaluator: turns (state, questions) into answers keyed by question id.

    `questions` maps a policy id to a `typesafe_sdk.Noul | Choice | Score` instance.
    The returned dict must map the same ids to their corresponding answer objects
    (the ones the `typesafe-sdk` client itself returns, e.g. `response.nouls["x"]`).
    Implement both methods; a sync-only backend can implement `evaluate` and leave
    `aevaluate` as a thin wrapper (see `MockBackend` for the pattern).
    """

    def evaluate(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]: ...

    async def aevaluate(
        self, state: Any, questions: dict[str, Any], model: str | None = None
    ) -> dict[str, Any]: ...
