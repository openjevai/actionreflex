from .base import Backend
from .mock import MockBackend

__all__ = ["Backend", "MockBackend", "TypeSafeBackend"]


def __getattr__(name: str):
    # Lazy import so `actionreflex.backends.MockBackend` works even in
    # environments where `typesafe-sdk` (a hard dependency, but still) hasn't
    # finished installing yet, and so the real ImportError message from
    # typesafe.py (not a generic AttributeError) is what users see.
    if name == "TypeSafeBackend":
        from .typesafe import TypeSafeBackend

        return TypeSafeBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
