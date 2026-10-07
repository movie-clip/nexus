"""Name -> graph builder registry.

A pipeline module registers an uncompiled ``StateGraph`` builder; the runner
compiles it with whatever checkpointer the run needs.
"""

from collections.abc import Callable

from langgraph.graph import StateGraph

Builder = Callable[[], StateGraph]

_builders: dict[str, Builder] = {}


def register(name: str) -> Callable[[Builder], Builder]:
    def decorator(builder: Builder) -> Builder:
        if name in _builders:
            raise ValueError(f"pipeline {name!r} is already registered")
        _builders[name] = builder
        return builder

    return decorator


def get(name: str) -> Builder:
    try:
        return _builders[name]
    except KeyError:
        known = ", ".join(sorted(_builders)) or "none"
        raise KeyError(f"unknown pipeline {name!r} (known: {known})") from None


def names() -> list[str]:
    return sorted(_builders)
