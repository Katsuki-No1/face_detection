"""Small compatibility shim for vendored DEIMv2 modules.

The upstream modules decorate classes with ``@register`` for their YAML
workspace. This repository instantiates those classes directly, so registration
only needs to preserve the decorated object.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def register(*args, **_kwargs):
    if args and len(args) == 1 and callable(args[0]):
        return args[0]

    def decorator(obj: T) -> T:
        return obj

    return decorator


__all__ = ["register"]
