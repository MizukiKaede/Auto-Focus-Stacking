"""Human/natural ordering for camera filenames and paths."""

from __future__ import annotations

import re
from os import fspath
from typing import Iterable, TypeVar


_TOKEN_RE = re.compile(r"(\d+)")
T = TypeVar("T")


def natural_sort_key(value: str | object) -> tuple[tuple[int, object], ...]:
    """Return a case-insensitive key where numeric runs compare numerically.

    ``DSC9.JPG`` therefore sorts before ``DSC10.JPG``.  A tuple marker keeps
    numeric and text chunks comparable on all supported Python versions.
    """

    text = fspath(value) if hasattr(value, "__fspath__") else str(value)
    chunks = _TOKEN_RE.split(text.casefold())
    result: list[tuple[int, object]] = []
    for chunk in chunks:
        if chunk.isdigit():
            result.append((1, int(chunk)))
        else:
            result.append((0, chunk))
    return tuple(result)


def natural_sorted(values: Iterable[T], *, key=None, reverse: bool = False) -> list[T]:
    """Sort an iterable with a natural filename key."""

    if key is None:
        return sorted(values, key=natural_sort_key, reverse=reverse)
    return sorted(values, key=lambda item: natural_sort_key(key(item)), reverse=reverse)


# Common aliases used by third-party integrations and older prototypes.
natural_key = natural_sort_key
natsorted = natural_sorted

__all__ = ["natural_sort_key", "natural_key", "natural_sorted", "natsorted"]

