"""Stable keys and JSON-safe values for lightweight analysis plans."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.image_io import image_fingerprint


GROUPING_PLAN_VERSION = "scene-grouping-v1"
SELECTION_PLAN_VERSION = "whole-frame-selection-v1"


def json_compatible(value: Any) -> Any:
    """Convert configuration/plan values to deterministic JSON primitives."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Enum):
        return json_compatible(value.value)
    if isinstance(value, (Path, os.PathLike)):
        return os.path.normcase(os.path.abspath(os.fspath(value)))
    if is_dataclass(value):
        return json_compatible(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return json_compatible(tolist())
    values = getattr(value, "__dict__", None)
    if isinstance(values, Mapping):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "values": json_compatible({key: item for key, item in values.items() if not key.startswith("_")}),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def ordered_file_fingerprints(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    """Fingerprint files in capture order, retaining canonical paths explicitly."""

    return [
        {
            "path": os.path.normcase(os.path.abspath(os.fspath(path))),
            "fingerprint": image_fingerprint(path),
        }
        for path in paths
    ]


def optional_file_fingerprint(path: str | Path) -> dict[str, Any]:
    canonical = os.path.normcase(os.path.abspath(os.fspath(path)))
    if not Path(path).is_file():
        return {"path": canonical, "exists": False}
    return {"path": canonical, "exists": True, "fingerprint": image_fingerprint(path)}


def build_plan_key(
    kind: str,
    algorithm_version: str,
    paths: Sequence[str | Path],
    effective_config: Any,
    *,
    extra: Any = None,
) -> str:
    payload = {
        "kind": str(kind),
        "algorithm_version": str(algorithm_version),
        "files": ordered_file_fingerprints(paths),
        "effective_config": json_compatible(effective_config),
        "extra": json_compatible(extra),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GROUPING_PLAN_VERSION",
    "SELECTION_PLAN_VERSION",
    "build_plan_key",
    "json_compatible",
    "optional_file_fingerprint",
    "ordered_file_fingerprints",
]
