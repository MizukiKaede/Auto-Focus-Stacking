"""Apply source-local filename groups before automatic scene detection."""

from __future__ import annotations

import json
from pathlib import Path

from .types import SceneGroup


def group_with_overrides(records, config_path: Path, detect):
    if not config_path.is_file():
        return detect(records)
    definitions = json.loads(config_path.read_text(encoding="utf-8"))["groups"]
    membership = {}
    for index, names in enumerate(definitions):
        if not isinstance(names, list) or not names:
            raise ValueError("Each manual group must contain filenames")
        for name in names:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Manual groups require plain filenames")
            key = name.casefold()
            if key in membership:
                raise ValueError(f"Duplicate manual group filename: {name}")
            membership[key] = index

    manual = {}
    automatic = []
    groups = []
    positions = {id(record): i for i, record in enumerate(records)}

    def flush():
        if automatic:
            groups.extend(detect(list(automatic)))
            automatic.clear()

    for record in records:
        key = membership.get(Path(record.path).name.casefold())
        if key is None:
            automatic.append(record)
        else:
            flush()
            manual.setdefault(key, []).append(record)
    flush()
    for items in manual.values():
        groups.append(SceneGroup(0, items, confidence=1.0))
    groups.sort(key=lambda group: min(positions[id(item)] for item in group.items))
    for index, group in enumerate(groups, 1):
        group.group_id = index
        group.start_index = min(positions[id(item)] for item in group.items)
        group.end_index = max(positions[id(item)] for item in group.items)
    return groups

