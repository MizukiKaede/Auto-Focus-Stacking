"""Apply source-local filename groups before automatic scene detection."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path, PureWindowsPath

from .types import SceneGroup


def group_with_overrides(records, config_path: Path, detect):
    if not config_path.is_file():
        return detect(records)
    definitions = json.loads(config_path.read_text(encoding="utf-8"))["groups"]
    if not isinstance(definitions, list):
        raise ValueError("手动分组 groups 必须是数组")
    root = config_path.parent.resolve()
    candidates = defaultdict(list)
    relative = {}
    for record in records:
        path = Path(getattr(record, "original_path", None) or record.path).resolve()
        display = path.relative_to(root).as_posix()
        key = display.casefold()
        relative[key] = record
        candidates[path.name.casefold()].append(display)

    membership = {}
    declared = set()
    missing = []
    for index, names in enumerate(definitions):
        if not isinstance(names, list) or not names:
            raise ValueError("Each manual group must contain filenames")
        for name in names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("手动分组需要文件名或相对路径")
            item = PureWindowsPath(name)
            if item.anchor or ".." in item.parts:
                raise ValueError(f"手动分组路径必须位于源目录内：{name}")
            key = item.as_posix().casefold()
            if key in declared:
                raise ValueError(f"Duplicate manual group filename: {name}")
            declared.add(key)
            if "/" not in name and "\\" not in name:
                matches = candidates[key]
                if len(matches) > 1:
                    raise ValueError(f"分组文件名 {name} 有多个匹配：{', '.join(matches)}")
                if matches:
                    key = matches[0].casefold()
            if key not in relative:
                missing.append(f"组 {index + 1}: {name}")
                continue
            identity = id(relative[key])
            if identity in membership:
                raise ValueError(f"Duplicate manual group filename: {name}")
            membership[identity] = index
    if missing:
        raise ValueError("分组文件缺失：" + ", ".join(missing))

    manual = {}
    automatic = []
    groups = []
    positions = {id(record): i for i, record in enumerate(records)}

    def flush():
        if automatic:
            groups.extend(detect(list(automatic)))
            automatic.clear()

    for record in records:
        key = membership.get(id(record))
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
