import json
from pathlib import Path
from types import SimpleNamespace

import tempfile
import unittest

from focus_stack_app.core.manual_groups import group_with_overrides
from focus_stack_app.core.types import SceneGroup


def check_manual_groups_override_splits_and_keep_other_images(tmp_path):
    config = tmp_path / "stack_groups.json"
    config.write_text(json.dumps({"groups": [["b.JPG", "c.JPG"], ["d.JPG", "e.JPG"]]}))
    records = [SimpleNamespace(path=tmp_path / f"{name}.JPG") for name in "abcdefg"]
    calls = []

    def detect(items):
        calls.append(items)
        return [SceneGroup(99, items)]

    groups = group_with_overrides(records, config, detect)
    assert [[item.path.stem for item in group.items] for group in groups] == [
        ["a"], ["b", "c"], ["d", "e"], ["f", "g"],
    ]
    assert [[item.path.stem for item in items] for items in calls] == [["a"], ["f", "g"]]
    assert [(g.group_id, g.start_index, g.end_index) for g in groups] == [
        (1, 0, 0), (2, 1, 2), (3, 3, 4), (4, 5, 6),
    ]


def check_no_config_preserves_detector_result(tmp_path):
    result = [SceneGroup(42, [])]
    assert group_with_overrides([], tmp_path / "missing.json", lambda _: result) is result


def check_overlapping_manual_groups_fail(tmp_path):
    config = tmp_path / "stack_groups.json"
    config.write_text(json.dumps({"groups": [["a.JPG"], ["A.jpg"]]}))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Duplicate"):
        group_with_overrides([], config, lambda _: [])


class ManualGroupsTests(unittest.TestCase):
    def test_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            check_manual_groups_override_splits_and_keep_other_images(Path(directory))

    def test_no_config(self):
        with tempfile.TemporaryDirectory() as directory:
            check_no_config_preserves_detector_result(Path(directory))

    def test_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            check_overlapping_manual_groups_fail(Path(directory))
