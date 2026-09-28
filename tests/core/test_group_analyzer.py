"""Integration tests for the streaming group-analysis boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from focus_stack_app.core.coverage_selector import CoverageConfig
from focus_stack_app.core.focus_cluster import FocusClusterConfig
from focus_stack_app.core.focus_map import FocusMapConfig
from focus_stack_app.core.group_analyzer import (
    CACHE_ALGORITHM_VERSION,
    GroupAnalyzer,
    GroupAnalyzerConfig,
    _focus_map_cache_namespace,
    _load_or_create_cached_map,
)
from focus_stack_app.core.types import SceneGroup
from focus_stack_app.storage.cache import DiskCache
from focus_stack_app.storage.models import ImageRecord


class _RecordingRepository:
    """Small repository double that exercises the persistence adapter."""

    def __init__(self) -> None:
        self.group_statuses = []
        self.image_groups = []
        self.analyses = []

    def set_group_status(self, group_id, status, **values):
        self.group_statuses.append((group_id, status, values))

    def set_image_group(self, image_id, group_id, *, status=None):
        self.image_groups.append((image_id, group_id, status))

    def upsert_analysis(self, record):
        self.analyses.append(record)


class _FakeFocusMemory:
    """Minimal bounded-cache shape used to exercise namespaced cache IO."""

    def __init__(self) -> None:
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def put(self, key, value):
        self.values[key] = value

    def delete(self, key):
        self.values.pop(key, None)


class _FakeFocusCache:
    """Path/bytes cache double; no source fingerprint implementation needed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.focus_memory = _FakeFocusMemory()
        self.reads = []
        self.writes = []

    def focus_map_path(self, image_path, *, suffix=".npz"):
        # This fixed stem stands in for DiskCache's source fingerprint.  The
        # analyzer must retain it while adding the algorithm namespace.
        return self.root / f"source-fingerprint{suffix}"

    def read_bytes(self, path):
        self.reads.append(Path(path))
        try:
            return Path(path).read_bytes()
        except OSError:
            return None

    def _write_atomic(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.writes.append(path)


def _write_pattern(path: Path, variant: int) -> None:
    """Write a small JPEG with focus evidence in a different region."""

    Image = pytest.importorskip("PIL.Image")
    height, width = 72, 108
    y, x = np.mgrid[:height, :width]
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    image[..., 0] = ((x * 2 + y + variant * 13) % 220).astype(np.uint8)
    image[..., 1] = ((y * 3 + variant * 17) % 220).astype(np.uint8)
    image[..., 2] = 90
    if variant % 2:
        image[:, : width // 3, :] = (
            (x[:, : width // 3, None] * 7 + y[:, : width // 3, None] * 3) % 255
        ).astype(np.uint8)
    else:
        image[:, -width // 3 :, :] = (
            (x[:, -width // 3 :, None] * 7 + y[:, -width // 3 :, None] * 3) % 255
        ).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=95)


def _analyzer(tmp_path: Path, repository: _RecordingRepository) -> GroupAnalyzer:
    config = GroupAnalyzerConfig(
        analysis_long_edge=96,
        output_long_edge=32,
        duplicate_focus_threshold=0.96,
        coverage_target=0.90,
        min_coverage_gain=0.0,
    )
    # Keep this test focused on the end-to-end boundary: the deliberately
    # strict cluster threshold ensures the distinct synthetic patterns reach
    # maximum coverage even if JPEG/OpenCV versions score them differently.
    return GroupAnalyzer(
        config,
        database=repository,
        cache=DiskCache(tmp_path),
        cluster_config=FocusClusterConfig(similarity_threshold=1.1),
        coverage_config=CoverageConfig(
            coverage_target=0.90, min_coverage_gain=0.0, transition_reorder=True,
        ),
    )


def test_focus_map_cache_namespace_invalidates_config_and_reuses_same_fake_cache(tmp_path):
    source = tmp_path / "source.jpg"
    config_a = FocusMapConfig(
        analysis_long_edge=1600, output_long_edge=512, output_dtype="float16",
        scales=(1.0, 2.5, 5.0), saturation_penalty=0.35,
    )
    config_same = FocusMapConfig(
        analysis_long_edge=1600, output_long_edge=512, output_dtype="float16",
        scales=[1, 2.5, 5], saturation_penalty=0.35,
    )
    config_b = FocusMapConfig(output_long_edge=256)
    namespace_a = _focus_map_cache_namespace(config_a)
    namespace_same = _focus_map_cache_namespace(config_same)
    namespace_b = _focus_map_cache_namespace(config_b)

    assert namespace_a == namespace_same
    assert namespace_a != namespace_b
    assert namespace_a == _focus_map_cache_namespace(GroupAnalyzerConfig(output_long_edge=512))
    assert namespace_a != _focus_map_cache_namespace(GroupAnalyzerConfig(output_long_edge=256))
    assert CACHE_ALGORITHM_VERSION

    cache = _FakeFocusCache(tmp_path / "maps")
    calls = []

    def create(value):
        def _create():
            calls.append(value)
            return np.full((4, 5), value, dtype=np.float16)
        return _create

    first, path_a = _load_or_create_cached_map(
        cache, source, create(1), namespace=namespace_a,
    )
    memory_hit, same_path = _load_or_create_cached_map(
        cache, source, create(2), namespace=namespace_same,
    )
    disk_hit, disk_path = _load_or_create_cached_map(
        _FakeFocusCache(tmp_path / "maps"), source, create(3), namespace=namespace_a,
    )
    changed, path_b = _load_or_create_cached_map(
        cache, source, create(4), namespace=namespace_b,
    )

    assert calls == [1, 4]
    assert path_a == same_path == disk_path
    assert path_b != path_a
    assert path_a.name.startswith("source-fingerprint.")
    assert path_a.name.endswith(f"{namespace_a}.npz")
    assert path_b.name.endswith(f"{namespace_b}.npz")
    np.testing.assert_array_equal(first, memory_hit)
    np.testing.assert_array_equal(first, disk_hit)
    np.testing.assert_array_equal(changed, np.full((4, 5), 4, dtype=np.float16))
    assert len(cache.writes) == 2


def test_group_analyzer_accepts_synthetic_loader_without_pillow_or_opencv():
    arrays = [
        np.pad(np.ones((32, 12), np.uint8) * 220, ((0, 0), (index * 10, 20 - index * 10)))
        for index in range(3)
    ]
    records = [ImageRecord.from_path(f"C:/synthetic/frame_{index}.jpg", id=index + 1)
               for index in range(len(arrays))]
    calls = []

    def loader(item, **kwargs):
        calls.append((item.id, kwargs["max_long_edge"]))
        return arrays[item.id - 1]

    result = GroupAnalyzer(
        GroupAnalyzerConfig(analysis_long_edge=64, output_long_edge=24, min_coverage_gain=0.0),
        loader=loader,
        cluster_config=FocusClusterConfig(similarity_threshold=1.1),
    ).analyze_group(SceneGroup(group_id=16, items=records))

    assert calls[:3] == [(1, 1280), (2, 1280), (3, 1280)]
    assert {image_id for image_id, _ in calls} == {1, 2, 3}
    assert result["image_count"] == 3
    assert result["selected_count"] >= 1
    assert result["preview_reference_index"] in result["selected_indices"]
    assert all("quality" in row and "gain" in row and "reason" in row
               for row in result["per_image"])


def test_group_analyzer_streams_jpegs_and_returns_merge_boundary(tmp_path):
    paths = [tmp_path / f"frame_{index}.jpg" for index in range(3)]
    for index, path in enumerate(paths):
        _write_pattern(path, index)
    records = [ImageRecord.from_path(path, id=index + 1, sequence_index=index) for index, path in enumerate(paths)]
    group = SceneGroup(group_id=17, items=records, start_index=0, end_index=2, confidence=0.91)
    repository = _RecordingRepository()

    result = _analyzer(tmp_path / "cache", repository).analyze_group(group)

    assert result["group_id"] == 17
    assert result["all_images"] == records
    assert result["first_original_image"] is records[0]
    assert result["selected_count"] == len(result["selected_indices"])
    assert result["selected_count"] >= 1
    assert result["selected_paths"]
    assert len(result["per_image"]) == len(records)
    assert all("quality_score" in row and "coverage_gain" in row and "selection_reason" in row
               for row in result["per_image"])
    # The restored whole-frame selector keeps its decision maps in memory for
    # one group and does not publish the newer generic focus-map cache files.
    assert result["focus_map_paths"] == [None] * len(records)
    assert result["merge_status"] in {"READY_FOR_MERGE", "NO_MERGE_REPEATED", "NO_MERGE_TOO_SMALL"}
    assert repository.group_statuses[0][1] == "ANALYZING_FOCUS"
    assert repository.group_statuses[-1][1] in {"SELECTED", "CLASSIFIED"}
    assert len(repository.analyses) == len(records)


def test_repeated_focus_is_archive_only_and_direct_record_is_singleton(tmp_path):
    path = tmp_path / "same.jpg"
    _write_pattern(path, 1)
    first = ImageRecord.from_path(path, id=21, sequence_index=0)
    second = ImageRecord.from_path(path, id=22, sequence_index=1)
    group = SceneGroup(group_id=22, items=[first, second], start_index=0, end_index=1, confidence=0.98)

    result = GroupAnalyzer(
        GroupAnalyzerConfig(
            analysis_long_edge=96,
            output_long_edge=24,
            min_coverage_gain=0.0,
            minimum_stack_group_size=2,
            # Exact duplicate inputs are the deliberate opt-out for this
            # regression; production defaults retain two confirmed frames.
            minimum_stack_images=1,
        ),
    ).analyze_group(group)

    assert result["selected_count"] == 1
    assert result["merge_status"] == "NO_MERGE_REPEATED"
    assert result["needs_merge"] is False
    assert len(result["all_images"]) == 2

    singleton = GroupAnalyzer(
        GroupAnalyzerConfig(analysis_long_edge=96, output_long_edge=24),
    ).analyze_group(first)
    assert singleton["image_count"] == 1
    assert singleton["merge_status"] == "NO_MERGE_SINGLE"
    assert singleton["first_original_image"] is first
    assert singleton["needs_merge"] is False

