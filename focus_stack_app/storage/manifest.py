"""Portable CSV/JSON manifest export for completed and failed stack jobs."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields, is_dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable, Mapping, Sequence


MANIFEST_FIELDS: tuple[str, ...] = (
    "group_id",
    "first_image",
    "image",
    "selected",
    "selection_reason",
    "quality_score",
    "coverage_gain",
    "group_coverage",
    "merge_output",
    "status",
    "capture_order",
    "alignment_order",
    "actual_hugin_input_order",
    "alignment_order_confidence",
    "preview_reference",
    "fusion_backend",
    "requested_backend",
    "actual_backend",
    "alignment_level",
    "alignment_status",
    "crop_ratio",
    "fallback_used",
    "diagnostics",
)


@dataclass
class ManifestRow:
    group_id: int | str
    first_image: str = ""
    image: str = ""
    selected: bool = False
    selection_reason: str = ""
    quality_score: float | None = None
    coverage_gain: float | None = None
    group_coverage: float | None = None
    merge_output: str = ""
    status: str = ""
    capture_order: Any = ""
    alignment_order: Any = ""
    actual_hugin_input_order: Any = ""
    alignment_order_confidence: float | None = None
    preview_reference: str = ""
    fusion_backend: str = ""
    requested_backend: str = ""
    actual_backend: str = ""
    alignment_level: int | None = None
    alignment_status: str = ""
    crop_ratio: float | None = None
    fallback_used: bool = False
    diagnostics: Any = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # CSV booleans are intentionally lower-case so the manifest is easy to
        # consume from scripts and matches the project examples.
        data["selected"] = "true" if bool(data["selected"]) else "false"
        return data


def _as_mapping(row: ManifestRow | Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(row, ManifestRow):
        return row.as_dict()
    if isinstance(row, Mapping):
        return dict(row)
    if is_dataclass(row):
        return asdict(row)
    if hasattr(row, "__dict__"):
        return dict(vars(row))
    raise TypeError(f"Unsupported manifest row: {type(row)!r}")


def _normalise_row(row: ManifestRow | Mapping[str, Any] | Any) -> dict[str, Any]:
    source = _as_mapping(row)
    # Support common internal names without coupling to the algorithm agent's
    # model classes.
    aliases = {
        "first_original": "first_image",
        "first_original_image": "first_image",
        "filename": "image",
        "output_path": "merge_output",
        "coverage": "group_coverage",
        "reason": "selection_reason",
    }
    for old, new in aliases.items():
        if new not in source and old in source:
            source[new] = source[old]
    result = {key: source.get(key, "") for key in MANIFEST_FIELDS}
    if isinstance(result["selected"], bool):
        result["selected"] = "true" if result["selected"] else "false"
    elif result["selected"] is None:
        result["selected"] = "false"
    if isinstance(result.get("fallback_used"), bool):
        result["fallback_used"] = "true" if result["fallback_used"] else "false"
    for name in ("capture_order", "alignment_order", "actual_hugin_input_order", "diagnostics"):
        if isinstance(result.get(name), (list, tuple, dict)):
            result[name] = json.dumps(result[name], ensure_ascii=False, separators=(",", ":"))
    return result


def _atomic_path(path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    return Path(name), fd


class ManifestWriter:
    """Write stack rows atomically and keep output easy to inspect."""

    def __init__(self, path: os.PathLike[str] | str | None = None):
        self.path = Path(path) if path is not None else None
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.RLock()

    def rows_from_groups(self, groups: Iterable[Any]) -> list[dict[str, Any]]:
        """Expand group/image objects into manifest rows.

        This convenience method understands dictionaries, dataclasses and the
        common ``group.images``/``group.selected_images`` object shape.  A
        group with no image list still produces one row for diagnostics.
        """

        rows: list[dict[str, Any]] = []
        for group in groups:
            gm = _as_mapping(group)
            gid = gm.get("group_id", gm.get("id", ""))
            first = gm.get("first_image", gm.get("first_original_image", gm.get("first_original", "")))
            if hasattr(first, "filename"):
                first = getattr(first, "filename")
            output = gm.get("merge_output", gm.get("output_path", ""))
            status = gm.get("status", "")
            images = gm.get("images", gm.get("all_images", gm.get("image_paths", None)))
            if images is None and hasattr(group, "images"):
                images = getattr(group, "images")
            if images is None:
                images = []
            images = list(images)
            selected_values = gm.get("selected_images", gm.get("selected_paths", None))
            selected_ids: set[str] = set()
            if selected_values is not None:
                for item in selected_values:
                    selected_ids.add(str(_image_value(item)))
            if not images:
                rows.append(
                    _normalise_row(
                        ManifestRow(
                            group_id=gid,
                            first_image=str(first),
                            merge_output=str(output),
                            status=str(status),
                        )
                    )
                )
                continue
            for image in images:
                im = _as_mapping(image) if isinstance(image, (Mapping, ManifestRow)) or is_dataclass(image) or hasattr(image, "__dict__") else {}
                value = _image_value(image)
                selected = im.get("selected", str(value) in selected_ids)
                rows.append(
                    _normalise_row(
                        {
                            "group_id": gid,
                            "first_image": first,
                            "image": value,
                            "selected": selected,
                            "selection_reason": im.get("selection_reason", im.get("reason", "")),
                            "quality_score": im.get("quality_score", ""),
                            "coverage_gain": im.get("coverage_gain", ""),
                            "group_coverage": gm.get("coverage", gm.get("group_coverage", "")),
                            "merge_output": output,
                            "status": status,
                        }
                    )
                )
        return rows

    def rows_from_result(self, result: Any) -> list[dict[str, Any]]:
        """Expand one pipeline ``MergeResult`` into real image rows.

        The pipeline intentionally keeps algorithm result types at an adapter
        boundary.  This method therefore uses duck typing and prefers the
        archive records (which contain every source path) over a synthetic
        group summary.  It remains useful for failed classification jobs,
        where ``MergeResult`` has no output but still carries an
        ``ArchiveResult``.
        """

        if isinstance(result, Mapping):
            group = result.get("group", result)
            analysis = result.get("analysis")
        else:
            group = getattr(result, "group", result)
            analysis = getattr(result, "analysis", None)
        # GroupAnalyzer's public mapping is also a useful direct manifest
        # input; in that shape the mapping itself contains the selection
        # metrics normally carried by ``MergeResult.analysis``.
        if analysis is None and group is result:
            analysis = result
        archive = getattr(result, "archive_result", None)
        records = list(getattr(archive, "records", ()) or ())
        all_paths = _result_paths(result, group, analysis)
        if not all_paths:
            all_paths = [_path_string(getattr(record, "source_path", "")) for record in records]
            all_paths = [item for item in all_paths if item]
        if not all_paths:
            # Last-resort object expansion retains one row per actual group
            # image rather than emitting the old blank GroupRecord row.
            all_paths = _group_image_paths(group)

        selected_paths = _result_selected_paths(result, analysis, group, all_paths)
        selected_keys = {_path_key(path) for path in selected_paths}
        reasons = _result_metric(analysis, "selection_reasons", "reasons", "selection_reason")
        gains = _result_metric(analysis, "coverage_gains", "gains", "coverage_gain")
        qualities = _result_metric(analysis, "quality_scores", "qualities", "quality_score", "sharpness_scores")
        group_coverage = _first_value(
            result,
            "group_coverage",
            "coverage",
            default=_first_value(analysis, "group_coverage", "coverage", "coverage_fraction", default=""),
        )
        output = _first_value(result, "output_path", "merge_output", default="")
        status = _first_value(result, "status", default=_first_value(group, "status", default=""))
        gid = _first_value(result, "group_id", default=_first_value(group, "group_id", "id", default=""))
        first = _first_value(result, "first_original", "first_image", default="")
        if not first:
            first = _first_value(group, "first_original_image", "first_original", "first_image", "first_item", default="")
        first = _path_string(first)
        if not first and all_paths:
            first = str(all_paths[0])
        capture_order = _first_value(analysis, "capture_order", default=all_paths)
        alignment_order = _first_value(analysis, "alignment_order", default=selected_paths)
        actual_hugin_input_order = _first_value(result, "actual_hugin_input_order", default=())
        order_confidence = _first_value(analysis, "alignment_order_confidence", default="")
        preview_reference = _first_value(analysis, "preview_reference", default="")
        requested_backend = _first_value(result, "requested_backend", default="")
        actual_backend = _first_value(result, "actual_backend", default="")
        backend = actual_backend or requested_backend
        alignment_level = _first_value(result, "alignment_level", default="")
        alignment_status = _first_value(result, "alignment_status", default="")
        crop_ratio = _first_value(result, "crop_ratio", default="")
        fallback_used = bool(_first_value(result, "fallback_used", default=False))
        diagnostics = _first_value(result, "diagnostics", default=_first_value(analysis, "alignment_order_diagnostics", default=""))

        # Use an archive record's source path as the authoritative identity;
        # records can include a failed/missing source not present in the
        # analysis result, so append those paths while retaining one row each.
        paths: list[str] = [str(path) for path in all_paths]
        for record in records:
            source = _path_string(getattr(record, "source_path", ""))
            if source and _path_key(source) not in {_path_key(item) for item in paths}:
                paths.append(source)

        image_objects = _group_images(group)
        rows: list[dict[str, Any]] = []
        for index, path in enumerate(paths):
            image_object = image_objects[index] if index < len(image_objects) else None
            image_name = _path_string(path)
            selected = _path_key(path) in selected_keys
            if image_object is not None:
                object_selected = _first_value(image_object, "selected", default=None)
                if object_selected is not None and not selected_keys:
                    selected = bool(object_selected)
            row = {
                "group_id": gid,
                "first_image": first,
                "image": image_name,
                "selected": selected,
                "selection_reason": _metric_value(reasons, index, path, default="coverage" if selected else "not selected"),
                "quality_score": _metric_value(qualities, index, path, default=""),
                "coverage_gain": _metric_value(gains, index, path, default=""),
                "group_coverage": group_coverage,
                "merge_output": output,
                "status": status,
                "capture_order": capture_order,
                "alignment_order": alignment_order,
                "actual_hugin_input_order": actual_hugin_input_order,
                "alignment_order_confidence": order_confidence,
                "preview_reference": preview_reference,
                "fusion_backend": backend,
                "requested_backend": requested_backend,
                "actual_backend": actual_backend,
                "alignment_level": alignment_level,
                "alignment_status": alignment_status,
                "crop_ratio": crop_ratio,
                "fallback_used": fallback_used,
                "diagnostics": diagnostics,
            }
            # Per-image metadata is useful when a lightweight analyzer places
            # it on ImageRecord rather than on a parallel result mapping.
            if image_object is not None:
                row["selection_reason"] = _first_value(
                    image_object,
                    "selection_reason",
                    "reason",
                    default=row["selection_reason"],
                )
                row["quality_score"] = _first_value(image_object, "quality_score", default=row["quality_score"])
                row["coverage_gain"] = _first_value(image_object, "coverage_gain", default=row["coverage_gain"])
            rows.append(_normalise_row(row))
        return rows

    def update_result(self, result: Any, path: os.PathLike[str] | str | None = None) -> Path:
        """Upsert one group's rows and atomically rewrite the manifest.

        Merge workers may call this concurrently when two Hugin workers are
        enabled, hence the small writer-local lock.  ``path`` is optional and
        defaults to the writer's configured destination.
        """

        rows = self.rows_from_result(result)
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("Manifest path is required")
        with self._lock:
            for row in rows:
                key = (str(row.get("group_id", "")), str(row.get("image", "")))
                self._rows[key] = row
            return self.write_csv(list(self._rows.values()), target)

    append_result = update_result
    update = update_result
    append = update_result
    write_result = update_result

    def write_results(self, results: Iterable[Any], path: os.PathLike[str] | str | None = None) -> Path:
        """Persist a complete result iterable using the same upsert rules."""

        with self._lock:
            for result in results:
                for row in self.rows_from_result(result):
                    key = (str(row.get("group_id", "")), str(row.get("image", "")))
                    self._rows[key] = row
            target = Path(path) if path is not None else self.path
            if target is None:
                raise ValueError("Manifest path is required")
            return self.write_csv(list(self._rows.values()), target)

    def write_csv(self, rows: Iterable[ManifestRow | Mapping[str, Any] | Any], path: os.PathLike[str] | str | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("Manifest path is required")
        temp, fd = _atomic_path(target)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(MANIFEST_FIELDS), extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(_normalise_row(row))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            temp.unlink(missing_ok=True)
            raise
        return target

    def write_json(self, rows: Iterable[ManifestRow | Mapping[str, Any] | Any], path: os.PathLike[str] | str | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("Manifest path is required")
        temp, fd = _atomic_path(target)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump([_normalise_row(row) for row in rows], stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            temp.unlink(missing_ok=True)
            raise
        return target

    def write(self, rows: Iterable[ManifestRow | Mapping[str, Any] | Any], path: os.PathLike[str] | str | None = None, *, format: str = "csv") -> Path:
        if format.lower() == "json":
            return self.write_json(rows, path)
        return self.write_csv(rows, path)

    export = write


def _image_value(image: Any) -> str:
    if isinstance(image, (str, os.PathLike)):
        return os.fspath(image)
    if isinstance(image, Mapping):
        return str(image.get("filename", image.get("path", image.get("current_path", image.get("original_path", "")))))
    for name in ("filename", "path", "current_path", "original_path"):
        value = getattr(image, name, None)
        if value is not None:
            return str(value)
    return str(image)


def _path_string(value: Any) -> str:
    """Return a stable path string from a Path or image-like object."""

    if value is None:
        return ""
    if isinstance(value, (str, os.PathLike)):
        return os.fspath(value)
    return _image_value(value)


def _path_key(value: Any) -> str:
    text = _path_string(value)
    if not text:
        return ""
    try:
        return os.path.normcase(os.path.abspath(text))
    except (TypeError, ValueError):
        return os.path.normcase(text)


def _first_value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
    else:
        for name in names:
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if value is not None:
                return value
    return default


def _group_images(group: Any) -> list[Any]:
    values = _first_value(group, "images", "items", "all_images", "image_records", "image_paths", default=[])
    if values is None or isinstance(values, (str, bytes, os.PathLike)):
        return [values] if values else []
    try:
        return list(values)
    except TypeError:
        return []


def _group_image_paths(group: Any) -> list[str]:
    return [path for path in (_path_string(item) for item in _group_images(group)) if path]


def _result_paths(result: Any, group: Any, analysis: Any) -> list[str]:
    values = _first_value(result, "all_paths", "all_images", "images", default=None)
    if values is None:
        values = _first_value(analysis, "all_paths", "all_images", "images", "items", "image_paths", default=None)
    if values is None:
        values = _group_image_paths(group)
    if isinstance(values, (str, bytes, os.PathLike)):
        values = [values]
    try:
        return [path for path in (_path_string(item) for item in values) if path]
    except TypeError:
        return []


def _result_selected_paths(result: Any, analysis: Any, group: Any, all_paths: Sequence[str]) -> list[str]:
    values = _first_value(result, "selected_source_paths", "selected_paths", default=None)
    if values is None or (isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and not values):
        values = _first_value(analysis, "selected_paths", "selected_images", "selected_items", "selected_indices", "selected", default=None)
    if values is None or (isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and not values):
        values = _first_value(group, "selected_paths", "selected_images", default=None)
    if values is None:
        values = []
    if isinstance(values, (str, bytes, os.PathLike)):
        values = [values]
    try:
        values = list(values)
    except TypeError:
        values = []
    if values and all(isinstance(value, int) for value in values):
        return [all_paths[value] for value in values if -len(all_paths) <= value < len(all_paths)]
    selected: list[str] = []
    current_keys = {_path_key(path) for path in all_paths}
    image_objects = _group_images(group)
    for value in values:
        path = _path_string(value)
        if not path:
            continue
        if _path_key(path) in current_keys:
            selected.append(path)
            continue
        # Cached analysis may identify an image by its original path after a
        # restart, while result/all_paths now expose the archived current path.
        matched = False
        raw_key = _path_key(path)
        for index, image in enumerate(image_objects):
            original = _first_value(image, "original_path", default=None)
            if original is not None and _path_key(original) == raw_key and index < len(all_paths):
                selected.append(all_paths[index])
                matched = True
                break
        if matched:
            continue
        basename_matches = [
            current for current in all_paths if Path(current).name.casefold() == Path(path).name.casefold()
        ]
        if len(basename_matches) == 1:
            selected.append(basename_matches[0])
    return selected


def _result_metric(obj: Any, *names: str) -> Any:
    return _first_value(obj, *names, default=None)


def _metric_value(metric: Any, index: int, path: str, *, default: Any = "") -> Any:
    if metric is None:
        return default
    if isinstance(metric, Mapping):
        keys: tuple[Any, ...] = (index, str(index), path, _path_key(path), Path(path).name)
        for key in keys:
            if key in metric:
                return metric[key]
        return default
    if isinstance(metric, Sequence) and not isinstance(metric, (str, bytes)):
        try:
            return metric[index]
        except (IndexError, TypeError):
            return default
    # A scalar quality/coverage value is meaningful for a one-image result;
    # avoid repeating arbitrary objects such as a focus-map array in rows.
    return metric if isinstance(metric, (int, float, str)) else default


def write_manifest(rows: Iterable[ManifestRow | Mapping[str, Any] | Any], path: os.PathLike[str] | str, *, format: str = "csv") -> Path:
    """Functional convenience API used by CLI and tests."""

    return ManifestWriter(path).write(rows, format=format)


def write_stack_manifest(groups: Iterable[Any], output_dir: os.PathLike[str] | str) -> Path:
    """Write the canonical ``stack_manifest.csv`` in *output_dir*."""

    writer = ManifestWriter(Path(output_dir) / "stack_manifest.csv")
    return writer.write_csv(writer.rows_from_groups(groups))


__all__ = [
    "MANIFEST_FIELDS",
    "ManifestRow",
    "ManifestWriter",
    "write_manifest",
    "write_stack_manifest",
]

