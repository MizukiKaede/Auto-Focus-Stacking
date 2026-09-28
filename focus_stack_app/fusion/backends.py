"""Standard Hugin/Enfuse and experimental OpenCV fusion backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import logging
import math
from pathlib import Path
import threading
from typing import Any, Mapping, Sequence

from ..hugin.align import AlignConfig, AlignImageStack, AlignmentError, AlignmentResult
from ..hugin.enfuse import Enfuser
from ..hugin.output_encoder import encode_output, encode_image_output, OutputCollisionError, _same_path
from ..hugin.hugin_locator import HuginToolNotFound
from ..utils.image_io import load_rgb


@dataclass(slots=True)
class FusionResult:
    output_path: Path
    actual_backend: str
    aligned_paths: tuple[Path, ...] = ()
    alignment_level: int | None = None
    alignment_status: str = "NOT_RUN"
    crop_ratio: float | None = None
    fallback_used: bool = False
    diagnostics: tuple[str, ...] = ()
    hugin_command: tuple[str, ...] = ()
    hugin_exit_code: int | None = None
    enfuse_command: tuple[str, ...] = ()
    enfuse_exit_code: int | None = None
    actual_hugin_input_order: tuple[Path, ...] = ()


class FusionBackend(ABC):
    name: str

    @abstractmethod
    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event) -> FusionResult:
        raise NotImplementedError


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _image_info(path: Path) -> tuple[int, int, str, int]:
    from PIL import Image
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        bits_value = getattr(image, "tag_v2", {}).get(258, None)
        if isinstance(bits_value, (tuple, list)):
            bits = max(int(value) for value in bits_value)
        elif bits_value is not None:
            bits = int(bits_value)
        else:
            bits = 16 if "16" in mode or mode in {"I", "F"} else 8
    allowed_modes = {"L", "LA", "RGB", "RGBA", "CMYK", "I;16", "I;16B", "I;16L", "I", "F"}
    if width <= 0 or height <= 0 or mode not in allowed_modes or bits not in {8, 16, 32}:
        raise ValueError(f"unsupported image layout: {path} ({width}x{height}, {mode})")
    return width, height, mode, bits


class HuginEnfuseBackend(FusionBackend):
    name = "hugin_enfuse"

    def __init__(self, aligner: AlignImageStack | Any = None, enfuser: Enfuser | Any = None,
                 *, runtime_config: Any = None, logger: logging.Logger | None = None):
        self.aligner = aligner or AlignImageStack()
        self.enfuser = enfuser or Enfuser()
        self.runtime_config = runtime_config
        self.logger = logger or logging.getLogger(__name__)

    def _threshold(self, name: str, default: float) -> float:
        return float(_value(self.runtime_config, name, default))

    def _validate_alignment(self, alignment: AlignmentResult, anchor: Path) -> tuple[float, list[str]]:
        from ..core.group_detector import subject_feature, subject_changed
        from .alignment_quality import colour_subject_mask, colour_subject_mismatch

        input_paths = tuple(getattr(alignment, "input_paths", ()))
        if input_paths and len(alignment.aligned_paths) != len(input_paths):
            raise AlignmentError("aligned TIFF count does not match input count", alignment)
        dimensions = []
        subject_anchor = None
        colour_anchor = None
        have_anchor = False
        for path in alignment.aligned_paths:
            minimum_bytes = int(self._threshold("min_alignment_tiff_bytes", 128)) if isinstance(self.aligner, AlignImageStack) else 1
            if not path.is_file() or path.stat().st_size < minimum_bytes:
                raise AlignmentError(f"aligned TIFF is missing or empty: {path}", alignment)
            if isinstance(self.aligner, AlignImageStack):
                try:
                    dimensions.append(_image_info(path)[:2])
                except Exception as exc:
                    raise AlignmentError(f"aligned TIFF is unreadable: {path}: {exc}", alignment) from exc
                preview = load_rgb(path, 1280)
                colour = colour_subject_mask(preview)
                if have_anchor:
                    mismatch = colour_subject_mismatch(colour_anchor, colour)
                    if mismatch:
                        raise AlignmentError(f"主体对齐检查未通过，可能存在转面或重影：{path}: {mismatch}", alignment)
                else:
                    colour_anchor = colour
                    have_anchor = True
                subject = subject_feature(load_rgb(path, 640))
                if subject_anchor is None:
                    subject_anchor = subject
                elif subject_changed(subject_anchor, subject):
                    raise AlignmentError(f"主体对齐检查未通过，可能存在转面或重影：{path}", alignment)
        if not isinstance(self.aligner, AlignImageStack):
            return 1.0, ["CUSTOM_ALIGNER_VALIDATION_LIMITED"]
        if len(set(dimensions)) != 1:
            raise AlignmentError(f"aligned TIFF dimensions disagree: {dimensions}", alignment)
        try:
            anchor_width, anchor_height, _, _ = _image_info(anchor)
        except Exception as exc:
            raise ValueError(f"group anchor image is unreadable: {anchor}: {exc}") from exc
        width, height = dimensions[0]
        crop_ratio = (width * height) / max(1, anchor_width * anchor_height)
        fail = self._threshold("crop_ratio_fail", 0.35)
        warn = self._threshold("crop_ratio_warning", 0.65)
        if crop_ratio < fail or crop_ratio > self._threshold("crop_ratio_max", 1.75):
            raise AlignmentError(f"alignment crop ratio {crop_ratio:.3f} is outside safe bounds", alignment)
        diagnostics = [f"CROP_RATIO_WARNING:{crop_ratio:.3f}"] if crop_ratio < warn else []
        return crop_ratio, diagnostics

    @staticmethod
    def _retryable(exc: Exception, cancel_event: threading.Event) -> bool:
        if cancel_event.is_set() or isinstance(exc, (HuginToolNotFound, FileNotFoundError, ValueError)):
            return False
        message = str(exc).casefold()
        if any(token in message for token in ("corrupt", "damaged", "unsupported input", "cannot decode", "unreadable input")):
            return False
        result = getattr(exc, "result", None)
        command = getattr(result, "command_result", None)
        return not bool(getattr(command, "cancelled", False) or getattr(command, "timed_out", False))

    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event) -> FusionResult:
        ordered = [Path(path) for path in _value(analysis, "alignment_order", ())]
        selected = [Path(path) for path in _value(analysis, "selected_paths", ())]
        paths = ordered or selected
        if len(paths) < 2:
            raise ValueError("Hugin fusion requires at least two selected images")
        anchor = Path(_value(analysis, "first_original_path", paths[0]))
        levels = (
            ((), False),
            (("--corr=0.8",), False),
            (("-d", "--corr=0.8"), False),
            (("-d", "--corr=0.8"), True),
        )
        diagnostics: list[str] = []
        alignment = None
        crop_ratio = None
        level = 0
        for level, (extra, centre) in enumerate(levels, 1):
            attempt_dir = Path(work_dir) / f"alignment_level_{level}"
            self.logger.info("Hugin alignment attempt level=%s input_order=%s work_dir=%s", level, paths, attempt_dir)
            try:
                if isinstance(self.aligner, AlignImageStack):
                    original = self.aligner.config
                    self.aligner.config = replace(
                        original, optimize_field_of_view=True, optimize_scale=True,
                        crop_to_fit=True, optimize_centre=centre,
                        extra_args=tuple(extra),
                    )
                    try:
                        alignment = self.aligner.align(paths, work_dir=attempt_dir, cancel_event=cancel_event)
                    finally:
                        self.aligner.config = original
                else:
                    alignment = self.aligner.align(paths, work_dir=attempt_dir, cancel_event=cancel_event)
                    if len(tuple(alignment.aligned_paths)) != len(paths):
                        raise AlignmentError("aligned TIFF count does not match input count", alignment)
                crop_ratio, warnings = self._validate_alignment(alignment, anchor)
                diagnostics.extend(warnings)
                self.logger.info("Hugin alignment validated level=%s aligned_tiff_count=%s crop_ratio=%.4f", level, len(alignment.aligned_paths), crop_ratio)
                break
            except Exception as exc:
                diagnostics.append(f"ALIGNMENT_LEVEL_{level}_FAILED:{exc}")
                if level == len(levels) or not self._retryable(exc, cancel_event):
                    raise
        assert alignment is not None
        fused = self.enfuser.fuse(
            alignment.aligned_paths, output_path, work_dir=Path(work_dir) / "enfuse",
            cancel_event=cancel_event, output_config=output_config,
        )
        final = Path(fused.output_path)
        minimum_output_bytes = int(self._threshold("min_fusion_output_bytes", 128)) if isinstance(self.enfuser, Enfuser) else 1
        if not final.is_file() or final.stat().st_size < minimum_output_bytes:
            raise RuntimeError(f"fusion output is missing or empty: {final}")
        if isinstance(self.enfuser, Enfuser):
            try:
                _image_info(final)
            except Exception as exc:
                raise RuntimeError(f"fusion output is unreadable: {final}: {exc}") from exc
        alignment_command = getattr(alignment, "command_result", None)
        enfuse_command = getattr(fused, "command_result", None)
        group_id = _value(group, "group_id", _value(group, "id", None))
        self.logger.info(
            "fusion commands group_id=%s hugin_level=%s hugin_command=%s hugin_exit_code=%s "
            "aligned_tiff_count=%s enfuse_command=%s enfuse_exit_code=%s",
            group_id, level, getattr(alignment_command, "command", ()),
            getattr(alignment_command, "returncode", None), len(alignment.aligned_paths),
            getattr(enfuse_command, "command", ()), getattr(enfuse_command, "returncode", None),
        )
        return FusionResult(
            final, self.name, tuple(alignment.aligned_paths), level, "VALIDATED", crop_ratio,
            # Alignment levels are retries inside the requested backend, not
            # a silent backend fallback to OpenCV.
            fallback_used=False, diagnostics=tuple(diagnostics),
            hugin_command=tuple(getattr(alignment_command, "command", ())),
            hugin_exit_code=getattr(alignment_command, "returncode", None),
            enfuse_command=tuple(getattr(enfuse_command, "command", ())),
            enfuse_exit_code=getattr(enfuse_command, "returncode", None),
            actual_hugin_input_order=tuple(paths),
        )


class OpenCVFusionBackend(FusionBackend):
    """Fast experimental whole-frame focus fusion."""
    name = "opencv"

    def __init__(self, *, aligned_cache_bytes=None):
        from .aligned_cache import DEFAULT_ALIGNED_CACHE_BYTES
        self.aligned_cache_bytes = DEFAULT_ALIGNED_CACHE_BYTES if aligned_cache_bytes is None else int(aligned_cache_bytes)
        if self.aligned_cache_bytes < 0:
            raise ValueError("aligned cache limit cannot be negative")

    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event) -> FusionResult:
        import cv2
        import numpy as np
        from PIL import Image

        paths = [Path(path) for path in _value(analysis, "selected_paths", ())]
        indices = list(_value(analysis, "selected_indices", range(len(paths))))
        transforms = list(_value(analysis, "preview_transforms", ()))
        analysis_shapes = list(_value(analysis, "analysis_shapes", ()))
        reference_index = int(_value(analysis, "preview_reference_index", indices[0] if indices else 0))
        if len(paths) < 2:
            raise ValueError("OpenCV fusion requires at least two selected images")
        if len(indices) != len(paths) or any(index < 0 for index in indices):
            raise ValueError("OpenCV selected indices must match selected paths")
        reference_path = Path(_value(analysis, "preview_reference", paths[0]))
        destination = Path(output_path)
        if any(_same_path(path, destination) for path in [*paths, reference_path]):
            raise OutputCollisionError(f"Output cannot overwrite an input image: {destination}")
        if destination.exists() and not _value(output_config, "overwrite", False):
            raise OutputCollisionError(f"Refusing to overwrite existing output: {destination}")
        reference_analysis_shape = _value(analysis, "reference_analysis_shape", None)
        if not reference_analysis_shape:
            reference_analysis_shape = analysis_shapes[reference_index] if 0 <= reference_index < len(analysis_shapes) else None

        # Recovery rows do not persist the coordinate sizes of preview
        # transforms. Rebuild registration instead of interpreting full-size
        # source coordinates as 640-pixel preview coordinates.
        rebuilt_registration = not reference_analysis_shape or any(
            index >= len(transforms) or index >= len(analysis_shapes) or not analysis_shapes[index]
            for index in indices
        )
        if rebuilt_registration:
            from ..core.registration import register_images

            ref = load_rgb(reference_path, 640)
            transforms = [None] * (max(indices) + 1)
            analysis_shapes = [None] * len(transforms)
            reference_analysis_shape = ref.shape[:2]
            for index, path in zip(indices, paths):
                if cancel_event.is_set():
                    raise RuntimeError("OpenCV fusion cancelled")
                if path.resolve() == reference_path.resolve():
                    preview = ref
                    matrix = np.eye(3, dtype=np.float32)
                else:
                    preview = load_rgb(path, 640)
                    registration = register_images(ref, preview)
                    if not registration.valid:
                        raise RuntimeError(f"OpenCV could not rebuild alignment for {path}: {registration.message}")
                    matrix = registration.matrix
                transforms[index] = matrix
                analysis_shapes[index] = preview.shape[:2]
            del ref, preview

        def scaled_matrix(matrix, source_from, source_to, target_from, target_to):
            matrix = np.asarray(matrix, np.float32) if matrix is not None else np.eye(3, dtype=np.float32)
            sfh, sfw = source_from
            sth, stw = source_to
            tfh, tfw = target_from
            tth, ttw = target_to
            source_to_analysis = np.diag([sfw / max(1, stw), sfh / max(1, sth), 1.0]).astype(np.float32)
            analysis_to_target = np.diag([ttw / max(1, tfw), tth / max(1, tfh), 1.0]).astype(np.float32)
            return analysis_to_target @ matrix @ source_to_analysis

        from .focus_masks import build_focus_labels, blend_focus_pyramid
        from .aligned_cache import AlignedFrameCache

        with Image.open(reference_path) as reference_image:
            width, height = reference_image.size

        def load_aligned(local):
            index, path = indices[local], paths[local]
            matrix = transforms[index] if index < len(transforms) else None
            source = load_rgb(path)
            source_analysis_shape = analysis_shapes[index]
            full_matrix = scaled_matrix(
                matrix, source_analysis_shape, source.shape[:2],
                reference_analysis_shape, (height, width),
            )
            return cv2.warpPerspective(
                source, full_matrix, (width, height), flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        cache = AlignedFrameCache(
            load_aligned, max_bytes=self.aligned_cache_bytes,
            working_bytes=width * height * 80,
        )
        try:
            labels = build_focus_labels(len(paths), cache.for_focus, cancel_event=cancel_event)
            result = blend_focus_pyramid(len(paths), cache.for_blend, labels, cancel_event=cancel_event)
        finally:
            cache.clear()
        del labels
        if cancel_event.is_set():
            raise RuntimeError("OpenCV fusion cancelled")
        with Image.fromarray(result) as image:
            final = encode_image_output(image, output_path, config=output_config, original_path=reference_path)
        diagnostics = ("EXPERIMENTAL_BACKEND", "FULL_RESOLUTION_FOCUS_MASKS", "CONVEX_FOCUS_BLEND",
                       f"ALIGNED_CACHE_HITS:{cache.hits}/{len(paths)}",
                       f"ALIGNED_CACHE_PEAK_BYTES:{cache.peak_bytes}")
        if rebuilt_registration:
            diagnostics += ("PREVIEW_REGISTRATION_REBUILT",)
        return FusionResult(Path(final), self.name, alignment_status="PREVIEW_TRANSFORMS", diagnostics=diagnostics)


class LegacyWholeFrameBackend(FusionBackend):
    """Opt-in adapter for the original renderer; never an automatic fallback.

    Interface retained for comparisons:
    StackMergeService(..., fusion_backend=LegacyWholeFrameBackend()).
    Default construction still selects HuginEnfuseBackend and its existing
    alignment order, retry and export workflow.
    """
    name = "legacy_whole_frame"

    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event) -> FusionResult:
        import cv2
        import numpy as np
        from .legacy_whole_frame import render_plan

        # Generic analysis publishes source-to-reference homogeneous matrices;
        # the original renderer expects reference-to-source affine matrices.
        paths = list(_value(analysis, "capture_order", ()))
        reference = str(_value(analysis, "preview_reference", ""))
        selected = [str(path) for path in _value(analysis, "selected_paths", ())]
        transforms = _value(analysis, "preview_transforms", ())
        original_paths = [str(_value(row, "path")) for row in _value(analysis, "per_image", ())]
        if original_paths:
            paths = original_paths
        if not paths or reference not in paths or len(transforms) != len(paths):
            raise ValueError("legacy fusion requires original-order paths and preview transforms")
        plan = {
            "paths": paths, "reference_index": paths.index(reference),
            "selected_indices": [paths.index(path) for path in selected],
            "preview_shape": _value(analysis, "reference_analysis_shape"),
            "matrices": [cv2.invertAffineTransform(np.asarray(matrix, np.float32)[:2]).tolist()
                         for matrix in transforms],
        }
        intermediate = render_plan(plan, Path(work_dir) / "legacy_whole_frame.tif", cancel_event=cancel_event)
        final = encode_output(intermediate, output_path, config=output_config)
        return FusionResult(Path(final), self.name, alignment_status="PREVIEW_TRANSFORMS",
                            diagnostics=("LEGACY_WHOLE_FRAME_OPT_IN",))


__all__ = ["FusionBackend", "FusionResult", "HuginEnfuseBackend", "OpenCVFusionBackend", "LegacyWholeFrameBackend"]

