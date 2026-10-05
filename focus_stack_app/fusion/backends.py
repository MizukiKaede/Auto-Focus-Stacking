"""The production quality backend and the experimental Hugin backend."""
from __future__ import annotations
from .native_runtime import native_fusion

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, replace
import logging
import math
from pathlib import Path
import threading
from typing import Any, Mapping

from ..hugin.align import AlignConfig, AlignImageStack, AlignmentError, AlignmentResult
from ..hugin.enfuse import Enfuser
from ..hugin.output_encoder import encode_image_output, OutputCollisionError, _same_path
from ..hugin.hugin_locator import HuginToolNotFound
from ..utils.image_io import load_rgb
from ..utils.performance import profiled_fusion, stage, diagnostic


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


def _report_cache_then_clear(cache: Any, diagnostic_name: str) -> None:
    """Always release a fusion cache, even if its diagnostic sink fails."""
    try:
        if hasattr(cache, "stats"):
            diagnostic(diagnostic_name, **cache.stats)
    finally:
        cache.clear()


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
        self.runtime_config = (runtime_config.get("runtime", runtime_config)
                               if isinstance(runtime_config, Mapping)
                               else getattr(runtime_config, "runtime", runtime_config))
        self.logger = logger or logging.getLogger(__name__)

    def _threshold(self, name: str, default: float) -> float:
        return float(_value(self.runtime_config, name, default))

    def _validate_alignment(
        self, alignment: AlignmentResult, anchor: Path, *, aligned_cache=None,
    ) -> tuple[float, list[str]]:
        from ..core.group_detector import subject_feature, subject_changed
        from .alignment_quality import colour_subject_mask, colour_subject_mismatch, defocus_geometry_consistent

        input_paths = tuple(getattr(alignment, "input_paths", ()))
        if input_paths and len(alignment.aligned_paths) != len(input_paths):
            raise AlignmentError("aligned TIFF count does not match input count", alignment)
        dimensions = []
        subject_anchor = None
        colour_anchor = None
        anchor_preview = None
        defocus_confirmations = 0
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
                if aligned_cache is None:
                    preview = load_rgb(path, 1280)
                    small_preview = load_rgb(path, 640)
                else:
                    previews = aligned_cache.validation_previews(path)
                    preview, small_preview = previews[1280], previews[640]
                colour = colour_subject_mask(preview)
                if have_anchor:
                    mismatch = colour_subject_mismatch(colour_anchor, colour)
                    if mismatch and defocus_geometry_consistent(anchor_preview, preview, colour_anchor, colour):
                        defocus_confirmations += 1
                        mismatch = None
                    if mismatch:
                        raise AlignmentError(f"主体对齐检查未通过，可能存在转面或重影：{path}: {mismatch}", alignment)
                else:
                    colour_anchor = colour
                    anchor_preview = preview
                    have_anchor = True
                subject = subject_feature(small_preview)
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
        if defocus_confirmations:
            diagnostics.append(f"DEFOCUS_GEOMETRY_CONFIRMED:{defocus_confirmations}")
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

    def _release_superseded_tiffs(self, alignment, work_dir):
        """Keep the latest retry's TIFFs, with prior failures recorded in logs."""
        if alignment is None or not isinstance(self.aligner, AlignImageStack):
            return
        root = Path(work_dir).resolve()
        attempt = Path(alignment.work_dir).resolve()
        if attempt.parent != root or (root / ".keep").exists() or (attempt / ".keep").exists():
            return
        for path in alignment.aligned_paths:
            path = Path(path)
            # Only remove TIFFs returned by our previous attempt, directly
            # inside this group's private attempt directory. Never follow
            # symlinks to an original or touch unrelated diagnostic files.
            if path.resolve().parent != attempt or path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                self.logger.warning("Unable to release superseded alignment TIFF %s", path)

    @profiled_fusion
    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event) -> FusionResult:
        ordered = [Path(path) for path in _value(analysis, "alignment_order", ())]
        selected = [Path(path) for path in _value(analysis, "selected_paths", ())]
        paths = ordered or selected
        if len(paths) < 2:
            raise ValueError("Hugin fusion requires at least two selected images")
        if selected and ordered and (len(ordered) != len(selected)
                or {path.resolve() for path in ordered} != {path.resolve() for path in selected}):
            raise ValueError("Hugin alignment order does not contain exactly the selected inputs")
        preset = _value(self.runtime_config, "hugin_alignment_preset", "legacy")
        if preset not in {"legacy", "first", "reference_first"}:
            raise ValueError("hugin_alignment_preset must be legacy, first or reference_first")
        if preset == "reference_first":
            reference = _value(analysis, "preview_reference", None)
            reference_path = next((path for path in paths
                                   if reference is not None and _same_path(path, Path(reference))), None)
            if reference_path is None:
                raise ValueError("Hugin reference_first requires a selected preview reference")
            paths = [reference_path, *(path for path in paths if path != reference_path)]
        anchor = Path(_value(analysis, "first_original_path", paths[0]))
        levels = (
            ((), False),
            (("--corr=0.8",), False),
            (("-d", "--corr=0.8"), False),
            (("-d", "--corr=0.8"), True),
        )
        # A sparse registration tour can jump between focus planes. Above
        # 20 selected frames, prefer capture order on the first attempt.
        # Smaller stacks retry capture order before flexible lens parameters.
        # Never infer chronology from
        # filenames (camera numbering can wrap), or add/drop selected frames.
        selected_keys = {path.resolve() for path in paths}
        capture_paths = [Path(path) for path in _value(analysis, "capture_order", ())
                         if Path(path).resolve() in selected_keys]
        capture_order_complete = (len(capture_paths) == len(paths)
                                  and {path.resolve() for path in capture_paths} == selected_keys)
        prefer_capture_order = (preset != "reference_first" and
                                len(paths) > 20 and capture_order_complete)
        if prefer_capture_order:
            paths = capture_paths
        use_capture_retry = (preset != "reference_first" and
                             capture_order_complete and capture_paths != paths)
        if use_capture_retry:
            levels = (((), False), ((), False), (("--corr=0.8",), False),
                      (("-d", "--corr=0.8"), False))
        from .aligned_cache import AlignedTIFFImageCache, DEFAULT_ALIGNED_TIFF_CACHE_BYTES
        from ..utils.shared_cache_budget import process_global_cache_budget

        aligned_cache = AlignedTIFFImageCache(
            max_bytes=int(_value(
                self.runtime_config, "aligned_tiff_cache_bytes",
                DEFAULT_ALIGNED_TIFF_CACHE_BYTES,
            )),
            shared_budget=process_global_cache_budget(
                int(_value(self.runtime_config, "fusion_cache_budget_bytes", 3 * 1024**3))),
        )
        diagnostics: list[str] = []
        if preset == "reference_first":
            diagnostics.append("ALIGNMENT_PREVIEW_REFERENCE_FIRST")
        if prefer_capture_order:
            diagnostics.append("ALIGNMENT_CAPTURE_ORDER_PRIMARY")
        alignment = None
        crop_ratio = None
        level = 0
        previous_attempt = None
        for level, (extra, centre) in enumerate(levels, 1):
            # Failed output is already described in the profile. Release it
            # before writing a replacement, including incomplete tool runs.
            # A diagnostic .keep marker explicitly retains the evidence.
            self._release_superseded_tiffs(previous_attempt, work_dir)
            previous_attempt = None
            if level == 2 and use_capture_retry:
                paths = capture_paths
                diagnostics.append("ALIGNMENT_CAPTURE_ORDER_RETRY")
            attempt_dir = Path(work_dir) / f"alignment_level_{level}"
            self.logger.info("Hugin alignment attempt level=%s input_order=%s work_dir=%s", level, paths, attempt_dir)
            try:
                if isinstance(self.aligner, AlignImageStack):
                    # Each worker owns its retry configuration. The runner
                    # and logger may be shared, but never temporarily replace
                    # the shared adapter's config across a subprocess wait.
                    attempt_aligner = copy(self.aligner)
                    original = self.aligner.config
                    preset_args = ("--align-to-first",) if preset in {"first", "reference_first"} else ()
                    attempt_aligner.config = replace(
                        original, optimize_field_of_view=True, optimize_scale=True,
                        crop_to_fit=True, optimize_centre=centre,
                        extra_args=tuple(original.extra_args) + preset_args + tuple(extra),
                    )
                    alignment = attempt_aligner.align(paths, work_dir=attempt_dir, cancel_event=cancel_event)
                else:
                    alignment = self.aligner.align(paths, work_dir=attempt_dir, cancel_event=cancel_event)
                    if len(tuple(alignment.aligned_paths)) != len(paths):
                        raise AlignmentError("aligned TIFF count does not match input count", alignment)
                # Retain this attempt on a terminal failure; a later retry
                # releases it before writing its own TIFFs unless .keep exists.
                previous_attempt = alignment
                command_result = getattr(alignment, "command_result", None)
                tiff_layouts = []
                for tiff in alignment.aligned_paths:
                    try:
                        tiff_layouts.append(dict(path=str(tiff), bytes=tiff.stat().st_size,
                                                layout=_image_info(tiff)))
                    except Exception:
                        pass
                diagnostic("hugin_alignment_chain", attempt=level, input_order=[str(p) for p in paths],
                           command=list(getattr(command_result, "command", ())), retry_args=list(extra),
                           tiffs=tiff_layouts, total_tiff_bytes=sum(t["bytes"] for t in tiff_layouts),
                           timing_scope="alignment_includes_external_tiff_write")
                crop_ratio, warnings = self._validate_alignment(
                    alignment, anchor, aligned_cache=aligned_cache,
                )
                diagnostics.extend(warnings)
                self.logger.info("Hugin alignment validated level=%s aligned_tiff_count=%s crop_ratio=%.4f", level, len(alignment.aligned_paths), crop_ratio)
                break
            except Exception as exc:
                try:
                    if hasattr(aligned_cache, "stats"):
                        diagnostic("aligned_tiff_cache_attempt", attempt=level,
                                   **aligned_cache.stats)
                finally:
                    aligned_cache.clear()
                failed_alignment = getattr(exc, "result", None)
                if failed_alignment is not None:
                    previous_attempt = failed_alignment
                failed_command = getattr(failed_alignment, "command_result", None)
                diagnostic("hugin_alignment_failed", attempt=level, error=str(exc),
                           input_order=[str(p) for p in paths], retry_args=list(extra),
                           command=list(getattr(failed_command, "command", ())))
                diagnostics.append(f"ALIGNMENT_LEVEL_{level}_FAILED:{exc}")
                if level == len(levels) or not self._retryable(exc, cancel_event):
                    raise
        assert alignment is not None
        try:
            fuse_kwargs = {
                "work_dir": Path(work_dir) / "enfuse",
                "cancel_event": cancel_event,
                "output_config": output_config,
            }
            if isinstance(self.enfuser, Enfuser):
                fuse_kwargs["hugin_parallel_cpu_budget"] = _value(
                    self.runtime_config, "hugin_parallel_cpu_budget", 12,
                )
                fuse_kwargs["image_loader"] = (
                    lambda index: aligned_cache.load(alignment.aligned_paths[index])
                )
                fuse_kwargs["preparation_image_loader"] = (
                    lambda index: aligned_cache.peek(alignment.aligned_paths[index])
                )
                fuse_kwargs["refined_frame_cache_bytes"] = int(_value(
                    self.runtime_config, "aligned_tiff_cache_bytes",
                    DEFAULT_ALIGNED_TIFF_CACHE_BYTES,
                ))
                fuse_kwargs["refined_frame_cache_shared_budget"] = aligned_cache.shared_budget
                fuse_kwargs["tone_tiff_compression"] = str(_value(
                    self.runtime_config, "hugin_tone_tiff_compression", "raw",
                ))
                fuse_kwargs["cleanup_on_success"] = not (Path(work_dir) / ".keep").exists()
                mask_mode = _value(self.runtime_config, "hugin_focus_mask_mode", "legacy")
                if mask_mode != "legacy":
                    reference = _value(analysis, "preview_reference", None)
                    reference_index = next((i for i, path in enumerate(paths)
                                            if reference is not None and _same_path(path, Path(reference))), None)
                    if reference_index is None and mask_mode == "gate":
                        raise ValueError("Hugin texture gate requires a selected preview reference")
                    # Hugin's repair statistics use the selected preview
                    # reference in the actual aligned input order.
                    if reference_index is None:
                        reference_index = 0
                    fuse_kwargs.update(focus_mask_mode=mask_mode, focus_reference_index=reference_index,
                                       focus_gate_edge_mode=_value(self.runtime_config, "quality_gate_edge_mode", "localized"),
                                       focus_edge_ownership=bool(_value(self.runtime_config, "hugin_edge_ownership", True)),
                                       focus_surface_tone=bool(_value(self.runtime_config, "hugin_surface_tone", True)))
            fused = self.enfuser.fuse(
                alignment.aligned_paths, output_path, **fuse_kwargs,
            )
        finally:
            _report_cache_then_clear(aligned_cache, "aligned_tiff_cache")
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
            # Alignment levels are retries inside the requested backend.
            fallback_used=False, diagnostics=tuple(diagnostics),
            hugin_command=tuple(getattr(alignment_command, "command", ())),
            hugin_exit_code=getattr(alignment_command, "returncode", None),
            enfuse_command=tuple(getattr(enfuse_command, "command", ())),
            enfuse_exit_code=getattr(enfuse_command, "returncode", None),
            actual_hugin_input_order=tuple(paths),
        )


class QualityFusionBackend(FusionBackend):
    """Full-resolution, in-memory focus fusion with coherent neutral edges."""

    name = "quality"

    def __init__(self, *, aligned_cache_bytes=None, runtime_config=None):
        from .aligned_cache import DEFAULT_ALIGNED_CACHE_BYTES
        self.runtime_config = (runtime_config.get("runtime", runtime_config)
                               if isinstance(runtime_config, Mapping)
                               else getattr(runtime_config, "runtime", runtime_config))
        configured_cache_bytes = _value(self.runtime_config, "opencv_aligned_cache_bytes", None)
        if aligned_cache_bytes is None:
            aligned_cache_bytes = (DEFAULT_ALIGNED_CACHE_BYTES if configured_cache_bytes is None
                                   else configured_cache_bytes)
        self.aligned_cache_bytes = int(aligned_cache_bytes)
        if self.aligned_cache_bytes < 0:
            raise ValueError("aligned cache limit cannot be negative")

    @profiled_fusion
    @native_fusion
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
            raise ValueError("Quality fusion requires at least two selected images")
        if len(indices) != len(paths) or any(index < 0 for index in indices):
            raise ValueError("Quality selected indices must match selected paths")
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
                    raise RuntimeError("Quality fusion cancelled")
                if path.resolve() == reference_path.resolve():
                    preview = ref
                    matrix = np.eye(3, dtype=np.float32)
                else:
                    preview = load_rgb(path, 640)
                    registration = register_images(ref, preview)
                    if not registration.valid:
                        raise RuntimeError(f"Quality fusion could not rebuild alignment for {path}: {registration.message}")
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
        from .quality_fusion import stabilize_neutral_labels
        from .aligned_cache import AlignedFrameCache

        with Image.open(reference_path) as reference_image:
            width, height = reference_image.size
        execution = _value(self.runtime_config, "quality_execution", "cached")
        decoder = _value(self.runtime_config, "quality_jpeg_decoder", "pillow")
        if execution not in {"cached", "memory", "streaming"}:
            raise ValueError("quality_execution must be cached, memory or streaming")
        if decoder not in {"pillow", "opencv"}:
            raise ValueError("quality_jpeg_decoder must be pillow or opencv")
        if execution != "cached" and decoder != "opencv":
            raise ValueError("two-pass and memory-reference Quality require opencv JPEG decoding")
        pass_number = 1
        if decoder == "opencv":
            from ..utils.image_io import load_quality_rgb
            def load_source(path):
                return load_quality_rgb(path, pass_index=pass_number)
        else:
            load_source = load_rgb
        diagnostic("quality_configuration", input_order=[str(p) for p in paths],
                   selected_indices=indices, reference=str(reference_path),
                   dimensions=[width, height], blend_levels=1, aligned_cache_bytes=self.aligned_cache_bytes,
                   reference_coordinate_shape=reference_analysis_shape,
                   source_coordinate_shapes=[analysis_shapes[i] for i in indices],
                   registration_diagnostics=bool(_value(self.runtime_config, "registration_diagnostics", False)),
                   execution=execution, jpeg_decoder=decoder,
                   printed_edge_guard=bool(_value(self.runtime_config, "quality_printed_edge_guard", False)
                                           and _value(self.runtime_config, "quality_variant", "legacy") in {"gate", "clean"}),
                   surface_tone=bool(_value(self.runtime_config, "quality_surface_tone", False)
                                     and _value(self.runtime_config, "quality_variant", "legacy") in {"gate", "clean"}))

        variant = _value(self.runtime_config, "quality_variant", "legacy")
        if variant not in {"legacy", "gain", "gate", "clean"}:
            raise ValueError("quality_variant must be legacy, gain, gate or clean")
        gain_model = None
        seeded_source = {}
        selected_reference = next((i for i, path in enumerate(paths)
                                   if _same_path(path, reference_path)), None)
        if execution != "cached" and selected_reference is None:
            raise ValueError("two-pass Quality requires a selected preview reference")
        fusion_reference = selected_reference if selected_reference is not None else 0
        surface_tone = None
        surface_boundary = None
        printed_guard = None
        valid_sources = None
        if variant in {"gate", "clean"}:
            from .opencv_v20 import ValidSourceOwnership
            valid_sources = ValidSourceOwnership()
        if variant in {"gate", "clean"} and _value(self.runtime_config, "quality_printed_edge_guard", False):
            from .quality_fusion import PrintedEdgeOwnership
            printed_guard = PrintedEdgeOwnership()
        if variant in {"gate", "clean"} and _value(self.runtime_config, "quality_surface_tone", False):
            from .surface_tone import SurfaceToneHarmonizer
            surface_tone = SurfaceToneHarmonizer(fusion_reference)
            from .opencv_v20 import OpenCVBoundaryOwnership
            surface_boundary = OpenCVBoundaryOwnership()
        use_gain = (variant == "gain" or (variant in {"gate", "clean"} and
                                         _value(self.runtime_config, "quality_exposure_gain", False)))
        # Seed the reference once for the sequential pass; diagnostics and
        # gain previews retain compact data rather than rereading its JPEG.
        raw_reference = None
        if execution != "cached" or use_gain:
            raw_reference = load_source(paths[fusion_reference])
            seeded_source[fusion_reference] = raw_reference
        residuals = None
        if _value(self.runtime_config, "registration_diagnostics", False):
            from .registration_diagnostics import RegistrationResiduals
            residuals = RegistrationResiduals(
                raw_reference if raw_reference is not None and _same_path(paths[fusion_reference], reference_path)
                else load_source(reference_path)
            )
        if use_gain:
            from .exposure_gain import ExposureGain
            global_index = indices[fusion_reference]
            reference_matrix = scaled_matrix(transforms[global_index], analysis_shapes[global_index],
                                             raw_reference.shape[:2], reference_analysis_shape, (height, width))
            gain_model = ExposureGain(raw_reference, reference_matrix, (height, width), fusion_reference)
        del raw_reference
        diagnostic("quality_variant", variant=variant, fusion_reference_index=fusion_reference)
        pass_number = 1

        def load_aligned(local):
            index, path = indices[local], paths[local]
            matrix = transforms[index] if index < len(transforms) else None
            with stage("quality_read_decode", path=str(path), frame=local, pass_index=pass_number):
                source = seeded_source.pop(local, None)
                if source is None:
                    source = load_source(path)
            source_analysis_shape = analysis_shapes[index]
            full_matrix = scaled_matrix(
                matrix, source_analysis_shape, source.shape[:2],
                reference_analysis_shape, (height, width),
            )
            if valid_sources is not None:
                valid_sources.register(local, full_matrix, source.shape[:2])
            if gain_model is not None:
                if pass_number == 2 and local not in gain_model.gains:
                    raise RuntimeError("second Quality pass lacks the first pass brightness gain")
                source = gain_model.apply(local, source, full_matrix)
            with stage("warp", frame=local, pass_index=pass_number):
                aligned = cv2.warpPerspective(
                    source, full_matrix, (width, height), flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            if residuals is not None and pass_number == 1:
                try:
                    residuals.measure(aligned, local, full_matrix, source.shape[:2])
                except Exception as exc:
                    diagnostic("registration_residual_unavailable", frame=local, reason=str(exc))
            return aligned

        if execution == "memory":
            from .quality_streaming import MemoryReferenceFrames
            cache = MemoryReferenceFrames(load_aligned, working_bytes=width * height * 80)
        else:
            from ..utils.shared_cache_budget import process_global_cache_budget
            cache = AlignedFrameCache(
                load_aligned, max_bytes=0 if execution == "streaming" else self.aligned_cache_bytes,
                working_bytes=width * height * 80,
                shared_budget=process_global_cache_budget(
                    int(_value(self.runtime_config, "fusion_cache_budget_bytes", 3 * 1024**3))),
            )
        try:
            texture = None
            if variant in {"gate", "clean"}:
                from .quality_fusion import FlatTextureStatistics
                texture = FlatTextureStatistics(
                    fusion_reference,
                    edge_mode=_value(self.runtime_config, "quality_gate_edge_mode", "coherent"),
                )
            compatibility_mask = None

            def observe_frame(index, rgb, gray, score):
                nonlocal compatibility_mask
                valid = None
                if valid_sources is not None:
                    valid = valid_sources.observe(index, score)
                if printed_guard is not None:
                    printed_guard.observe(index, rgb, gray, score)
                if surface_tone is not None:
                    surface_tone.observe(index, rgb)
                    surface_boundary.observe(index, rgb, score, valid=valid)
                if texture is not None:
                    texture.observe(index, rgb, gray, score)
                elif execution == "streaming" and index == 0:
                    from .quality_fusion import neutral_mask
                    compatibility_mask = neutral_mask(rgb, gray)

            frame_order = ([fusion_reference, *(i for i in range(len(paths)) if i != fusion_reference)]
                           if execution == "streaming" else None)
            diagnostic("quality_scan_configuration", execution=execution,
                       focus_order=frame_order or list(range(len(paths))),
                       rgb_order=list(range(len(paths))), prefetch=execution == "cached",
                       disk_read_concurrency=1 if decoder == "opencv" else "legacy",
                       version="quality-two-pass-v1" if execution != "cached" else "quality-cached-p1")
            with stage("quality_pass_1"):
                labels = build_focus_labels(
                    len(paths), cache.for_focus, cancel_event=cancel_event,
                    stabilize_background=True, frame_order=frame_order,
                    prefetch=execution == "cached", frame_observer=observe_frame,
                    # Apply the guard after the other label regularizers;
                    # they must not split its coherent ink/fringe coverage.
                    printed_edge_guard=False,
                )
            if cancel_event.is_set():
                raise RuntimeError("Quality fusion cancelled")
            if texture is None:
                if execution == "streaming":
                    from .quality_fusion import stabilize_neutral_mask
                    labels = stabilize_neutral_mask(labels, compatibility_mask)
                    del compatibility_mask
                else:
                    labels = stabilize_neutral_labels(labels, cache.peek(0))
            else:
                labels, protected = texture.apply(labels)
                del texture
                if variant == "clean":
                    from .quality_fusion import clean_tiny_labels
                    labels = clean_tiny_labels(labels, protected, cancel_event=cancel_event)
                del protected
            if surface_boundary is not None:
                labels = surface_boundary.apply(labels)
            if printed_guard is not None:
                labels = printed_guard.apply(labels, protected_texture=(
                    surface_boundary.texture_protection(labels.shape) if surface_boundary is not None else None))
            if valid_sources is not None:
                labels = valid_sources.apply(labels)
            pass_number = 2
            blend_prefetch = bool(_value(self.runtime_config, "quality_blend_prefetch", True))

            def load_blend_frame(index):
                rgb = cache.for_blend(index)
                # Remove each material's source drift before focus ownership
                # turns it into sharp contours. Pure reference cores and
                # continuous confidence avoid the old per-frame hard gates.
                if surface_tone is not None:
                    rgb = surface_tone.correct(rgb, index)
                return rgb
            with stage("quality_pass_2", second_decode=execution == "streaming",
                       second_warp=execution == "streaming",
                       blend_prefetch=blend_prefetch if execution != "streaming" else None):
                if execution == "streaming":
                    from .quality_streaming import blend_focus_narrow
                    result = blend_focus_narrow(len(paths), load_blend_frame, labels, cancel_event=cancel_event)
                else:
                    result = blend_focus_pyramid(
                        len(paths), load_blend_frame, labels, cancel_event=cancel_event,
                        prefetch=blend_prefetch,
                    )
        finally:
            _report_cache_then_clear(cache, "aligned_frame_cache")
        del labels
        if cancel_event.is_set():
            raise RuntimeError("Quality fusion cancelled")
        with stage("encoding"), Image.fromarray(result) as image:
            final = encode_image_output(image, output_path, config=output_config, original_path=reference_path)
        diagnostics = ("FULL_RESOLUTION_PREVIEW_ALIGNMENT", "NEUTRAL_EDGE_COHERENCE",
                       "CONVEX_FOCUS_BLEND", f"ALIGNED_CACHE_HITS:{cache.hits}/{len(paths)}",
                       f"ALIGNED_CACHE_PEAK_BYTES:{cache.peak_bytes}")
        if rebuilt_registration:
            diagnostics += ("PREVIEW_REGISTRATION_REBUILT",)
        if surface_tone is not None:
            diagnostics += ("MATERIAL_PAIRED_SURFACE_TONE_V17",)
        if valid_sources is not None:
            diagnostics += ("OPENCV_VALID_SOURCE_LOCAL_DETAIL_V20_ASTRA",)
        return FusionResult(Path(final), self.name, alignment_status="PREVIEW_TRANSFORMS", diagnostics=diagnostics)


__all__ = ["FusionBackend", "FusionResult", "HuginEnfuseBackend", "QualityFusionBackend"]
