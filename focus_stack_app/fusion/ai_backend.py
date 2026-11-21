"""StackMFF-V3 AI focus-fusion backend (ONNX Runtime DirectML / CUDA / CPU).

Reuses the existing scene grouping, coverage selection, and ECC alignment
transforms from ``GroupAnalyzer`` and replaces only the fusion engine.
Because StackMFF-V3 classifies per-pixel source probabilities directly over
observed input pixels, no post-fusion surface tone repair is applied.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .ai_runtime import (
    AiRuntimeStatus,
    StackMFFV3Runner,
    enforce_valid_probabilities,
    probe_ai_runtime,
)
from .backends import FusionBackend, FusionResult, _quality_metadata, _value
from ..hugin.output_encoder import OutputCollisionError, _same_path, encode_image_output
from ..utils.image_io import load_quality_rgb, load_rgb
from ..utils.performance import diagnostic, profiled_fusion, stage


VERSION = "ai-stackmff-v3-onnx-v1"


def _check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("AI fusion cancelled")


def _parse_scales(raw: Any) -> tuple[float, ...]:
    if isinstance(raw, (tuple, list)):
        values = [float(v) for v in raw]
    else:
        values = [float(part.strip()) for part in str(raw or "1.0,0.5,0.25").split(",") if part.strip()]
    if not values or any(v <= 0 or v > 2.0 for v in values):
        raise ValueError(f"Invalid ai_scales: {raw!r}")
    return tuple(values)


def _scaled_matrix(matrix, source_from, source_to, target_from, target_to) -> np.ndarray:
    if matrix is None:
        mat = np.eye(3, dtype=np.float32)
    else:
        mat = np.asarray(matrix, dtype=np.float32)
        if mat.shape == (2, 3):
            hom = np.eye(3, dtype=np.float32)
            hom[:2] = mat
            mat = hom
    sfh, sfw = source_from
    sth, stw = source_to
    tfh, tfw = target_from
    tth, ttw = target_to
    source_to_analysis = np.diag([sfw / max(1, stw), sfh / max(1, sth), 1.0]).astype(np.float32)
    analysis_to_target = np.diag([ttw / max(1, tfw), tth / max(1, tfh), 1.0]).astype(np.float32)
    return analysis_to_target @ mat @ source_to_analysis


def _select_tile_size(
    provider: str,
    precision: str,
    frame_count: int,
    configured_tile: int,
) -> tuple[int, int]:
    """Select (tile_size, overlap) tuned to avoid GPU VRAM paging."""
    if configured_tile > 0:
        tile = int(configured_tile)
    elif provider == "CUDA":
        if frame_count <= 24:
            tile = 1024
        elif frame_count <= 64:
            tile = 512
        else:
            tile = 384
    elif provider == "DIRECTML":
        if precision == "fp16":
            if frame_count <= 60:
                tile = 512
            elif frame_count <= 100:
                tile = 384
            else:
                tile = 256
        else:
            if frame_count <= 28:
                tile = 512
            elif frame_count <= 60:
                tile = 384
            else:
                tile = 256
    else:
        tile = 512
    overlap = 128 if tile >= 1024 else 64
    return tile, overlap


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = max(64, tile - overlap)
    starts = list(range(0, length - tile, stride))
    if not starts or starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def make_tile_weight(
    th: int,
    tw: int,
    overlap: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Smooth 2D raised-cosine feather window eliminating tile seams."""
    wy = np.ones(th, dtype=np.float32)
    wx = np.ones(tw, dtype=np.float32)
    if overlap > 0:
        eff_y = min(overlap, th // 2)
        eff_x = min(overlap, tw // 2)
        if eff_y > 0:
            ramp_y = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, eff_y + 2, dtype=np.float32)[1:-1])
            if y0 > 0:
                wy[:eff_y] *= ramp_y
            if y1 < height:
                wy[-eff_y:] *= ramp_y[::-1]
        if eff_x > 0:
            ramp_x = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, eff_x + 2, dtype=np.float32)[1:-1])
            if x0 > 0:
                wx[:eff_x] *= ramp_x
            if x1 < width:
                wx[-eff_x:] *= ramp_x[::-1]
    return wy[:, None] * wx[None, :]


class _FrameFootprint:
    """Compact per-frame photographed footprint geometry (zero full-res mask RAM).

    Matches ``ValidSourceOwnership`` semantics: pixels whose inverse-warped
    source coordinates lie inside ``[0, source_w - 1] x [0, source_h - 1]``
    are valid; reflected warp padding outside the source bounds is excluded.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        source_shape: tuple[int, int],
        target_shape: tuple[int, int],
    ) -> None:
        self.source_h, self.source_w = int(source_shape[0]), int(source_shape[1])
        self.target_h, self.target_w = int(target_shape[0]), int(target_shape[1])
        mat = np.asarray(matrix, dtype=np.float64)
        self.all_valid = False
        self.inv_matrix: np.ndarray | None = None
        self.is_affine = False
        if (
            self.source_h == self.target_h
            and self.source_w == self.target_w
            and np.allclose(mat, np.eye(3, dtype=np.float64), atol=1e-6)
        ):
            self.all_valid = True
            return
        try:
            inv = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            self.inv_matrix = None
            return
        self.inv_matrix = inv
        self.is_affine = (
            abs(inv[2, 0]) <= 1e-12
            and abs(inv[2, 1]) <= 1e-12
            and abs(inv[2, 2] - 1.0) <= 1e-12
        )
        if self._rect_all_valid(0, self.target_h, 0, self.target_w):
            self.all_valid = True

    def _rect_all_valid(self, y0: int, y1: int, x0: int, x1: int) -> bool:
        if self.all_valid:
            return True
        if self.inv_matrix is None or y1 <= y0 or x1 <= x0:
            return False
        corners = np.array(
            [
                [float(x0), float(y0), 1.0],
                [float(x1 - 1), float(y0), 1.0],
                [float(x0), float(y1 - 1), 1.0],
                [float(x1 - 1), float(y1 - 1), 1.0],
            ],
            dtype=np.float64,
        )
        mapped = corners @ self.inv_matrix.T
        w = mapped[:, 2]
        if not np.all(w > 1e-6):
            return False
        sx = mapped[:, 0] / w
        sy = mapped[:, 1] / w
        margin = 1e-5
        return bool(
            np.all(sx >= margin)
            and np.all(sx <= (self.source_w - 1.0) - margin)
            and np.all(sy >= margin)
            and np.all(sy <= (self.source_h - 1.0) - margin)
        )

    def tile_mask(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray | None:
        """Return bool [y1-y0, x1-x0] valid mask, or None when every pixel is valid."""
        if self.all_valid or self._rect_all_valid(y0, y1, x0, x1):
            return None
        th, tw = y1 - y0, x1 - x0
        if self.inv_matrix is None:
            return np.zeros((th, tw), dtype=bool)
        inv = self.inv_matrix
        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        eps = 1e-5
        max_x = (self.source_w - 1.0) + eps
        max_y = (self.source_h - 1.0) + eps
        if self.is_affine:
            sx = inv[0, 0] * xs[None, :] + inv[0, 1] * ys[:, None] + inv[0, 2]
            sy = inv[1, 0] * xs[None, :] + inv[1, 1] * ys[:, None] + inv[1, 2]
            return (sx >= -eps) & (sx <= max_x) & (sy >= -eps) & (sy <= max_y)
        w = inv[2, 0] * xs[None, :] + inv[2, 1] * ys[:, None] + inv[2, 2]
        valid = w > 1e-6
        safe_w = np.where(valid, w, 1.0)
        sx = (inv[0, 0] * xs[None, :] + inv[0, 1] * ys[:, None] + inv[0, 2]) / safe_w
        sy = (inv[1, 0] * xs[None, :] + inv[1, 1] * ys[:, None] + inv[1, 2]) / safe_w
        return valid & (sx >= -eps) & (sx <= max_x) & (sy >= -eps) & (sy <= max_y)

    def points_valid(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Evaluate footprint validity at sparse 1D coordinate arrays (ys, xs)."""
        if self.all_valid:
            return np.ones(ys.shape, dtype=bool)
        if self.inv_matrix is None:
            return np.zeros(ys.shape, dtype=bool)
        inv = self.inv_matrix
        xf = xs.astype(np.float64, copy=False)
        yf = ys.astype(np.float64, copy=False)
        eps = 1e-5
        max_x = (self.source_w - 1.0) + eps
        max_y = (self.source_h - 1.0) + eps
        if self.is_affine:
            sx = inv[0, 0] * xf + inv[0, 1] * yf + inv[0, 2]
            sy = inv[1, 0] * xf + inv[1, 1] * yf + inv[1, 2]
            return (sx >= -eps) & (sx <= max_x) & (sy >= -eps) & (sy <= max_y)
        w = inv[2, 0] * xf + inv[2, 1] * yf + inv[2, 2]
        valid = w > 1e-6
        safe_w = np.where(valid, w, 1.0)
        sx = (inv[0, 0] * xf + inv[0, 1] * yf + inv[0, 2]) / safe_w
        sy = (inv[1, 0] * xf + inv[1, 1] * yf + inv[1, 2]) / safe_w
        return valid & (sx >= -eps) & (sx <= max_x) & (sy >= -eps) & (sy <= max_y)


def _build_tile_valid_mask(
    footprints: list[_FrameFootprint | None],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray | None:
    """Build [N, th, tw] bool valid mask for a tile, or None if all frames cover it."""
    masks: list[np.ndarray | None] = []
    any_partial = False
    for fp in footprints:
        m = None if fp is None else fp.tile_mask(y0, y1, x0, x1)
        if m is not None:
            any_partial = True
        masks.append(m)
    if not any_partial:
        return None
    full = np.ones((len(footprints), y1 - y0, x1 - x0), dtype=bool)
    for idx, m in enumerate(masks):
        if m is not None:
            full[idx] = m
    return full


def _predict_tile_probabilities(
    runner: Any,
    gray_tile: np.ndarray,
    *,
    scales: tuple[float, ...],
    sigma: float,
    valid_mask: np.ndarray | None,
    cancel_event: threading.Event | None,
) -> np.ndarray:
    """Invoke runner with optional valid_mask and enforce valid-source probabilities."""
    try:
        probs = runner.predict_probabilities(
            gray_tile,
            scales=scales,
            sigma=sigma,
            valid_mask=valid_mask,
            cancel_event=cancel_event,
        )
    except TypeError:
        probs = runner.predict_probabilities(
            gray_tile,
            scales=scales,
            sigma=sigma,
            cancel_event=cancel_event,
        )
        if valid_mask is not None:
            probs = enforce_valid_probabilities(probs, valid_mask, in_place=True)
        return probs
    if valid_mask is not None and not isinstance(runner, StackMFFV3Runner):
        probs = enforce_valid_probabilities(probs, valid_mask, in_place=True)
    return probs


def _blend_tile_rgb(
    probs: np.ndarray,
    aligned_rgb: list[np.ndarray],
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    *,
    hard: bool,
    sparse_threshold: float = 1e-4,
    get_frame: Callable[[int], np.ndarray] | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Combine per-frame probabilities with aligned RGB crops (uint8 -> float32)."""
    layers, th, tw = probs.shape
    frame_at = get_frame if get_frame is not None else aligned_rgb.__getitem__
    if hard:
        if valid_mask is not None:
            any_valid = np.any(valid_mask, axis=0)
            scored = np.where((~valid_mask) & any_valid[None, :, :], -1.0, probs)
            winners = np.argmax(scored, axis=0)
        else:
            winners = np.argmax(probs, axis=0)
        tile_fused = np.empty((th, tw, 3), dtype=np.float32)
        for idx in range(layers):
            mask = winners == idx
            if np.any(mask):
                tile_fused[mask] = frame_at(idx)[y0:y1, x0:x1][mask]
        return tile_fused

    tile_fused = np.zeros((th, tw, 3), dtype=np.float32)
    weight_sum = np.zeros((th, tw), dtype=np.float32)
    for idx in range(layers):
        prob_i = probs[idx]
        if float(prob_i.max()) < sparse_threshold:
            continue
        tile_fused += prob_i[:, :, None] * frame_at(idx)[y0:y1, x0:x1]
        weight_sum += prob_i
    tile_fused /= np.maximum(weight_sum[:, :, None], 1e-6)
    return tile_fused


def _store_core_topk(
    probs: np.ndarray,
    top_indices: np.ndarray,
    top_weights: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    height: int,
    width: int,
    overlap: int,
    hard: bool,
    valid_mask: np.ndarray | None = None,
) -> None:
    """Write interior core of tile probabilities into full-res Top-K maps."""
    half = overlap // 2
    ly0 = half if y0 > 0 else 0
    ly1 = (y1 - y0) - half if y1 < height else (y1 - y0)
    lx0 = half if x0 > 0 else 0
    lx1 = (x1 - x0) - half if x1 < width else (x1 - x0)
    gy0, gy1 = y0 + ly0, y0 + ly1
    gx0, gx1 = x0 + lx0, x0 + lx1
    if gy1 <= gy0 or gx1 <= gx0:
        return

    core = probs[:, ly0:ly1, lx0:lx1]
    k = top_indices.shape[0]
    if hard or k == 1:
        if valid_mask is not None:
            v_core = valid_mask[:, ly0:ly1, lx0:lx1]
            any_valid = np.any(v_core, axis=0)
            scored = np.where((~v_core) & any_valid[None, :, :], -1.0, core)
            top_indices[0, gy0:gy1, gx0:gx1] = np.argmax(scored, axis=0).astype(np.uint16)
        else:
            top_indices[0, gy0:gy1, gx0:gx1] = np.argmax(core, axis=0).astype(np.uint16)
        top_weights[0, gy0:gy1, gx0:gx1] = np.float16(1.0)
        return

    layers = core.shape[0]
    if layers <= k:
        for idx in range(layers):
            top_indices[idx, gy0:gy1, gx0:gx1] = idx
            top_weights[idx, gy0:gy1, gx0:gx1] = core[idx].astype(np.float16)
        return

    part = np.argpartition(core, layers - k, axis=0)[-k:]
    vals = np.take_along_axis(core, part, axis=0)
    vals_sum = np.sum(vals, axis=0, keepdims=True)
    vals = vals / np.maximum(vals_sum, 1e-6)
    top_indices[:, gy0:gy1, gx0:gx1] = part.astype(np.uint16)
    top_weights[:, gy0:gy1, gx0:gx1] = vals.astype(np.float16)


class AiFusionBackend(FusionBackend):
    """Order-invariant StackMFF-V3 neural focus fusion without surface tone repair."""

    name = "ai"

    def __init__(
        self,
        *,
        runtime_config: Any = None,
        logger: logging.Logger | None = None,
        runner_factory: Callable[[AiRuntimeStatus], Any] | None = None,
        runtime_prober: Callable[..., AiRuntimeStatus] | None = None,
    ) -> None:
        self.runtime_config = (
            runtime_config.get("runtime", runtime_config)
            if isinstance(runtime_config, Mapping)
            else getattr(runtime_config, "runtime", runtime_config)
        )
        self.logger = logger or logging.getLogger(__name__)
        self._runner_factory = runner_factory
        self._runtime_prober = runtime_prober or probe_ai_runtime

    @profiled_fusion
    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event) -> FusionResult:
        from PIL import Image, ImageOps

        _check_cancel(cancel_event)
        paths = [Path(path) for path in _value(analysis, "selected_paths", ())]
        indices = list(_value(analysis, "selected_indices", range(len(paths))))
        transforms = list(_value(analysis, "preview_transforms", ()))
        analysis_shapes = list(_value(analysis, "analysis_shapes", ()))
        reference_index = int(_value(analysis, "preview_reference_index", indices[0] if indices else 0))
        if len(paths) < 2:
            raise ValueError("AI fusion requires at least two selected images")
        if len(indices) != len(paths) or any(index < 0 for index in indices):
            raise ValueError("AI selected indices must match selected paths")

        reference_path = Path(_value(analysis, "preview_reference", paths[0]))
        destination = Path(output_path)
        if any(_same_path(path, destination) for path in [*paths, reference_path]):
            raise OutputCollisionError(f"Output cannot overwrite an input image: {destination}")
        if destination.exists() and not _value(output_config, "overwrite", False):
            raise OutputCollisionError(f"Refusing to overwrite existing output: {destination}")

        requested_device = str(_value(self.runtime_config, "ai_device", "auto") or "auto")
        explicit_model = _value(self.runtime_config, "ai_model_path", None)
        scales = _parse_scales(_value(self.runtime_config, "ai_scales", "1.0,0.5,0.25"))
        sigma = float(_value(self.runtime_config, "ai_sigma", 0.5))
        hard = bool(_value(self.runtime_config, "ai_hard_selection", False))
        configured_tile = int(_value(self.runtime_config, "ai_tile_size", 0) or 0)
        cpu_proxy_edge = int(_value(self.runtime_config, "ai_cpu_proxy_long_edge", 1600) or 0)
        use_surface_tone = bool(_value(self.runtime_config, "ai_surface_tone", False))
        raw_group_cache = _value(self.runtime_config, "opencv_aligned_cache_bytes", None)
        group_cache_budget = 2 * 1024**3 if raw_group_cache is None else max(0, int(raw_group_cache))
        raw_global_cache = _value(self.runtime_config, "fusion_cache_budget_bytes", None)
        global_cache_budget = 4 * 1024**3 if raw_global_cache is None else max(0, int(raw_global_cache))
        effective_cache_budget = min(group_cache_budget, global_cache_budget)
        low_memory_mode = bool(_value(self.runtime_config, "low_memory_mode", False))

        status = self._runtime_prober(explicit_model, requested_device)
        if not status.available:
            raise RuntimeError(status.status_text or status.error or "AI fusion runtime is unavailable")

        reference_analysis_shape = _value(analysis, "reference_analysis_shape", None)
        if not reference_analysis_shape:
            reference_analysis_shape = (
                analysis_shapes[reference_index]
                if 0 <= reference_index < len(analysis_shapes)
                else None
            )

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
                _check_cancel(cancel_event)
                if path.resolve() == reference_path.resolve():
                    preview = ref
                    matrix = np.eye(3, dtype=np.float32)
                else:
                    preview = load_rgb(path, 640)
                    registration = register_images(ref, preview)
                    if not registration.valid:
                        raise RuntimeError(f"AI fusion could not rebuild alignment for {path}: {registration.message}")
                    matrix = registration.matrix
                transforms[index] = matrix
                analysis_shapes[index] = preview.shape[:2]
            del ref, preview

        with Image.open(reference_path) as reference_image:
            width, height = reference_image.size
            tags, orientation, icc = _quality_metadata(reference_image)

        identity_matrix = np.eye(3, dtype=np.float32)
        decoder = str(_value(self.runtime_config, "quality_jpeg_decoder", "opencv") or "opencv")
        footprints: list[_FrameFootprint | None] = [None] * len(paths)

        def load_aligned_rgb(local_idx: int, pass_index: int = 1) -> np.ndarray:
            index, path = indices[local_idx], paths[local_idx]
            matrix = transforms[index] if index < len(transforms) else None
            with stage("ai_read_decode", path=str(path), frame=local_idx, pass_index=pass_index):
                if decoder == "opencv":
                    source = load_quality_rgb(path, pass_index=pass_index)
                else:
                    source = load_rgb(path)
            full_matrix = _scaled_matrix(
                matrix,
                analysis_shapes[index],
                source.shape[:2],
                reference_analysis_shape,
                (height, width),
            )
            if pass_index == 1 or footprints[local_idx] is None:
                footprints[local_idx] = _FrameFootprint(full_matrix, source.shape[:2], (height, width))
            with stage("ai_warp", frame=local_idx, pass_index=pass_index):
                if (
                    source.shape[0] == height
                    and source.shape[1] == width
                    and np.allclose(full_matrix, identity_matrix, atol=1e-6)
                ):
                    return source
                return cv2.warpPerspective(
                    source,
                    full_matrix,
                    (width, height),
                    flags=cv2.INTER_LANCZOS4,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

        use_cpu_proxy = (
            status.active_provider == "CPU"
            and cpu_proxy_edge > 0
            and max(height, width) > cpu_proxy_edge
        )
        effective_scales = (1.0,) if (use_cpu_proxy and scales in {(1.0, 0.75), (1.0, 0.5, 0.25)}) else scales
        tile_size, overlap = _select_tile_size(
            status.active_provider, status.precision, len(paths), configured_tile
        )
        needed_cache_bytes = len(paths) * height * width * (3 if use_cpu_proxy else 4)
        shared_budget = None
        reserved_shared_bytes = 0
        allow_rgb_cache = (
            not low_memory_mode
            and effective_cache_budget > 0
            and needed_cache_bytes <= effective_cache_budget
        )
        if allow_rgb_cache:
            from ..utils.shared_cache_budget import process_global_cache_budget

            shared_budget = process_global_cache_budget(global_cache_budget)
            if shared_budget.try_reserve(needed_cache_bytes):
                reserved_shared_bytes = needed_cache_bytes
            else:
                allow_rgb_cache = False
        use_low_memory_two_pass = (not use_cpu_proxy) and (not allow_rgb_cache)
        selected_reference = next(
            (i for i, path in enumerate(paths) if _same_path(path, reference_path)),
            None,
        )
        fusion_reference = selected_reference if selected_reference is not None else 0
        surface_tone = None
        if use_surface_tone:
            from .surface_tone import SurfaceToneHarmonizer

            surface_tone = SurfaceToneHarmonizer(fusion_reference)

        diagnostic(
            "ai_configuration",
            version=VERSION,
            provider=status.active_provider,
            ort_provider=status.ort_provider,
            precision=status.precision,
            model_path=str(status.model_path) if status.model_path else "",
            host_cuda12_detected=status.host_cuda12_detected,
            input_count=len(paths),
            dimensions=[width, height],
            scales=list(effective_scales),
            sigma=0.0 if hard else sigma,
            hard_selection=hard,
            tile_size=tile_size,
            overlap=overlap,
            cpu_proxy=use_cpu_proxy,
            two_pass_low_memory=use_low_memory_two_pass,
            group_cache_budget_bytes=group_cache_budget,
            global_cache_budget_bytes=global_cache_budget,
            cache_rgb_enabled=allow_rgb_cache,
            surface_tone=use_surface_tone,
            fusion_reference_index=fusion_reference,
        )

        runner = None
        t_infer_total = 0.0
        t_blend_total = 0.0
        t_tone_total = 0.0
        tiles_run = 0
        try:
            runner = (
                self._runner_factory(status)
                if self._runner_factory is not None
                else StackMFFV3Runner(status)
            )
            if use_cpu_proxy:
                result, t_infer_total, t_blend_total, t_tone_total = self._fuse_cpu_proxy(
                    runner,
                    len(paths),
                    height,
                    width,
                    load_aligned_rgb,
                    footprints,
                    fusion_reference=fusion_reference,
                    proxy_long_edge=cpu_proxy_edge,
                    scales=effective_scales,
                    sigma=0.0 if hard else sigma,
                    hard=hard,
                    cache_rgb=allow_rgb_cache,
                    surface_tone=surface_tone,
                    cancel_event=cancel_event,
                )
            elif use_low_memory_two_pass:
                result, tiles_run, t_infer_total, t_blend_total, t_tone_total = self._fuse_two_pass_tiled(
                    runner,
                    len(paths),
                    height,
                    width,
                    load_aligned_rgb,
                    footprints,
                    fusion_reference=fusion_reference,
                    tile_size=tile_size,
                    overlap=overlap,
                    scales=effective_scales,
                    sigma=0.0 if hard else sigma,
                    hard=hard,
                    surface_tone=surface_tone,
                    cancel_event=cancel_event,
                )
            else:
                result, tiles_run, t_infer_total, t_blend_total, t_tone_total = self._fuse_cached_tiled(
                    runner,
                    len(paths),
                    height,
                    width,
                    load_aligned_rgb,
                    footprints,
                    tile_size=tile_size,
                    overlap=overlap,
                    scales=effective_scales,
                    sigma=0.0 if hard else sigma,
                    hard=hard,
                    surface_tone=surface_tone,
                    cancel_event=cancel_event,
                )
        finally:
            if runner is not None and hasattr(runner, "close"):
                runner.close()
            if shared_budget is not None and reserved_shared_bytes > 0:
                shared_budget.release(reserved_shared_bytes)

        _check_cancel(cancel_event)
        with stage("encoding"), Image.fromarray(result) as image:
            if orientation in range(2, 9):
                image.getexif()[274] = orientation
                ImageOps.exif_transpose(image, in_place=True)
            if tags or orientation != 1:
                tags[274] = 1
                image.info["exif"] = tags.tobytes()
                image.info["tiff_capture_tags"] = tags
            if icc:
                image.info["icc_profile"] = icc
            final = encode_image_output(image, destination, config=output_config, original_path=reference_path)

        masked_frames = sum(1 for fp in footprints if fp is not None and not fp.all_valid)
        diag_items: list[str] = [
            VERSION,
            f"AI_PROVIDER:{status.active_provider}",
            f"AI_PRECISION:{status.precision.upper()}",
            f"AI_VALID_SOURCE_MASK:{masked_frames}/{len(paths)}",
            "MATERIAL_PAIRED_SURFACE_TONE_V17" if use_surface_tone else "AI_NO_TONE_REPAIR",
            f"AI_SCALES:{','.join(f'{s:g}' for s in effective_scales)}",
            f"AI_INFER_SECONDS:{t_infer_total:.2f}",
            f"AI_BLEND_SECONDS:{t_blend_total:.2f}",
        ]
        if use_surface_tone:
            diag_items.append(f"AI_TONE_SECONDS:{t_tone_total:.2f}")
        if use_cpu_proxy:
            diag_items.append(f"AI_CPU_PROXY_FALLBACK:{cpu_proxy_edge}")
        else:
            diag_items.append(f"AI_TILES:{tile_size}x{tile_size}x{tiles_run}")
            if use_low_memory_two_pass:
                diag_items.append("AI_LOW_MEMORY_TOPK_2PASS")
        if hard:
            diag_items.append("AI_HARD_SELECTION")
        if rebuilt_registration:
            diag_items.append("PREVIEW_REGISTRATION_REBUILT")
        if status.active_provider == "DIRECTML" and status.host_cuda12_detected:
            diag_items.append("AI_HOST_CUDA12_AVAILABLE_FOR_ORT_GPU")

        return FusionResult(
            Path(final),
            self.name,
            alignment_status="PREVIEW_TRANSFORMS",
            diagnostics=tuple(diag_items),
        )

    def _fuse_cached_tiled(
        self,
        runner: Any,
        frame_count: int,
        height: int,
        width: int,
        load_aligned_rgb: Callable[[int, int], np.ndarray],
        footprints: list[_FrameFootprint | None],
        *,
        tile_size: int,
        overlap: int,
        scales: tuple[float, ...],
        sigma: float,
        hard: bool,
        surface_tone: Any = None,
        cancel_event: threading.Event | None,
    ) -> tuple[np.ndarray, int, float, float, float]:
        aligned_rgb: list[np.ndarray] = []
        aligned_gray: list[np.ndarray] = []
        t_tone = 0.0
        with stage("ai_align_stack", frames=frame_count):
            for idx in range(frame_count):
                _check_cancel(cancel_event)
                rgb = load_aligned_rgb(idx, 1)
                if surface_tone is not None:
                    tt0 = time.perf_counter()
                    surface_tone.observe(idx, rgb)
                    t_tone += time.perf_counter() - tt0
                aligned_rgb.append(rgb)
                aligned_gray.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

        corrected = [False] * frame_count

        def get_toned_rgb(idx: int) -> np.ndarray:
            nonlocal t_tone
            if surface_tone is not None and not corrected[idx]:
                tt0 = time.perf_counter()
                aligned_rgb[idx] = surface_tone.correct(aligned_rgb[idx], idx, strip_rows=256)
                t_tone += time.perf_counter() - tt0
                corrected[idx] = True
            return aligned_rgb[idx]

        acc_rgb = np.zeros((height, width, 3), dtype=np.float32)
        acc_weight = np.zeros((height, width), dtype=np.float32)
        ys = tile_starts(height, tile_size, overlap)
        xs = tile_starts(width, tile_size, overlap)
        tiles_run = 0
        t_infer = 0.0
        t_blend = 0.0

        for y0 in ys:
            y1 = min(height, y0 + tile_size)
            for x0 in xs:
                _check_cancel(cancel_event)
                x1 = min(width, x0 + tile_size)
                th, tw = y1 - y0, x1 - x0
                tiles_run += 1
                tile_valid = _build_tile_valid_mask(footprints, y0, y1, x0, x1)

                t0 = time.perf_counter()
                with stage("ai_infer_tile", tile=tiles_run, shape=[th, tw]):
                    gray_tile = np.empty((frame_count, th, tw), dtype=np.float32)
                    for idx in range(frame_count):
                        gray_tile[idx] = aligned_gray[idx][y0:y1, x0:x1]
                    gray_tile *= (1.0 / 255.0)
                    probs = _predict_tile_probabilities(
                        runner,
                        gray_tile,
                        scales=scales,
                        sigma=sigma,
                        valid_mask=tile_valid,
                        cancel_event=cancel_event,
                    )
                t_infer += time.perf_counter() - t0

                t1 = time.perf_counter()
                t_tone_before = t_tone
                with stage("ai_blend_tile", tile=tiles_run):
                    tile_fused = _blend_tile_rgb(
                        probs,
                        aligned_rgb,
                        y0,
                        y1,
                        x0,
                        x1,
                        hard=hard,
                        get_frame=get_toned_rgb if surface_tone is not None else None,
                        valid_mask=tile_valid,
                    )
                    w2d = make_tile_weight(th, tw, overlap, y0, y1, x0, x1, height, width)
                    acc_rgb[y0:y1, x0:x1] += tile_fused * w2d[:, :, None]
                    acc_weight[y0:y1, x0:x1] += w2d
                t_blend += (time.perf_counter() - t1) - (t_tone - t_tone_before)

        aligned_rgb.clear()
        aligned_gray.clear()
        fused = np.clip(np.round(acc_rgb / np.maximum(acc_weight[:, :, None], 1e-6)), 0, 255).astype(np.uint8)
        return fused, tiles_run, t_infer, max(0.0, t_blend), t_tone

    def _fuse_two_pass_tiled(
        self,
        runner: Any,
        frame_count: int,
        height: int,
        width: int,
        load_aligned_rgb: Callable[[int, int], np.ndarray],
        footprints: list[_FrameFootprint | None],
        *,
        fusion_reference: int,
        tile_size: int,
        overlap: int,
        scales: tuple[float, ...],
        sigma: float,
        hard: bool,
        surface_tone: Any = None,
        cancel_event: threading.Event | None,
    ) -> tuple[np.ndarray, int, float, float, float]:
        """Low-memory 2-pass tiled fusion storing only Top-4 per-pixel owners."""
        aligned_gray: list[np.ndarray] = []
        t_tone = 0.0
        with stage("ai_pass_1_gray", frames=frame_count):
            for idx in range(frame_count):
                _check_cancel(cancel_event)
                rgb = load_aligned_rgb(idx, 1)
                if surface_tone is not None:
                    tt0 = time.perf_counter()
                    surface_tone.observe(idx, rgb)
                    t_tone += time.perf_counter() - tt0
                aligned_gray.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
                del rgb

        top_k = 1 if hard else min(4, frame_count)
        top_indices = np.zeros((top_k, height, width), dtype=np.uint16)
        top_weights = np.zeros((top_k, height, width), dtype=np.float16)

        ys = tile_starts(height, tile_size, overlap)
        xs = tile_starts(width, tile_size, overlap)
        tiles_run = 0
        t_infer = 0.0
        for y0 in ys:
            y1 = min(height, y0 + tile_size)
            for x0 in xs:
                _check_cancel(cancel_event)
                x1 = min(width, x0 + tile_size)
                th, tw = y1 - y0, x1 - x0
                tiles_run += 1
                tile_valid = _build_tile_valid_mask(footprints, y0, y1, x0, x1)

                t0 = time.perf_counter()
                with stage("ai_infer_tile", tile=tiles_run, shape=[th, tw]):
                    gray_tile = np.empty((frame_count, th, tw), dtype=np.float32)
                    for idx in range(frame_count):
                        gray_tile[idx] = aligned_gray[idx][y0:y1, x0:x1]
                    gray_tile *= (1.0 / 255.0)
                    probs = _predict_tile_probabilities(
                        runner,
                        gray_tile,
                        scales=scales,
                        sigma=sigma,
                        valid_mask=tile_valid,
                        cancel_event=cancel_event,
                    )
                t_infer += time.perf_counter() - t0
                _store_core_topk(
                    probs,
                    top_indices,
                    top_weights,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    height=height,
                    width=width,
                    overlap=overlap,
                    hard=hard,
                    valid_mask=tile_valid,
                )
        aligned_gray.clear()

        t1 = time.perf_counter()
        t_tone_before = t_tone
        acc_rgb = np.zeros((height, width, 3), dtype=np.float32)
        acc_w = np.zeros((height, width), dtype=np.float32)
        positive_slots = top_weights > 0
        used_frames = set(np.unique(top_indices[positive_slots]).tolist()) if np.any(positive_slots) else {fusion_reference}
        with stage("ai_pass_2_blend", used_frames=len(used_frames), total_frames=frame_count):
            for idx in range(frame_count):
                if idx not in used_frames:
                    continue
                _check_cancel(cancel_event)
                rgb = load_aligned_rgb(idx, 2)
                if surface_tone is not None:
                    tt0 = time.perf_counter()
                    rgb = surface_tone.correct(rgb, idx, strip_rows=256)
                    t_tone += time.perf_counter() - tt0
                weight_map = np.zeros((height, width), dtype=np.float32)
                has_any = False
                for slot in range(top_k):
                    mask = (top_indices[slot] == idx) & (top_weights[slot] > 0)
                    if np.any(mask):
                        has_any = True
                        weight_map[mask] += top_weights[slot][mask].astype(np.float32)
                if has_any:
                    acc_rgb += weight_map[:, :, None] * rgb
                    acc_w += weight_map
                del rgb
        t_blend = (time.perf_counter() - t1) - (t_tone - t_tone_before)
        fused = np.clip(np.round(acc_rgb / np.maximum(acc_w[:, :, None], 1e-6)), 0, 255).astype(np.uint8)
        return fused, tiles_run, t_infer, max(0.0, t_blend), t_tone

    def _fuse_cpu_proxy(
        self,
        runner: Any,
        frame_count: int,
        height: int,
        width: int,
        load_aligned_rgb: Callable[[int, int], np.ndarray],
        footprints: list[_FrameFootprint | None],
        *,
        fusion_reference: int,
        proxy_long_edge: int,
        scales: tuple[float, ...],
        sigma: float,
        hard: bool,
        cache_rgb: bool,
        surface_tone: Any = None,
        cancel_event: threading.Event | None,
    ) -> tuple[np.ndarray, float, float, float]:
        scale = min(1.0, float(proxy_long_edge) / float(max(height, width)))
        ph = max(64, int(round(height * scale)))
        pw = max(64, int(round(width * scale)))

        cached: list[np.ndarray] | None = [] if cache_rgb else None
        proxy_gray = np.empty((frame_count, ph, pw), dtype=np.float32)
        proxy_valid_layers: list[np.ndarray | None] = [None] * frame_count
        any_partial = False
        t_tone = 0.0
        with stage("ai_cpu_proxy_align", frames=frame_count, proxy_shape=[ph, pw]):
            for idx in range(frame_count):
                _check_cancel(cancel_event)
                rgb = load_aligned_rgb(idx, 1)
                if surface_tone is not None:
                    tt0 = time.perf_counter()
                    surface_tone.observe(idx, rgb)
                    t_tone += time.perf_counter() - tt0
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                proxy_gray[idx] = cv2.resize(gray, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)
                fp = footprints[idx]
                if fp is not None and not fp.all_valid:
                    full_valid = fp.tile_mask(0, height, 0, width)
                    if full_valid is not None:
                        any_partial = True
                        invalid_f = (~full_valid).astype(np.float32)
                        proxy_valid_layers[idx] = (
                            cv2.resize(invalid_f, (pw, ph), interpolation=cv2.INTER_AREA) == 0.0
                        )
                        del full_valid, invalid_f
                if cached is not None:
                    cached.append(rgb)
                else:
                    del rgb

        proxy_valid: np.ndarray | None = None
        if any_partial:
            proxy_valid = np.ones((frame_count, ph, pw), dtype=bool)
            for idx, layer in enumerate(proxy_valid_layers):
                if layer is not None:
                    proxy_valid[idx] = layer
        proxy_valid_layers.clear()

        t0 = time.perf_counter()
        with stage("ai_cpu_proxy_infer", proxy_shape=[ph, pw]):
            probs_proxy = _predict_tile_probabilities(
                runner,
                proxy_gray,
                scales=scales,
                sigma=sigma,
                valid_mask=proxy_valid,
                cancel_event=cancel_event,
            )
        t_infer = time.perf_counter() - t0
        del proxy_gray

        def get_blend_rgb(idx: int) -> np.ndarray:
            nonlocal t_tone
            rgb = cached[idx] if cached is not None else load_aligned_rgb(idx, 2)
            if surface_tone is not None:
                tt0 = time.perf_counter()
                rgb = surface_tone.correct(rgb, idx, strip_rows=256)
                t_tone += time.perf_counter() - tt0
            return rgb

        t1 = time.perf_counter()
        t_tone_before = t_tone
        with stage("ai_cpu_proxy_blend"):
            if hard:
                if proxy_valid is not None:
                    any_vp = np.any(proxy_valid, axis=0)
                    scored = np.where((~proxy_valid) & any_vp[None, :, :], -1.0, probs_proxy)
                    winners_proxy = np.argmax(scored, axis=0).astype(np.uint16)
                else:
                    winners_proxy = np.argmax(probs_proxy, axis=0).astype(np.uint16)
                winners = cv2.resize(winners_proxy, (width, height), interpolation=cv2.INTER_NEAREST)
                if any_partial:
                    bad_mask = np.zeros((height, width), dtype=bool)
                    for idx in range(frame_count):
                        fp = footprints[idx]
                        if fp is None or fp.all_valid:
                            continue
                        sel = winners == idx
                        if not np.any(sel):
                            continue
                        fv = fp.tile_mask(0, height, 0, width)
                        if fv is not None:
                            bad_mask |= sel & (~fv)
                    if np.any(bad_mask):
                        by, bx = np.nonzero(bad_mask)
                        py = np.clip((by * ph) // height, 0, ph - 1)
                        px = np.clip((bx * pw) // width, 0, pw - 1)
                        best_score = np.full(by.size, -1.0, dtype=np.float32)
                        best_idx = winners[by, bx].copy()
                        for idx in range(frame_count):
                            fp = footprints[idx]
                            v_pts = np.ones(by.size, dtype=bool) if fp is None else fp.points_valid(by, bx)
                            sc = probs_proxy[idx, py, px]
                            better = v_pts & (sc > best_score)
                            best_score[better] = sc[better]
                            best_idx[better] = idx
                        winners[by, bx] = best_idx
                fused = np.zeros((height, width, 3), dtype=np.uint8)
                for idx in range(frame_count):
                    mask = winners == idx
                    if not np.any(mask):
                        continue
                    _check_cancel(cancel_event)
                    rgb = get_blend_rgb(idx)
                    fused[mask] = rgb[mask]
            else:
                acc_rgb = np.zeros((height, width, 3), dtype=np.float32)
                acc_w = np.zeros((height, width), dtype=np.float32)
                for idx in range(frame_count):
                    p_small = probs_proxy[idx]
                    if float(p_small.max()) < 1e-4:
                        continue
                    _check_cancel(cancel_event)
                    p_full = cv2.resize(p_small, (width, height), interpolation=cv2.INTER_LINEAR)
                    fp = footprints[idx]
                    if fp is not None and not fp.all_valid:
                        fv = fp.tile_mask(0, height, 0, width)
                        if fv is not None:
                            p_full[~fv] = 0.0
                            del fv
                    if float(p_full.max()) < 1e-6:
                        continue
                    rgb = get_blend_rgb(idx)
                    acc_rgb += p_full[:, :, None] * rgb
                    acc_w += p_full
                if any_partial:
                    uncovered = acc_w <= 1e-6
                    if np.any(uncovered):
                        uy, ux = np.nonzero(uncovered)
                        remaining = np.ones(uy.size, dtype=bool)
                        order = [fusion_reference, *(i for i in range(frame_count) if i != fusion_reference)]
                        for idx in order:
                            if not np.any(remaining):
                                break
                            fp = footprints[idx]
                            v_pts = np.ones(uy.size, dtype=bool) if fp is None else fp.points_valid(uy, ux)
                            take = remaining & v_pts
                            if not np.any(take):
                                continue
                            _check_cancel(cancel_event)
                            rgb = get_blend_rgb(idx)
                            ty, tx = uy[take], ux[take]
                            acc_rgb[ty, tx] = rgb[ty, tx]
                            acc_w[ty, tx] = 1.0
                            remaining[take] = False
                        if np.any(remaining):
                            rgb = get_blend_rgb(fusion_reference)
                            ty, tx = uy[remaining], ux[remaining]
                            acc_rgb[ty, tx] = rgb[ty, tx]
                            acc_w[ty, tx] = 1.0
                fused = np.clip(np.round(acc_rgb / np.maximum(acc_w[:, :, None], 1e-6)), 0, 255).astype(np.uint8)
        t_blend = (time.perf_counter() - t1) - (t_tone - t_tone_before)
        if cached is not None:
            cached.clear()
        return fused, t_infer, max(0.0, t_blend), t_tone


__all__ = [
    "AiFusionBackend",
    "VERSION",
    "make_tile_weight",
    "tile_starts",
]
