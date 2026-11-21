"""Lightweight ONNX Runtime execution layer for StackMFF-V3 AI fusion.

Default Windows runtime uses ``onnxruntime-directml`` (~25 MB) with an 835 KB
FP16 ONNX graph for GPU acceleration across NVIDIA, AMD, and Intel GPUs.
When ``onnxruntime-gpu`` with working CUDA 12 is installed in the environment,
``CUDAExecutionProvider`` is automatically prioritized. CPU fallback always
uses the FP32 ONNX graph (1.20 MB) to avoid slow emulated FP16 on CPUs.

All ``onnxruntime`` imports are strictly lazy so non-AI backends work without it.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import sys
import threading
from typing import Any, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

FP32_MODEL_FILENAME = "stackmff_v3_final.onnx"
FP16_MODEL_FILENAME = "stackmff_v3_final_fp16.onnx"

_PROVIDER_NAMES = {
    "CUDA": "CUDAExecutionProvider",
    "DIRECTML": "DmlExecutionProvider",
    "CPU": "CPUExecutionProvider",
}

_GPU_INFERENCE_LOCK = threading.Lock()
_PROBE_LOCK = threading.Lock()
_PROBE_CACHE: dict[tuple[str | None, str], "AiRuntimeStatus"] = {}


@dataclass(frozen=True, slots=True)
class HostCudaInfo:
    cuda12_detected: bool
    nvidia_driver_detected: bool
    detail: str = ""


@dataclass(frozen=True, slots=True)
class AiRuntimeStatus:
    available: bool
    active_provider: str  # "CUDA", "DIRECTML", "CPU", or "NONE"
    ort_provider: str  # "CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider", or ""
    precision: str  # "fp16", "fp32", or ""
    model_path: Path | None
    available_providers: tuple[str, ...] = ()
    host_cuda12_detected: bool = False
    host_cuda_detail: str = ""
    status_text: str = ""
    error: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.active_provider in {"CUDA", "DIRECTML"}


def weights_directory() -> Path:
    """Locate bundled StackMFF-V3 ONNX weights in source or frozen builds."""
    local = Path(__file__).resolve().parent / "weights"
    if local.is_dir():
        return local
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        bundled = Path(meipass) / "focus_stack_app" / "fusion" / "weights"
        if bundled.is_dir():
            return bundled
    return local


def resolve_model_path(precision: str = "fp16", explicit_path: str | Path | None = None) -> Path | None:
    """Resolve explicit or bundled ONNX model path for the requested precision."""
    if explicit_path is not None:
        candidate = Path(explicit_path)
        return candidate if candidate.is_file() else None
    root = weights_directory()
    if precision == "fp16":
        fp16 = root / FP16_MODEL_FILENAME
        if fp16.is_file():
            return fp16
        fp32 = root / FP32_MODEL_FILENAME
        return fp32 if fp32.is_file() else None
    fp32 = root / FP32_MODEL_FILENAME
    return fp32 if fp32.is_file() else None


def detect_host_cuda12() -> HostCudaInfo:
    """Fast, non-blocking check for local NVIDIA driver and CUDA 12 toolkit."""
    cuda12_paths: list[str] = []
    for key, value in os.environ.items():
        if not value:
            continue
        upper = key.upper()
        if upper == "CUDA_PATH" or upper.startswith("CUDA_PATH_V12"):
            path = Path(value)
            if "12." in path.name or upper.startswith("CUDA_PATH_V12"):
                if path.exists():
                    cuda12_paths.append(str(path))
    if not cuda12_paths and os.name == "nt":
        toolkit_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if toolkit_root.is_dir():
            for child in sorted(toolkit_root.glob("v12.*"), reverse=True):
                if child.is_dir():
                    cuda12_paths.append(str(child))
                    break
    if not cuda12_paths:
        nvcc = shutil.which("nvcc")
        if nvcc and ("v12." in nvcc.lower() or "cuda-12" in nvcc.lower()):
            cuda12_paths.append(nvcc)

    nvidia_driver = False
    if os.name == "nt":
        sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        nvidia_driver = (sys32 / "nvcuda.dll").is_file()
        if not cuda12_paths and (sys32 / "cudart64_12.dll").is_file():
            cuda12_paths.append(str(sys32 / "cudart64_12.dll"))
    else:
        nvidia_driver = shutil.which("nvidia-smi") is not None

    if cuda12_paths:
        return HostCudaInfo(True, nvidia_driver or True, cuda12_paths[0])
    return HostCudaInfo(False, nvidia_driver, "nvcuda.dll" if nvidia_driver else "")


def _import_ort() -> Any | None:
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
        return ort
    except ImportError:
        return None


def _create_ort_session(
    ort: Any,
    model_path: Path,
    provider_kind: str,
    *,
    shape: tuple[int, int, int] | None = None,
    threads: int = 0,
) -> Any:
    """Create one ORT InferenceSession and verify the requested provider is active."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.enable_cpu_mem_arena = False
    if threads > 0:
        opts.intra_op_num_threads = int(threads)

    if provider_kind == "DIRECTML":
        if shape is not None:
            for name, value in zip(("layers", "height", "width"), shape):
                opts.add_free_dimension_override_by_name(name, int(value))
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers: list[Any] = [("DmlExecutionProvider", {"device_id": 0})]
    elif provider_kind == "CUDA":
        providers = [("CUDAExecutionProvider", {"device_id": 0})]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    expected = _PROVIDER_NAMES[provider_kind]
    active = tuple(session.get_providers())
    if expected not in active:
        raise RuntimeError(f"Requested {expected} is not active (got {active})")
    return session


def _warmup_probe(ort: Any, model_path: Path, provider_kind: str) -> None:
    shape = (2, 64, 64)
    session = _create_ort_session(ort, model_path, provider_kind, shape=shape)
    try:
        dummy = np.zeros((1, 2, 64, 64), dtype=np.float32)
        out = session.run(["logits"], {"input": dummy})[0]
        if out.shape != (1, 2, 64, 64) or not np.isfinite(out).all():
            raise RuntimeError(f"Unexpected warmup output shape/values from {provider_kind}: {out.shape}")
    finally:
        del session


def _provider_order(requested_device: str) -> tuple[str, ...]:
    mode = str(requested_device or "auto").strip().casefold()
    if mode == "cpu":
        return ("CPU",)
    if mode == "dml":
        return ("DIRECTML", "CPU")
    if mode == "cuda":
        return ("CUDA", "DIRECTML", "CPU")
    return ("CUDA", "DIRECTML", "CPU")


def probe_ai_runtime(
    model_path: str | Path | None = None,
    requested_device: str = "auto",
    *,
    refresh: bool = False,
) -> AiRuntimeStatus:
    """Probe ONNX Runtime availability, active accelerator, and model weights.

    Provider priority:
    1. ``CUDAExecutionProvider`` (when ``onnxruntime-gpu`` + CUDA 12 are usable)
    2. ``DmlExecutionProvider`` (default via ``onnxruntime-directml``)
    3. ``CPUExecutionProvider`` (FP32 model fallback)
    """
    explicit_key = str(Path(model_path)) if model_path is not None else None
    device_key = str(requested_device or "auto").strip().casefold()
    cache_key = (explicit_key, device_key)
    with _PROBE_LOCK:
        if not refresh and cache_key in _PROBE_CACHE:
            return _PROBE_CACHE[cache_key]

    host_cuda = detect_host_cuda12()
    ort = _import_ort()
    if ort is None:
        hint = (
            "未安装 ONNX Runtime（默认请安装 onnxruntime-directml；检测到本机具备 CUDA 12，也可安装 onnxruntime-gpu 启用 CUDAExecutionProvider）"
            if host_cuda.cuda12_detected
            else "未安装 ONNX Runtime（请运行 pip install onnxruntime-directml）"
        )
        status = AiRuntimeStatus(
            available=False,
            active_provider="NONE",
            ort_provider="",
            precision="",
            model_path=None,
            available_providers=(),
            host_cuda12_detected=host_cuda.cuda12_detected,
            host_cuda_detail=host_cuda.detail,
            status_text=hint,
            error="onnxruntime is not installed",
        )
        with _PROBE_LOCK:
            _PROBE_CACHE[cache_key] = status
        return status

    available_providers = tuple(ort.get_available_providers())
    errors: list[str] = []

    for kind in _provider_order(device_key):
        ort_name = _PROVIDER_NAMES[kind]
        if ort_name not in available_providers:
            continue
        preferred_precision = "fp16" if kind in {"CUDA", "DIRECTML"} else "fp32"
        resolved = resolve_model_path(preferred_precision, model_path)
        if resolved is None:
            errors.append(f"{kind}: missing model file")
            continue
        actual_precision = "fp16" if "fp16" in resolved.name.lower() else "fp32"
        # Never run FP16 on CPU when falling back automatically.
        if kind == "CPU" and model_path is None and actual_precision != "fp32":
            errors.append("CPU: FP32 model missing")
            continue
        try:
            _warmup_probe(ort, resolved, kind)
        except Exception as exc:
            LOGGER.debug("AI provider %s warmup failed: %s", kind, exc)
            errors.append(f"{kind}: {exc}")
            continue

        if kind == "CUDA":
            text = f"已优先启用 CUDAExecutionProvider 加速（CUDA 12 · {actual_precision.upper()}）"
        elif kind == "DIRECTML":
            if host_cuda.cuda12_detected:
                text = (
                    f"已启用 DirectML GPU 加速（{actual_precision.upper()} · 默认轻量引擎；"
                    "检测到本机具备 CUDA 12，若安装 onnxruntime-gpu 将自动优先启用 CUDAExecutionProvider）"
                )
            else:
                text = (
                    f"已启用 DirectML GPU 加速（{actual_precision.upper()} · 默认轻量引擎；"
                    "若本机具备 CUDA 12 并安装 onnxruntime-gpu 将自动优先启用 CUDAExecutionProvider）"
                )
        else:
            if host_cuda.cuda12_detected:
                text = (
                    "未检测到 GPU 加速，将使用 CPU 推理（FP32 · 自动启用 Proxy 1600 提速；"
                    "检测到本机具备 CUDA 12，可安装 onnxruntime-gpu 启用 CUDAExecutionProvider）"
                )
            else:
                text = "未检测到 GPU 加速，将使用 CPU 推理（FP32 · 自动启用 Proxy 1600 提速）"

        status = AiRuntimeStatus(
            available=True,
            active_provider=kind,
            ort_provider=ort_name,
            precision=actual_precision,
            model_path=resolved,
            available_providers=available_providers,
            host_cuda12_detected=host_cuda.cuda12_detected,
            host_cuda_detail=host_cuda.detail,
            status_text=text,
            error="; ".join(errors),
        )
        with _PROBE_LOCK:
            _PROBE_CACHE[cache_key] = status
        return status

    missing_model = resolve_model_path("fp32", model_path) is None and resolve_model_path("fp16", model_path) is None
    if missing_model:
        text = "未找到 StackMFF-V3 ONNX 模型权重（focus_stack_app/fusion/weights/）"
    else:
        text = f"AI 推理引擎初始化失败：{'; '.join(errors) or '无可用执行器'}"
    status = AiRuntimeStatus(
        available=False,
        active_provider="NONE",
        ort_provider="",
        precision="",
        model_path=None,
        available_providers=available_providers,
        host_cuda12_detected=host_cuda.cuda12_detected,
        host_cuda_detail=host_cuda.detail,
        status_text=text,
        error="; ".join(errors) or "no usable ONNX Runtime execution provider",
    )
    with _PROBE_LOCK:
        _PROBE_CACHE[cache_key] = status
    return status


def clear_probe_cache() -> None:
    with _PROBE_LOCK:
        _PROBE_CACHE.clear()


def pad_to_64(array: np.ndarray) -> np.ndarray:
    """Edge-pad bottom/right of [..., H, W] to multiples of 64 pixels."""
    height, width = array.shape[-2:]
    pad_h = (-height) % 64
    pad_w = (-width) % 64
    if pad_h == 0 and pad_w == 0:
        return np.ascontiguousarray(array, dtype=np.float32)
    pad_spec = [(0, 0)] * (array.ndim - 2) + [(0, pad_h), (0, pad_w)]
    return np.ascontiguousarray(np.pad(array, pad_spec, mode="edge"), dtype=np.float32)


def softmax_layers(logits: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Numerically stable softmax over axis 0 of [N, H, W] logits.

    When ``valid_mask`` (bool ``[N, H, W]``) is supplied, frames that fall
    outside their source footprint are masked to ``-inf`` before exponentiation
    on every pixel covered by at least one valid frame, preventing reflected
    warp padding from stealing probability mass or underflowing valid frames.
    """
    shifted = np.array(logits, dtype=np.float32, copy=True)
    if valid_mask is None:
        max_val = np.max(shifted, axis=0, keepdims=True)
        np.subtract(shifted, max_val, out=shifted)
        np.exp(shifted, out=shifted)
        denom = np.sum(shifted, axis=0, keepdims=True)
        np.maximum(denom, 1e-12, out=denom)
        np.divide(shifted, denom, out=shifted)
        return shifted

    valid = np.asarray(valid_mask, dtype=bool)
    any_valid = np.any(valid, axis=0)
    all_covered = bool(any_valid.all())
    partial_indices: list[int] = []
    for idx in range(valid.shape[0]):
        layer = valid[idx]
        if layer.all():
            continue
        partial_indices.append(idx)
        inv = ~layer if all_covered else ((~layer) & any_valid)
        shifted[idx][inv] = -np.inf
    max_val = np.max(shifted, axis=0, keepdims=True)
    np.subtract(shifted, max_val, out=shifted)
    np.exp(shifted, out=shifted)
    for idx in partial_indices:
        layer = valid[idx]
        inv = ~layer if all_covered else ((~layer) & any_valid)
        shifted[idx][inv] = 0.0
    denom = np.sum(shifted, axis=0, keepdims=True)
    np.maximum(denom, 1e-12, out=denom)
    np.divide(shifted, denom, out=shifted)
    return shifted


def enforce_valid_probabilities(
    probs: np.ndarray,
    valid_mask: np.ndarray | None,
    *,
    in_place: bool = False,
) -> np.ndarray:
    """Zero invalid-footprint frame weights and re-normalize over valid frames."""
    if valid_mask is None:
        return probs
    valid = np.asarray(valid_mask, dtype=bool)
    any_valid = np.any(valid, axis=0)
    all_covered = bool(any_valid.all())
    out = probs if (in_place and probs.dtype == np.float32) else np.array(probs, dtype=np.float32, copy=True)
    changed = False
    for idx in range(valid.shape[0]):
        layer = valid[idx]
        if layer.all():
            continue
        changed = True
        inv = ~layer if all_covered else ((~layer) & any_valid)
        out[idx][inv] = 0.0
    if not changed:
        return out
    denom = np.sum(out, axis=0)
    if all_covered and float(denom.min()) > 1e-6:
        np.divide(out, denom[None, :, :], out=out)
        return out
    safe_denom = np.where(any_valid & (denom > 1e-6), denom, 1.0)
    np.divide(out, safe_denom[None, :, :], out=out)
    zero_covered = any_valid & (denom <= 1e-6)
    if np.any(zero_covered):
        counts = np.sum(valid[:, zero_covered], axis=0, dtype=np.float32)
        out[:, zero_covered] = valid[:, zero_covered].astype(np.float32) / np.maximum(counts[None, :], 1.0)
    return out


def _resample_valid_mask(valid_mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Conservatively resample a [N, H, W] boolean valid mask to (target_h, target_w)."""
    layers, src_h, src_w = valid_mask.shape
    if src_h == target_h and src_w == target_w:
        return valid_mask
    resized = np.empty((layers, target_h, target_w), dtype=bool)
    shrinking = target_h < src_h or target_w < src_w
    for idx in range(layers):
        layer = valid_mask[idx]
        if layer.all():
            resized[idx] = True
        elif not layer.any():
            resized[idx] = False
        elif shrinking:
            invalid = (~layer).astype(np.float32)
            resized[idx] = cv2.resize(invalid, (target_w, target_h), interpolation=cv2.INTER_AREA) == 0.0
        else:
            resized[idx] = (
                cv2.resize(layer.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST) != 0
            )
    return resized


def smooth_probabilities(probs: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth each [H, W] probability map and re-normalize over N."""
    if sigma <= 0.0:
        return probs
    radius = max(1, int(3.0 * sigma + 0.5))
    ksize = 2 * radius + 1
    smoothed = np.empty_like(probs, dtype=np.float32)
    for index in range(probs.shape[0]):
        smoothed[index] = cv2.GaussianBlur(
            probs[index], (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
        )
    denom = np.sum(smoothed, axis=0, keepdims=True)
    smoothed /= np.maximum(denom, 1e-12)
    return smoothed


class StackMFFV3Runner:
    """Per-job StackMFF-V3 ONNX Runtime runner with bounded session caching."""

    def __init__(
        self,
        status: AiRuntimeStatus,
        *,
        max_cached_shapes: int = 6,
        cpu_threads: int = 0,
    ) -> None:
        if not status.available or status.model_path is None:
            raise RuntimeError(status.status_text or status.error or "AI runtime is unavailable")
        self.status = status
        self.max_cached_shapes = max(1, int(max_cached_shapes))
        self.cpu_threads = max(0, int(cpu_threads))
        self._ort = _import_ort()
        if self._ort is None:
            raise RuntimeError("onnxruntime is not installed")
        self._sessions: OrderedDict[tuple[int, int, int], Any] = OrderedDict()

    def close(self) -> None:
        self._sessions.clear()

    def __enter__(self) -> "StackMFFV3Runner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _get_session(self, layers: int, height: int, width: int) -> Any:
        key = (layers, height, width) if self.status.active_provider == "DIRECTML" else (0, 0, 0)
        existing = self._sessions.get(key)
        if existing is not None:
            self._sessions.move_to_end(key)
            return existing
        while len(self._sessions) >= self.max_cached_shapes:
            self._sessions.popitem(last=False)
        session = _create_ort_session(
            self._ort,
            self.status.model_path,
            self.status.active_provider,
            shape=(layers, height, width) if self.status.active_provider == "DIRECTML" else None,
            threads=self.cpu_threads,
        )
        self._sessions[key] = session
        return session

    def forward_logits(self, gray_stack: np.ndarray) -> np.ndarray:
        """Run one forward pass on [N, H, W] float32 luma in [0, 1] -> [N, H, W] logits."""
        if gray_stack.ndim != 3 or gray_stack.shape[0] < 2:
            raise ValueError(f"Expected [N>=2, H, W] grayscale stack, got {gray_stack.shape}")
        layers, height, width = gray_stack.shape
        padded = pad_to_64(gray_stack[None, ...])
        _, _, pad_h, pad_w = padded.shape

        def _execute() -> np.ndarray:
            session = self._get_session(layers, pad_h, pad_w)
            return session.run(["logits"], {"input": padded})[0]

        if self.status.is_gpu:
            with _GPU_INFERENCE_LOCK:
                raw = _execute()
        else:
            raw = _execute()
        return np.asarray(raw[0, :, :height, :width], dtype=np.float32)

    def predict_probabilities(
        self,
        gray_stack: np.ndarray,
        *,
        scales: Sequence[float] = (1.0, 0.75),
        sigma: float = 0.5,
        valid_mask: np.ndarray | None = None,
        cancel_event: threading.Event | None = None,
    ) -> np.ndarray:
        """Predict multi-scale smoothed probabilities [N, H, W] summing to 1 over N."""
        layers, height, width = gray_stack.shape
        if valid_mask is not None and valid_mask.all():
            valid_mask = None
        accumulated: np.ndarray | None = None
        for scale in scales:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("AI fusion cancelled")
            s = float(scale)
            if abs(s - 1.0) < 1e-6:
                logits = self.forward_logits(gray_stack)
                probs = softmax_layers(logits, valid_mask=valid_mask)
            else:
                sh = max(64, int(round(height * s)))
                sw = max(64, int(round(width * s)))
                interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
                scaled = np.empty((layers, sh, sw), dtype=np.float32)
                for idx in range(layers):
                    scaled[idx] = cv2.resize(gray_stack[idx], (sw, sh), interpolation=interp)
                logits_s = self.forward_logits(scaled)
                valid_s = _resample_valid_mask(valid_mask, sh, sw) if valid_mask is not None else None
                probs_s = softmax_layers(logits_s, valid_mask=valid_s)
                probs = np.empty((layers, height, width), dtype=np.float32)
                for idx in range(layers):
                    probs[idx] = cv2.resize(probs_s[idx], (width, height), interpolation=cv2.INTER_LINEAR)
            if accumulated is None:
                accumulated = probs
            else:
                accumulated += probs
        assert accumulated is not None
        if len(scales) > 1:
            accumulated *= (1.0 / float(len(scales)))
        probs_avg = accumulated
        if sigma > 0.0:
            probs_avg = smooth_probabilities(probs_avg, sigma)
        if valid_mask is not None:
            probs_avg = enforce_valid_probabilities(probs_avg, valid_mask, in_place=True)
        return probs_avg


__all__ = [
    "AiRuntimeStatus",
    "FP16_MODEL_FILENAME",
    "FP32_MODEL_FILENAME",
    "HostCudaInfo",
    "StackMFFV3Runner",
    "clear_probe_cache",
    "detect_host_cuda12",
    "enforce_valid_probabilities",
    "pad_to_64",
    "probe_ai_runtime",
    "resolve_model_path",
    "smooth_probabilities",
    "softmax_layers",
    "weights_directory",
]
