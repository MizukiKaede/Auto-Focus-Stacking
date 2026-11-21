"""Unit and integration tests for the StackMFF-V3 AI fusion backend."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import types

import cv2
import numpy as np
from PIL import Image
import pytest

from focus_stack_app.config import AppConfig, OutputConfig, RuntimeConfig
from focus_stack_app.fusion import AiFusionBackend
from focus_stack_app.fusion.ai_backend import make_tile_weight, tile_starts
from focus_stack_app.fusion.ai_runtime import (
    AiRuntimeStatus,
    HostCudaInfo,
    clear_probe_cache,
    probe_ai_runtime,
)
from focus_stack_app.fusion_modes import AI, normalize_fusion_backend
from focus_stack_app.hugin.output_encoder import OutputCollisionError
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.ui.group_list import _backend_label


def _make_sharp_split_stack(tmp_path: Path, size: tuple[int, int] = (320, 240)) -> tuple[list[Path], np.ndarray]:
    """Create two images where left half is sharp in img0 and right half in img1."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // 4) + (yy // 4)) % 2).astype(np.float32)
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:, :, 0] = (40 + 180 * checker).astype(np.uint8)
    base[:, :, 1] = (80 + 120 * checker).astype(np.uint8)
    base[:, :, 2] = (180 - 100 * checker).astype(np.uint8)
    blurred = cv2.GaussianBlur(base, (15, 15), 4.0)

    img0 = blurred.copy()
    img0[:, : w // 2] = base[:, : w // 2]

    img1 = blurred.copy()
    img1[:, w // 2 :] = base[:, w // 2 :]

    p0 = tmp_path / "IMG_0001.jpg"
    p1 = tmp_path / "IMG_0002.jpg"
    Image.fromarray(img0).save(p0, quality=98)
    Image.fromarray(img1).save(p1, quality=98)
    return [p0, p1], base


class _SharpnessStubRunner:
    """Deterministic stub runner that assigns higher probability to sharper local texture."""

    def __init__(self, status: AiRuntimeStatus) -> None:
        self.status = status
        self.calls = 0
        self.closed = False

    def predict_probabilities(self, gray_stack, *, scales=(1.0, 0.75), sigma=0.5, cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("AI fusion cancelled")
        self.calls += 1
        scores = []
        for idx in range(gray_stack.shape[0]):
            lap = np.abs(cv2.Laplacian(gray_stack[idx], cv2.CV_32F, ksize=3))
            scores.append(cv2.GaussianBlur(lap, (9, 9), 2.0) + 1e-4)
        stacked = np.stack(scores, axis=0)
        # Sharpen probabilities via temperature softmax
        logits = stacked * 40.0
        logits -= logits.max(axis=0, keepdims=True)
        exp = np.exp(logits)
        return (exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-12)).astype(np.float32)

    def close(self) -> None:
        self.closed = True


def _analysis_for(paths: list[Path], shape: tuple[int, int]) -> dict:
    eye = np.eye(3, dtype=np.float32).tolist()
    return {
        "selected_paths": [str(p) for p in paths],
        "selected_indices": list(range(len(paths))),
        "preview_transforms": [eye for _ in paths],
        "analysis_shapes": [list(shape) for _ in paths],
        "preview_reference_index": 0,
        "preview_reference": str(paths[0]),
        "reference_analysis_shape": list(shape),
    }


def test_fusion_modes_and_runtime_config_ai_validation(tmp_path: Path):
    assert normalize_fusion_backend(None) == AI
    assert normalize_fusion_backend("ai") == AI
    assert normalize_fusion_backend("fast") == AI
    assert normalize_fusion_backend("stackmff") == AI
    assert normalize_fusion_backend("StackMFF_V3") == AI
    assert normalize_fusion_backend("ai_fusion") == AI
    assert RuntimeConfig().fusion_backend == "ai"
    with pytest.raises(ValueError):
        normalize_fusion_backend("unknown_mode")

    cfg = AppConfig(
        runtime=RuntimeConfig(
            fusion_backend="ai",
            ai_device="auto",
            ai_scales="1.0,0.5,0.25",
            ai_sigma=0.5,
            ai_tile_size=512,
            ai_cpu_proxy_long_edge=1600,
        )
    )
    path = tmp_path / "cfg.json"
    cfg.save(path)
    loaded = AppConfig.load(path)
    assert loaded.runtime.fusion_backend == "ai"
    assert loaded.runtime.ai_tile_size == 512
    assert loaded.runtime.ai_scales == "1,0.5,0.25"
    assert loaded.runtime.ai_surface_tone is False

    with pytest.raises(ValueError):
        RuntimeConfig(ai_device="vulkan")
    with pytest.raises(ValueError):
        RuntimeConfig(ai_tile_size=300)  # not multiple of 64
    with pytest.raises(ValueError):
        RuntimeConfig(ai_scales="0,-1")


def test_probe_priority_cuda_over_directml_and_host_cuda_note(monkeypatch, tmp_path: Path):
    clear_probe_cache()
    fp16_file = tmp_path / "stackmff_v3_final_fp16.onnx"
    fp32_file = tmp_path / "stackmff_v3_final.onnx"
    fp16_file.write_bytes(b"fp16")
    fp32_file.write_bytes(b"fp32")

    monkeypatch.setattr(
        "focus_stack_app.fusion.ai_runtime.resolve_model_path",
        lambda precision, explicit=None: fp16_file if precision == "fp16" else fp32_file,
    )

    class _FakeOrt:
        def __init__(self, providers):
            self._providers = providers

        def get_available_providers(self):
            return list(self._providers)

    # Case 1: Both CUDA and DirectML exist -> CUDAExecutionProvider wins
    monkeypatch.setattr(
        "focus_stack_app.fusion.ai_runtime._import_ort",
        lambda: _FakeOrt(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]),
    )
    monkeypatch.setattr(
        "focus_stack_app.fusion.ai_runtime.detect_host_cuda12",
        lambda: HostCudaInfo(True, True, r"C:\CUDA\v12.6"),
    )
    monkeypatch.setattr("focus_stack_app.fusion.ai_runtime._warmup_probe", lambda ort, p, kind: None)

    st_cuda = probe_ai_runtime(refresh=True)
    assert st_cuda.available is True
    assert st_cuda.active_provider == "CUDA"
    assert st_cuda.ort_provider == "CUDAExecutionProvider"
    assert st_cuda.precision == "fp16"
    assert "CUDAExecutionProvider" in st_cuda.status_text

    # Case 2: Only DirectML + CPU exist, host has CUDA 12 -> DirectML active + CUDA 12 upgrade hint
    monkeypatch.setattr(
        "focus_stack_app.fusion.ai_runtime._import_ort",
        lambda: _FakeOrt(["DmlExecutionProvider", "CPUExecutionProvider"]),
    )
    st_dml = probe_ai_runtime(refresh=True)
    assert st_dml.available is True
    assert st_dml.active_provider == "DIRECTML"
    assert st_dml.precision == "fp16"
    assert st_dml.host_cuda12_detected is True
    assert "onnxruntime-gpu" in st_dml.status_text
    assert "CUDAExecutionProvider" in st_dml.status_text

    # Case 3: DirectML warmup fails -> explicitly falls back to CPU with FP32 (never FP16!)
    def _fail_dml(ort, model_path, kind):
        if kind == "DIRECTML":
            raise RuntimeError("DML init error")

    monkeypatch.setattr("focus_stack_app.fusion.ai_runtime._warmup_probe", _fail_dml)
    st_cpu = probe_ai_runtime(refresh=True)
    assert st_cpu.available is True
    assert st_cpu.active_provider == "CPU"
    assert st_cpu.precision == "fp32"
    assert st_cpu.model_path == fp32_file

    # Case 4: onnxruntime not installed
    monkeypatch.setattr("focus_stack_app.fusion.ai_runtime._import_ort", lambda: None)
    st_none = probe_ai_runtime(refresh=True)
    assert st_none.available is False
    assert st_none.active_provider == "NONE"
    assert "onnxruntime-directml" in st_none.status_text
    clear_probe_cache()


def test_ai_backend_cached_and_two_pass_parity_and_tone_toggle(tmp_path: Path, monkeypatch):
    paths, target = _make_sharp_split_stack(tmp_path, size=(320, 240))
    analysis = _analysis_for(paths, (240, 320))
    fake_status = AiRuntimeStatus(
        available=True,
        active_provider="DIRECTML",
        ort_provider="DmlExecutionProvider",
        precision="fp16",
        model_path=tmp_path / "dummy.onnx",
        host_cuda12_detected=True,
        status_text="DirectML",
    )

    runners: list[_SharpnessStubRunner] = []

    def _factory(status):
        r = _SharpnessStubRunner(status)
        runners.append(r)
        return r

    # 1) Default Cached tiled path (ai_surface_tone=False, scales=(1,0.5,0.25))
    import focus_stack_app.fusion.surface_tone as surface_tone_mod

    orig_harmonizer = surface_tone_mod.SurfaceToneHarmonizer
    monkeypatch.setattr(
        surface_tone_mod,
        "SurfaceToneHarmonizer",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("SurfaceToneHarmonizer should be skipped by default")),
    )
    out_cached = tmp_path / "out_cached.jpg"
    backend_cached = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=192),
        runner_factory=_factory,
        runtime_prober=lambda *a, **kw: fake_status,
    )
    res_cached = backend_cached.fuse(
        {"group_id": 1},
        analysis,
        out_cached,
        tmp_path / "work1",
        OutputConfig(format="jpg", jpeg_quality=100),
        threading.Event(),
    )
    assert res_cached.actual_backend == "ai"
    assert "AI_NO_TONE_REPAIR" in res_cached.diagnostics
    assert "AI_SCALES:1,0.5,0.25" in res_cached.diagnostics
    assert "AI_PROVIDER:DIRECTML" in res_cached.diagnostics
    assert "AI_HOST_CUDA12_AVAILABLE_FOR_ORT_GPU" in res_cached.diagnostics
    assert runners[-1].closed is True
    assert runners[-1].calls >= 4

    # 2) Low-memory 2-pass Top-K path
    out_2pass = tmp_path / "out_2pass.jpg"
    backend_2pass = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=192, low_memory_mode=True),
        runner_factory=_factory,
        runtime_prober=lambda *a, **kw: fake_status,
    )
    res_2pass = backend_2pass.fuse(
        {"group_id": 2},
        analysis,
        out_2pass,
        tmp_path / "work2",
        OutputConfig(format="jpg", jpeg_quality=100),
        threading.Event(),
    )
    assert "AI_LOW_MEMORY_TOPK_2PASS" in res_2pass.diagnostics
    assert "AI_NO_TONE_REPAIR" in res_2pass.diagnostics

    im_cached = np.array(Image.open(out_cached))
    im_2pass = np.array(Image.open(out_2pass))
    assert im_cached.shape == target.shape
    mse_pair = float(np.mean((im_cached.astype(np.float32) - im_2pass.astype(np.float32)) ** 2))
    psnr_pair = 99.0 if mse_pair < 1e-9 else 10.0 * np.log10((255.0**2) / mse_pair)
    assert psnr_pair > 40.0

    # 3) Opt-in ai_surface_tone=True enables SurfaceToneHarmonizer v17
    monkeypatch.setattr(surface_tone_mod, "SurfaceToneHarmonizer", orig_harmonizer)
    out_toned = tmp_path / "out_toned.jpg"
    backend_toned = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=192, ai_surface_tone=True),
        runner_factory=_factory,
        runtime_prober=lambda *a, **kw: fake_status,
    )
    res_toned = backend_toned.fuse(
        {"group_id": 3},
        analysis,
        out_toned,
        tmp_path / "work_toned",
        OutputConfig(format="jpg", jpeg_quality=100),
        threading.Event(),
    )
    assert "MATERIAL_PAIRED_SURFACE_TONE_V17" in res_toned.diagnostics

    # Refusing overwrite when overwrite=False
    with pytest.raises(OutputCollisionError):
        backend_cached.fuse(
            {"group_id": 4},
            analysis,
            out_cached,
            tmp_path / "work3",
            OutputConfig(format="jpg", overwrite=False),
            threading.Event(),
        )


def test_ai_backend_cpu_proxy_and_cancellation(tmp_path: Path):
    paths, _ = _make_sharp_split_stack(tmp_path, size=(320, 240))
    analysis = _analysis_for(paths, (240, 320))
    cpu_status = AiRuntimeStatus(
        available=True,
        active_provider="CPU",
        ort_provider="CPUExecutionProvider",
        precision="fp32",
        model_path=tmp_path / "dummy_fp32.onnx",
        status_text="CPU",
    )
    backend = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_cpu_proxy_long_edge=160),
        runner_factory=lambda st: _SharpnessStubRunner(st),
        runtime_prober=lambda *a, **kw: cpu_status,
    )
    out = tmp_path / "out_cpu.jpg"
    res = backend.fuse(
        {"group_id": 1},
        analysis,
        out,
        tmp_path / "work_cpu",
        OutputConfig(format="jpg"),
        threading.Event(),
    )
    assert "AI_CPU_PROXY_FALLBACK:160" in res.diagnostics
    assert out.is_file()

    # Cancellation before start raises immediately
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        backend.fuse(
            {"group_id": 2},
            analysis,
            tmp_path / "cancelled.jpg",
            tmp_path / "work_cancel",
            OutputConfig(format="jpg"),
            cancel,
        )


def test_merge_service_constructs_ai_backend(tmp_path: Path):
    service = StackMergeService(
        tmp_path / "out",
        archive_enabled=False,
        fusion_backend="ai",
        runtime_config=RuntimeConfig(fusion_backend="ai"),
    )
    assert service.requested_backend == "ai"
    assert isinstance(service.backend, AiFusionBackend)


def test_tile_helpers_cover_full_extent():
    starts = tile_starts(1000, 512, 64)
    assert starts[0] == 0
    assert starts[-1] + 512 == 1000
    w = make_tile_weight(512, 512, 64, 0, 512, 0, 512, 1000, 1000)
    assert w[0, 0] == pytest.approx(1.0)
    assert w[-1, -1] < 0.05


def test_ui_backend_combo_and_status_note(monkeypatch, tmp_path: Path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from focus_stack_app.ui.main_window import MainWindow

    assert _backend_label("ai") == "AI (StackMFF-V3)"

    fake_status = AiRuntimeStatus(
        available=True,
        active_provider="DIRECTML",
        ort_provider="DmlExecutionProvider",
        precision="fp16",
        model_path=tmp_path / "model.onnx",
        host_cuda12_detected=True,
        status_text="已启用 DirectML GPU 加速（FP16；检测到 CUDA 12，可安装 onnxruntime-gpu 启用 CUDAExecutionProvider）",
    )
    monkeypatch.setattr("focus_stack_app.fusion.ai_runtime.probe_ai_runtime", lambda *a, **kw: fake_status)

    app = QApplication.instance() or QApplication([])
    received = []
    window = MainWindow(controller_factory=lambda **kw: received.append(kw) or types.SimpleNamespace(running=False, start=lambda: None, cancel=lambda: None, close=lambda: None, wait=lambda t: None))
    window.show()
    try:
        idx = window.backend_combo.findData("ai")
        assert idx == 0
        assert window.backend_combo.currentData() == "ai"
        assert window.backend_combo.findData("fast") == -1
        assert not window.ai_status_label.isVisible()
        window.advanced_toggle.setChecked(True)
        assert window.ai_status_label.isVisible()
        assert "CUDAExecutionProvider" in window.ai_status_label.text()
        # Switching to quality hides the AI status note
        q_idx = window.backend_combo.findData("quality")
        window.backend_combo.setCurrentIndex(q_idx)
        assert not window.ai_status_label.isVisible()
        # Switching back to ai shows it again
        window.backend_combo.setCurrentIndex(idx)
        assert window.ai_status_label.isVisible()
    finally:
        window.close()
        app.processEvents()


def test_real_onnx_bundled_model_smoke(tmp_path: Path):
    """End-to-end smoke test using the real bundled ONNX model + installed onnxruntime."""
    clear_probe_cache()
    status = probe_ai_runtime(refresh=True)
    if not status.available:
        pytest.skip(f"ONNX Runtime / bundled weights unavailable: {status.status_text}")

    paths, _ = _make_sharp_split_stack(tmp_path, size=(256, 192))
    analysis = _analysis_for(paths, (192, 256))
    out = tmp_path / "real_ai_fused.jpg"
    backend = AiFusionBackend(runtime_config=RuntimeConfig(fusion_backend="ai", ai_scales="1.0"))
    result = backend.fuse(
        {"group_id": 99},
        analysis,
        out,
        tmp_path / "work_real",
        OutputConfig(format="jpg", jpeg_quality=100),
        threading.Event(),
    )
    assert out.is_file()
    assert result.actual_backend == "ai"
    fused = np.array(Image.open(out))
    assert fused.shape == (192, 256, 3)
    # Sharpness of fused image should exceed either single half-blurred input
    lap_fused = float(cv2.Laplacian(cv2.cvtColor(fused, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    lap_0 = float(cv2.Laplacian(cv2.cvtColor(np.array(Image.open(paths[0])), cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    lap_1 = float(cv2.Laplacian(cv2.cvtColor(np.array(Image.open(paths[1])), cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
    assert lap_fused > max(lap_0, lap_1) * 1.1


def test_ai_zero_cache_budget_forces_two_pass_and_disables_rgb_cache(tmp_path: Path):
    """Setting group or global cache budget to 0 must disable full RGB caching."""
    paths, _ = _make_sharp_split_stack(tmp_path, size=(256, 192))
    analysis = _analysis_for(paths, (192, 256))
    gpu_status = AiRuntimeStatus(
        available=True,
        active_provider="DIRECTML",
        ort_provider="DmlExecutionProvider",
        precision="fp16",
        model_path=tmp_path / "dummy.onnx",
        status_text="DirectML",
    )

    # 1) opencv_aligned_cache_bytes == 0 -> must enter 2-pass tiled mode
    res_zero_group = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=128, opencv_aligned_cache_bytes=0),
        runner_factory=lambda st: _SharpnessStubRunner(st),
        runtime_prober=lambda *a, **kw: gpu_status,
    ).fuse(
        {"group_id": 10},
        analysis,
        tmp_path / "out_zero_group.jpg",
        tmp_path / "w10",
        OutputConfig(format="jpg"),
        threading.Event(),
    )
    assert "AI_LOW_MEMORY_TOPK_2PASS" in res_zero_group.diagnostics

    # 2) fusion_cache_budget_bytes == 0 -> must also enter 2-pass tiled mode
    res_zero_global = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=128, fusion_cache_budget_bytes=0),
        runner_factory=lambda st: _SharpnessStubRunner(st),
        runtime_prober=lambda *a, **kw: gpu_status,
    ).fuse(
        {"group_id": 11},
        analysis,
        tmp_path / "out_zero_global.jpg",
        tmp_path / "w11",
        OutputConfig(format="jpg"),
        threading.Event(),
    )
    assert "AI_LOW_MEMORY_TOPK_2PASS" in res_zero_global.diagnostics

    # 3) 1-byte budget -> also enters 2-pass tiled mode
    res_one_byte = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=128, opencv_aligned_cache_bytes=1),
        runner_factory=lambda st: _SharpnessStubRunner(st),
        runtime_prober=lambda *a, **kw: gpu_status,
    ).fuse(
        {"group_id": 12},
        analysis,
        tmp_path / "out_one_byte.jpg",
        tmp_path / "w12",
        OutputConfig(format="jpg"),
        threading.Event(),
    )
    assert "AI_LOW_MEMORY_TOPK_2PASS" in res_one_byte.diagnostics

    # 4) Sufficient budget -> uses cached tiled mode (no 2-pass diagnostic)
    res_cached = AiFusionBackend(
        runtime_config=RuntimeConfig(fusion_backend="ai", ai_tile_size=128, opencv_aligned_cache_bytes=64 * 1024**2),
        runner_factory=lambda st: _SharpnessStubRunner(st),
        runtime_prober=lambda *a, **kw: gpu_status,
    ).fuse(
        {"group_id": 13},
        analysis,
        tmp_path / "out_cached_ok.jpg",
        tmp_path / "w13",
        OutputConfig(format="jpg"),
        threading.Event(),
    )
    assert "AI_LOW_MEMORY_TOPK_2PASS" not in res_cached.diagnostics


def _make_shifted_border_stack(tmp_path: Path, shift_px: int = 16, size: tuple[int, int] = (256, 192)) -> tuple[list[Path], dict]:
    """Reference is flat gray (128); shifted frame has high-contrast checkerboard
    that BORDER_REFLECT_101 would mirror into the left `shift_px` invalid margin.
    """
    w, h = size
    ref = np.full((h, w, 3), 128, dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    checker = (((xx // 4) + (yy // 4)) % 2).astype(np.uint8)
    shifted_src = np.where(checker[:, :, None] == 1, 240, 16).astype(np.uint8)
    shifted_src = np.broadcast_to(shifted_src, (h, w, 3)).copy()

    p0 = tmp_path / "REF_GRAY.tif"
    p1 = tmp_path / "SHIFTED_CHECKER.tif"
    Image.fromarray(ref).save(p0)
    Image.fromarray(shifted_src).save(p1)

    eye = np.eye(3, dtype=np.float32).tolist()
    shift_mat = np.array([[1.0, 0.0, float(shift_px)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32).tolist()
    analysis = {
        "selected_paths": [str(p0), str(p1)],
        "selected_indices": [0, 1],
        "preview_transforms": [eye, shift_mat],
        "analysis_shapes": [[h, w], [h, w]],
        "preview_reference_index": 0,
        "preview_reference": str(p0),
        "reference_analysis_shape": [h, w],
    }
    return [p0, p1], analysis


def test_ai_masks_reflected_warp_border_all_paths(tmp_path: Path):
    """Reflected warp padding outside source footprint must never leak into output."""
    _, analysis = _make_shifted_border_stack(tmp_path, shift_px=16, size=(256, 192))
    gpu_status = AiRuntimeStatus(
        available=True,
        active_provider="DIRECTML",
        ort_provider="DmlExecutionProvider",
        precision="fp16",
        model_path=tmp_path / "dummy.onnx",
        status_text="DirectML",
    )
    cpu_status = AiRuntimeStatus(
        available=True,
        active_provider="CPU",
        ort_provider="CPUExecutionProvider",
        precision="fp32",
        model_path=tmp_path / "dummy_fp32.onnx",
        status_text="CPU",
    )

    cases = [
        ("cached_soft", gpu_status, RuntimeConfig(fusion_backend="ai", ai_tile_size=128, ai_hard_selection=False)),
        ("cached_hard", gpu_status, RuntimeConfig(fusion_backend="ai", ai_tile_size=128, ai_hard_selection=True)),
        ("twopass_soft", gpu_status, RuntimeConfig(fusion_backend="ai", ai_tile_size=128, opencv_aligned_cache_bytes=0, ai_hard_selection=False)),
        ("twopass_hard", gpu_status, RuntimeConfig(fusion_backend="ai", ai_tile_size=128, opencv_aligned_cache_bytes=0, ai_hard_selection=True)),
        ("cpu_proxy_soft", cpu_status, RuntimeConfig(fusion_backend="ai", ai_cpu_proxy_long_edge=128, ai_hard_selection=False)),
        ("cpu_proxy_hard", cpu_status, RuntimeConfig(fusion_backend="ai", ai_cpu_proxy_long_edge=128, ai_hard_selection=True)),
    ]
    for label, status, cfg in cases:
        out = tmp_path / f"border_{label}.tif"
        res = AiFusionBackend(
            runtime_config=cfg,
            runner_factory=lambda st: _SharpnessStubRunner(st),
            runtime_prober=lambda *a, **kw: status,
        ).fuse(
            {"group_id": 20},
            analysis,
            out,
            tmp_path / f"work_{label}",
            OutputConfig(format="tif"),
            threading.Event(),
        )
        assert "AI_VALID_SOURCE_MASK:1/2" in res.diagnostics
        fused = np.array(Image.open(out))
        margin = fused[:, :16]
        assert int(np.max(np.abs(margin.astype(np.int16) - 128))) == 0, f"Border leak in {label}: min={margin.min()}, max={margin.max()}"
        # Interior where shifted frame IS valid must retain sharp checkerboard
        interior = fused[:, 32:]
        assert float(interior.std()) > 60.0, f"Valid interior lost sharpness in {label}"


def test_real_onnx_shifted_border_no_reflected_checkerboard(tmp_path: Path):
    """Real bundled ONNX model (CPU and GPU if available) must keep 16px shifted margin pure gray."""
    clear_probe_cache()
    _, analysis = _make_shifted_border_stack(tmp_path, shift_px=16, size=(256, 192))

    # Always test CPU (both tiled full-res and CPU proxy) if ORT + weights exist
    cpu_status = probe_ai_runtime(requested_device="cpu", refresh=True)
    if not cpu_status.available:
        pytest.skip(f"ONNX Runtime / bundled weights unavailable: {cpu_status.status_text}")

    run_configs = [
        ("cpu_tiled", RuntimeConfig(fusion_backend="ai", ai_device="cpu", ai_cpu_proxy_long_edge=0, ai_scales="1.0,0.5,0.25")),
        ("cpu_proxy", RuntimeConfig(fusion_backend="ai", ai_device="cpu", ai_cpu_proxy_long_edge=128, ai_scales="1.0")),
    ]
    auto_status = probe_ai_runtime(requested_device="auto", refresh=True)
    if auto_status.available and auto_status.is_gpu:
        run_configs.append(
            ("gpu_tiled", RuntimeConfig(fusion_backend="ai", ai_device="auto", ai_scales="1.0,0.5,0.25"))
        )

    for label, cfg in run_configs:
        out = tmp_path / f"real_border_{label}.tif"
        res = AiFusionBackend(runtime_config=cfg).fuse(
            {"group_id": 30},
            analysis,
            out,
            tmp_path / f"work_real_{label}",
            OutputConfig(format="tif"),
            threading.Event(),
        )
        assert "AI_VALID_SOURCE_MASK:1/2" in res.diagnostics
        fused = np.array(Image.open(out))
        margin = fused[:, :16]
        assert int(np.max(np.abs(margin.astype(np.int16) - 128))) == 0, (
            f"Real ONNX reflected border leaked in {label}: min={margin.min()}, max={margin.max()}"
        )
        interior = fused[:, 32:]
        assert float(interior.std()) > 50.0, f"Real ONNX lost valid interior texture in {label}"
