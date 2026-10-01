from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from focus_stack_app.fusion import focus_masks
from focus_stack_app.fusion import quality_fusion, surface_tone
from focus_stack_app.hugin.enfuse import EnfuseConfig, EnfuseError, Enfuser
from focus_stack_app.hugin.focus_repair import HuginFocusRepair
from focus_stack_app.hugin.process import CommandResult


class _Runner:
    def __init__(self):
        self.commands = []

    def run(self, command, **kwargs):
        self.commands.append(tuple(map(str, command)))
        Image.new("RGB", (80, 64), (120, 120, 120)).save(command[command.index("-o") + 1])
        return CommandResult(tuple(command), 0)


def _frames(tmp_path):
    paths = []
    for index in range(2):
        pixels = np.full((64, 80, 3), 120 + index * 12, np.uint8)
        pixels[16:48, 20 + index * 2:60 + index * 2] = (200, 40, 40)
        path = tmp_path / f"frame{index}.tif"
        Image.fromarray(pixels).save(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize(
    "mode,expected_protect,expects_observer",
    [("legacy", True, False), ("quality", False, True), ("gate", False, True)],
)
def test_enfuser_focus_mode_contract(tmp_path, monkeypatch, mode, expected_protect, expects_observer):
    paths = _frames(tmp_path)
    captured = {}
    texture_applies = []
    original = focus_masks.build_focus_labels
    original_texture_apply = quality_fusion.FlatTextureStatistics.apply

    def spy(count, loader, **kwargs):
        captured.update(kwargs)
        return original(count, loader, **kwargs)

    def spy_texture_apply(self, labels):
        texture_applies.append(self)
        return original_texture_apply(self, labels)

    monkeypatch.setattr(focus_masks, "build_focus_labels", spy)
    monkeypatch.setattr(quality_fusion.FlatTextureStatistics, "apply", spy_texture_apply)
    result = Enfuser(
        "enfuse",
        config=EnfuseConfig(focus_blend_levels=1),
        runner=_Runner(),
    ).fuse(
        paths,
        tmp_path / f"{mode}.tif",
        work_dir=tmp_path / f"{mode}_work",
        cleanup_on_success=False,
        focus_mask_mode=mode,
        focus_reference_index=1,
        focus_gate_edge_mode="localized",
    )
    assert result.ok
    assert captured["protect_chromatic_edges"] is expected_protect
    assert captured["focus_support_radius"] == 7
    assert (captured.get("frame_observer") is not None) is expects_observer
    assert len(texture_applies) == (1 if mode in {"quality", "gate"} else 0)


def test_enfuser_focus_mode_validates_mode_and_reference(tmp_path):
    paths = _frames(tmp_path)
    enfuser = Enfuser("enfuse", runner=_Runner())
    with pytest.raises(ValueError, match="focus_mask_mode"):
        enfuser.fuse(paths, tmp_path / "bad_mode.tif", focus_mask_mode="unknown")
    with pytest.raises(ValueError, match="focus reference"):
        enfuser.fuse(paths, tmp_path / "bad_reference.tif", focus_reference_index=2)


def _drifting_red_frames(tmp_path):
    paths = []
    frames = []
    for index, paint in enumerate(((160, 45, 45), (170, 55, 55))):
        pixels = np.full((128, 160, 3), 218, np.uint8)
        pixels[24:104, 38:122] = paint
        # Fine structure stays fixed while only the broad red material drifts.
        pixels[38:90:8, 50:110] = (np.array(paint, np.float32) * 0.86).astype(np.uint8)
        path = tmp_path / f"tone-frame-{index}.tif"
        Image.fromarray(pixels).save(path)
        paths.append(path)
        frames.append(pixels)
    return paths, frames


@pytest.mark.parametrize("configured_levels,expected_levels", [(None, 1), (5, 5)])
def test_quality_mode_repairs_sources_before_real_enfuse_command(
    tmp_path, monkeypatch, configured_levels, expected_levels,
):
    paths, source_frames = _drifting_red_frames(tmp_path)
    runner = _Runner()
    steps = []

    original_stabilize = quality_fusion.stabilize_neutral_labels
    original_boundary_apply = quality_fusion.SurfaceBoundaryOwnership.apply
    original_print_apply = quality_fusion.PrintedEdgeOwnership.apply
    original_texture_apply = quality_fusion.FlatTextureStatistics.apply
    original_corrected_inputs = HuginFocusRepair.corrected_inputs

    def stabilize(labels, reference):
        steps.append("regularizer")
        return original_stabilize(labels, reference)

    def boundary_apply(self, labels):
        steps.append("boundary")
        return original_boundary_apply(self, labels)

    def print_apply(self, labels, *args, **kwargs):
        steps.append("printed")
        return original_print_apply(self, labels, *args, **kwargs)

    def texture_apply(self, labels):
        steps.append("texture")
        return original_texture_apply(self, labels)

    def corrected_inputs(self, *args, **kwargs):
        steps.append("tone_inputs")
        return original_corrected_inputs(self, *args, **kwargs)

    monkeypatch.setattr(quality_fusion, "stabilize_neutral_labels", stabilize)
    monkeypatch.setattr(quality_fusion.FlatTextureStatistics, "apply", texture_apply)
    monkeypatch.setattr(quality_fusion.SurfaceBoundaryOwnership, "apply", boundary_apply)
    monkeypatch.setattr(quality_fusion.PrintedEdgeOwnership, "apply", print_apply)
    monkeypatch.setattr(HuginFocusRepair, "corrected_inputs", corrected_inputs)

    result = Enfuser(
        "enfuse",
        config=EnfuseConfig(focus_blend_levels=configured_levels),
        runner=runner,
    ).fuse(
        paths,
        tmp_path / "tone-output.tif",
        work_dir=tmp_path / "tone-work",
        cleanup_on_success=False,
        focus_mask_mode="quality",
        focus_reference_index=0,
    )

    assert result.ok
    assert steps == ["regularizer", "texture", "boundary", "printed", "tone_inputs"]
    command = runner.commands[-1]
    assert f"--levels={expected_levels}" in command
    inputs = [Path(item) for item in command[command.index("-o") + 2:]]
    assert len(inputs) == 2
    assert all(path.parent.name == "tone_inputs" and path.is_file() for path in inputs)
    corrected = np.asarray(Image.open(inputs[1]))
    reference_pixel = source_frames[0][64, 80].astype(np.int16)
    corrected_pixel = corrected[64, 80].astype(np.int16)
    source_pixel = source_frames[1][64, 80].astype(np.int16)
    assert np.linalg.norm(corrected_pixel - reference_pixel) < np.linalg.norm(source_pixel - reference_pixel)
    for path, original in zip(paths, source_frames):
        np.testing.assert_array_equal(np.asarray(Image.open(path)), original)


def test_quality_repair_stabilizes_blocky_flat_background_to_reference():
    import cv2

    height, width = 128, 160
    y, x = np.indices((height, width))
    luma = (220 + (x // 32) % 3).astype(np.uint8)
    rgb = np.repeat(luma[:, :, None], 3, axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    score = np.zeros((height, width), np.float32)
    repair = HuginFocusRepair(
        0, edge_ownership=False, surface_tone=False,
        stabilize_texture=True, edge_mode="localized",
    )
    repair.observe(0, rgb, gray, score)
    repair.observe(1, rgb, gray, score)

    labels = (np.indices((height, width)).sum(axis=0) % 2).astype(np.uint16)
    repaired = repair.apply(labels)

    assert np.count_nonzero(repaired == 1) < np.count_nonzero(labels == 1)
    assert repair.texture is not None


def test_cancellation_during_material_correction_keeps_aligned_inputs_unchanged(
    tmp_path, monkeypatch,
):
    import threading

    paths, source_frames = _drifting_red_frames(tmp_path)
    runner = _Runner()
    cancel_event = threading.Event()
    original_correct = surface_tone.SurfaceToneHarmonizer.correct

    def cancel_after_first_correction(self, rgb, index=None, **kwargs):
        corrected = original_correct(self, rgb, index, **kwargs)
        cancel_event.set()
        return corrected

    monkeypatch.setattr(surface_tone.SurfaceToneHarmonizer, "correct", cancel_after_first_correction)
    with pytest.raises(EnfuseError, match="cancelled while correcting material tone"):
        Enfuser("enfuse", runner=runner).fuse(
            paths,
            tmp_path / "cancelled-output.tif",
            work_dir=tmp_path / "cancelled-work",
            cleanup_on_success=False,
            cancel_event=cancel_event,
            focus_mask_mode="quality",
            focus_reference_index=0,
        )

    assert runner.commands == []
    assert not (tmp_path / "cancelled-output.tif").exists()
    for path, original in zip(paths, source_frames):
        np.testing.assert_array_equal(np.asarray(Image.open(path)), original)
