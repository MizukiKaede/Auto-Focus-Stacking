"""Small ownership, tie and routing contracts for coloured printing fringes."""

from __future__ import annotations

import gc
import threading
import weakref

import cv2
import numpy as np
import pytest

from focus_stack_app.config import RuntimeConfig
from focus_stack_app.fusion.backends import QualityFusionBackend
from focus_stack_app.fusion.quality_fusion import PrintedEdgeOwnership
from focus_stack_app.fusion.focus_masks import build_focus_labels
from focus_stack_app.hugin.output_encoder import OutputConfig
from PIL import Image


def _red_object_with_white_print(height=240, width=320):
    rgb = np.full((height, width, 3), 128, dtype=np.uint8)
    object_mask = np.zeros((height, width), dtype=np.uint8)
    object_mask[35:205, 45:275] = 1
    rgb[object_mask != 0] = (190, 45, 55)
    # An enclosed neutral mark is the printed feature.  It is deliberately
    # large enough to survive the candidate's half-size map and erosion.
    cv2.rectangle(rgb, (105, 88), (215, 145), (242, 242, 242), thickness=-1)
    return rgb, object_mask.astype(bool)


def _observe_printed_pair(first_index=0, second_index=1, *, reverse=False):
    sharp, selected_coloured = _red_object_with_white_print()
    selected_coloured[88:146, 105:216] = False
    blurred = cv2.GaussianBlur(sharp, (0, 0), 5.0)
    scores = {
        first_index: np.full(sharp.shape[:2], 0.1, dtype=np.float32),
        second_index: np.full(sharp.shape[:2], 1.0, dtype=np.float32),
    }
    candidate = PrintedEdgeOwnership(support_radius=32)
    order = [(first_index, blurred), (second_index, sharp)]
    if reverse:
        order.reverse()
    for index, image in order:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        candidate.observe(index, image, gray, scores[index])
    labels = np.zeros(sharp.shape[:2], dtype=np.uint16)
    return candidate, candidate.apply(labels, selected_coloured), selected_coloured


def test_plain_single_colour_object_keeps_original_labels():
    rgb = np.full((180, 260, 3), 128, dtype=np.uint8)
    object_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    object_mask[30:150, 40:220] = 1
    rgb[object_mask != 0] = (190, 45, 55)
    labels = np.arange(rgb.shape[0] * rgb.shape[1], dtype=np.uint16).reshape(rgb.shape[:2]) % 4

    candidate = PrintedEdgeOwnership()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    candidate.observe(0, rgb, gray, np.ones(rgb.shape[:2], dtype=np.float32))
    result = candidate.apply(labels, object_mask.astype(bool))

    assert np.array_equal(result, labels)


def test_sharp_source_owns_ink_and_red_fringe_without_retaking_background():
    candidate, result, selected_coloured = _observe_printed_pair()
    _, object_mask = _red_object_with_white_print()
    white_print = np.zeros(object_mask.shape, dtype=bool)
    white_print[88:146, 105:216] = True
    red_fringe = cv2.dilate(white_print.astype(np.uint8), np.ones((21, 21), np.uint8)).astype(bool)
    red_fringe &= object_mask & ~white_print

    # Ink and its pale halo must share the sharp source. The exterior remains
    # outside that coverage even though its colour resembles the white ink.
    assert np.count_nonzero(result[red_fringe] == 1) > 0
    assert np.all(result[white_print] == 1)
    assert np.all(result[~object_mask] == 0)
    assert np.all(result[result != 0][...] == 1)


def test_isolated_white_highlights_in_saturated_texture_do_not_trigger_retake():
    height, width = 240, 320
    base = np.full((height, width, 3), (190, 35, 45), dtype=np.uint8)
    # A saturated, textured surface with no neutral stroke or enclosed mark.
    for y in range(50, 205, 20):
        cv2.line(base, (45, y), (275, y + 7), (35, 75, 205), 3)
    sparse_highlights = base.copy()
    sparse_highlights[108, 132] = (255, 255, 255)
    sparse_highlights[171, 211] = (255, 255, 255)
    coloured = np.zeros((height, width), dtype=bool)
    coloured[35:210, 40:280] = True

    candidate = PrintedEdgeOwnership()
    for index, frame in enumerate((base, sparse_highlights)):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        candidate.observe(index, frame, gray, np.full((height, width), index, np.float32))
    labels = np.zeros((height, width), dtype=np.uint16)
    result = candidate.apply(labels, coloured)

    assert not np.any(candidate.best > 0)
    np.testing.assert_array_equal(result, labels)


def test_thin_white_box_keeps_sharp_ownership_against_defocused_copy():
    height = width = 320
    sharp = np.full((height, width, 3), 128, dtype=np.uint8)
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[24:296, 24:296] = True
    sharp[object_mask] = (190, 45, 55)

    # A four-pixel print stroke is only about two pixels wide in the compact
    # map. It is continuous, unlike isolated highlights, and should remain
    # strong evidence even when a lower-focus copy has a soft halo.
    print_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(sharp, (92, 92), (228, 228), (242, 242, 242), thickness=4)
    cv2.rectangle(print_mask, (92, 92), (228, 228), 1, thickness=4)
    coloured = object_mask & (print_mask == 0)
    defocused = cv2.GaussianBlur(sharp, (0, 0), 2.0)

    candidate = PrintedEdgeOwnership(support_radius=32)
    for index, frame, value in ((0, defocused, 0.1), (1, sharp, 1.0)):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        candidate.observe(index, frame, gray,
                          np.full((height, width), value, dtype=np.float32))

    labels = np.zeros((height, width), dtype=np.uint16)
    result = candidate.apply(labels, coloured)
    halo = cv2.dilate(print_mask, np.ones((17, 17), np.uint8)) != 0
    halo &= coloured

    assert np.any(halo)
    assert np.count_nonzero(result[halo] == 1) >= round(0.8 * np.count_nonzero(halo))
    assert np.all(result[print_mask != 0] == 1)
    assert np.array_equal(result[~object_mask], labels[~object_mask])


def test_equal_strength_uses_original_index_independent_of_observation_order():
    sharp, selected_coloured = _red_object_with_white_print()
    gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
    score = np.ones(sharp.shape[:2], dtype=np.float32)
    labels = np.full(sharp.shape[:2], 99, dtype=np.uint16)

    owners = []
    for order in ((7, 0), (0, 7)):
        candidate = PrintedEdgeOwnership()
        for index in order:
            candidate.observe(index, sharp, gray, score)
        owners.append(candidate.apply(labels, selected_coloured))

    changed = owners[0] != 99
    assert np.any(changed)
    assert np.array_equal(owners[0], owners[1])
    assert np.all(owners[0][changed] == 0)


def test_maps_are_half_size_and_do_not_retain_input_frames():
    rgb, selected_coloured = _red_object_with_white_print()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    score = np.ones(rgb.shape[:2], dtype=np.float32)
    rgb_ref = weakref.ref(rgb)
    gray_ref = weakref.ref(gray)
    score_ref = weakref.ref(score)
    candidate = PrintedEdgeOwnership()
    candidate.observe(0, rgb, gray, score)
    candidate.observe(1, rgb, gray, score)
    full_shape = rgb.shape[:2]
    full_rgb_bytes = rgb.nbytes

    assert candidate.best.shape == (full_shape[0] // 2, full_shape[1] // 2)
    assert candidate.owner.shape == candidate.best.shape
    assert candidate.inside.shape == candidate.best.shape
    assert sum(value.nbytes for value in (candidate.best, candidate.owner,
                                          candidate.inside)) < full_rgb_bytes

    del rgb, gray, score, selected_coloured
    gc.collect()
    assert rgb_ref() is None
    assert gray_ref() is None
    assert score_ref() is None


def test_build_focus_labels_guard_is_opt_in_and_order_stable():
    sharp, _ = _red_object_with_white_print()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 5.0)
    frames = (blurred, sharp)
    loader = lambda index: frames[index]

    baseline = build_focus_labels(2, loader, prefetch=False)
    explicitly_disabled = build_focus_labels(
        2, loader, prefetch=False, printed_edge_guard=False,
    )
    natural = build_focus_labels(
        2, loader, prefetch=False, printed_edge_guard=True,
    )
    reference_first = build_focus_labels(
        2, loader, frame_order=[1, 0], prefetch=False,
        printed_edge_guard=True,
    )

    np.testing.assert_array_equal(explicitly_disabled, baseline)
    np.testing.assert_array_equal(reference_first, natural)


def test_runtime_printed_edge_guard_defaults_on_with_explicit_rollback():
    assert RuntimeConfig().quality_printed_edge_guard is True
    assert RuntimeConfig(quality_printed_edge_guard=False).quality_printed_edge_guard is False


def test_neighbouring_small_print_keeps_its_own_focus_plane():
    from focus_stack_app.fusion.focus_masks import blend_focus_pyramid, focus_response

    sharp = np.full((360, 600, 3), 128, np.uint8)
    sharp[24:336, 24:576] = (190, 45, 55)
    cv2.rectangle(sharp, (120, 90), (260, 240), (242, 242, 242), 14)
    cv2.putText(sharp, "4", (315, 145), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (242, 242, 242), 4, cv2.LINE_8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 5.0)
    small_plane = blurred.copy()
    small_plane[95:170, 290:365] = sharp[95:170, 290:365]
    large_plane = sharp.copy()
    large_plane[95:170, 290:365] = blurred[95:170, 290:365]
    frames = (small_plane, large_plane)

    guard = PrintedEdgeOwnership()
    scores = []
    for index, rgb in enumerate(frames):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        score = focus_response(rgb, gray)
        scores.append(score)
        guard.observe(index, rgb, gray, score)
    raw = np.argmax(np.stack(scores), axis=0).astype(np.uint16)
    labels = guard.apply(raw)
    result = blend_focus_pyramid(2, lambda index: frames[index], labels)

    small = np.s_[105:150, 310:345]
    white = np.all(sharp[small] == (242, 242, 242), axis=2)
    assert np.count_nonzero(white) > 40
    # A nearby larger stroke must not erase a small character's sharper plane.
    assert np.mean(np.abs(result[small][white].astype(np.int16) - sharp[small][white])) < 3
    # Preserve the other feature's independently focused edge as well.
    assert np.mean(np.abs(result[83:98, 155:225].astype(np.int16)
                          - sharp[83:98, 155:225].astype(np.int16))) < 3


def test_clipped_antialiased_ink_is_owned_without_filling_white_exterior():
    from focus_stack_app.fusion.focus_masks import blend_focus_pyramid, focus_response

    sharp = np.full((240, 320, 3), 220, np.uint8)
    sharp[:, 24:240] = (190, 40, 55)
    # The ink touches the image edge; its paint-coloured antialiasing is not
    # an enclosed neutral hole, while the exterior is a larger white region.
    sharp[:46, 110:166] = (242, 190, 202)
    sharp[:42, 114:162] = (242, 242, 242)
    frames = (cv2.GaussianBlur(sharp, (0, 0), 5), sharp)
    guard = PrintedEdgeOwnership(support_radius=64)
    for index, rgb in enumerate(frames):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        guard.observe(index, rgb, gray, focus_response(rgb, gray))
    labels = guard.apply(np.zeros(sharp.shape[:2], np.uint16))
    result = blend_focus_pyramid(2, lambda index: frames[index], labels)

    ink = np.s_[:42, 114:162]
    assert np.mean(np.abs(result[ink].astype(np.int16) - sharp[ink])) < 2
    assert np.all(labels[:, 270:] == 0)


def test_external_rim_prefers_focus_over_a_larger_colour_step():
    from focus_stack_app.fusion.quality_fusion import SurfaceBoundaryOwnership

    sharp = np.full((240, 320, 3), 220, np.uint8)
    sharp[35:205, 50:270] = (195, 80, 90)
    darker = sharp.copy()
    darker[35:205, 50:270] = (180, 25, 35)
    blurred = cv2.GaussianBlur(darker, (0, 0), 2)
    guard = SurfaceBoundaryOwnership()
    # Scale both supplied focus maps down together: confidence in a real
    # coloured rim must stay separate from the units of its ranking score.
    guard.observe(0, blurred, np.full(sharp.shape[:2], 1e-10, np.float32))
    guard.observe(1, sharp, np.full(sharp.shape[:2], 1e-8, np.float32))
    labels = guard.apply(np.zeros(sharp.shape[:2], np.uint16))

    rim = labels[75:165, 49:60]
    assert np.count_nonzero(rim == 1) >= 0.9 * rim.size
    assert np.all(labels[:, :20] == 0)


@pytest.mark.parametrize("variant,expected", [
    ("legacy", False), ("gain", False), ("gate", True), ("clean", True),
])
def test_backend_routes_print_guard_only_into_gate_or_clean(tmp_path, monkeypatch, variant, expected):
    frames = [np.full((48, 64, 3), 90 + index * 10, dtype=np.uint8)
              for index in range(2)]
    paths = [tmp_path / f"frame{index}.png" for index in range(2)]
    for path, frame in zip(paths, frames):
        Image.fromarray(frame).save(path)

    import focus_stack_app.fusion.focus_masks as focus_masks

    early_guard = []
    stages = []

    def fake_build(count, load_aligned, **kwargs):
        early_guard.append(kwargs["printed_edge_guard"])
        observer = kwargs.get("frame_observer")
        if observer is not None:
            for index in range(count):
                frame = load_aligned(index)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                observer(index, frame, gray,
                         np.ones(gray.shape, dtype=np.float32))
        return np.zeros(load_aligned(0).shape[:2], dtype=np.uint16)

    def fake_blend(count, load_aligned, labels, **kwargs):
        stages.append("blend")
        return np.zeros((*labels.shape, 3), dtype=np.uint8)

    from focus_stack_app.fusion.quality_fusion import FlatTextureStatistics
    original_texture = FlatTextureStatistics.apply
    original_print = PrintedEdgeOwnership.apply

    def track_texture(self, labels):
        stages.append("texture")
        return original_texture(self, labels)

    def track_print(self, labels, *args, **kwargs):
        stages.append("print")
        return original_print(self, labels, *args, **kwargs)

    monkeypatch.setattr(focus_masks, "build_focus_labels", fake_build)
    monkeypatch.setattr(focus_masks, "blend_focus_pyramid", fake_blend)
    monkeypatch.setattr(FlatTextureStatistics, "apply", track_texture)
    monkeypatch.setattr(PrintedEdgeOwnership, "apply", track_print)
    identity = np.eye(3, dtype=np.float32).tolist()
    analysis = {
        "selected_paths": paths,
        "selected_indices": [0, 1],
        "preview_reference": paths[0],
        "preview_reference_index": 0,
        "preview_transforms": [identity, identity],
        "analysis_shapes": [[48, 64], [48, 64]],
        "reference_analysis_shape": [48, 64],
    }
    backend = QualityFusionBackend(runtime_config={
        "quality_variant": variant,
        "quality_printed_edge_guard": True,
        "quality_surface_tone": False,
    })
    result = backend.fuse(
        {}, analysis, tmp_path / f"{variant}.jpg", tmp_path / f"work_{variant}",
        OutputConfig(overwrite=True), threading.Event(),
    )

    assert result.output_path.exists()
    assert early_guard == [False]
    assert ("print" in stages) == expected
    if expected:
        assert stages.index("texture") < stages.index("print") < stages.index("blend")
