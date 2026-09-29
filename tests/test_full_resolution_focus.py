import threading
import os
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from focus_stack_app.fusion.backends import QualityFusionBackend
from focus_stack_app.fusion.focus_masks import (
    _CoherentNeutralEdges, build_focus_labels, focus_weight, blend_focus_pyramid,
)
from focus_stack_app.hugin.output_encoder import OutputConfig
from focus_stack_app.hugin.enfuse import Enfuser, EnfuseConfig


def thin_details():
    truth = np.full((600, 1800, 3), 220, np.uint8)
    for x in (850, 920):
        truth[100:500, x:x+30] = (180, 30, 30)
        truth[110:490:4, x+4:x+26] = (250, 250, 250)
    blurred = cv2.GaussianBlur(truth, (0, 0), 5)
    left, right = blurred.copy(), blurred.copy()
    left[:, :900] = truth[:, :900]
    right[:, 900:] = truth[:, 900:]
    return truth, left, right


def test_quality_preserves_thin_details_at_adjacent_focus_depths(tmp_path):
    truth, left, right = thin_details()
    paths = [tmp_path / 'near.png', tmp_path / 'far.png']
    for path, pixels in zip(paths, (left, right)):
        Image.fromarray(pixels).save(path)
    result = QualityFusionBackend().fuse(
        {}, {'selected_paths': paths, 'selected_indices': [0, 1],
             'preview_reference': paths[0], 'preview_reference_index': 0,
             'preview_transforms': [np.eye(3), np.eye(3)],
             'analysis_shapes': [[600, 1800], [600, 1800]],
             'reference_analysis_shape': [600, 1800]},
        tmp_path / 'result.tif', tmp_path / 'work', OutputConfig(format='tif'), threading.Event(),
    )
    actual = np.asarray(Image.open(result.output_path).convert('RGB')).astype(float)
    for x in (850, 920):
        roi = np.s_[120:480, x:x+30]
        assert np.abs(actual[roi] - truth[roi]).mean() < 2.0


def test_focus_masks_reject_defocused_silhouette_fringe():
    sharp = np.full((180, 240, 3), 220, np.uint8)
    sharp[20:160, 95:145] = 30
    blurred = cv2.GaussianBlur(sharp, (0, 0), 4)
    labels = build_focus_labels(2, lambda i: (sharp, blurred)[i])
    # The sharp source is flat outside its silhouette, but must own the
    # blurred frame's extended edge too, to prevent a dark/bright halo.
    assert np.mean(labels[35:145, 87:95] == 0) > .99
    assert np.mean(labels[35:145, 145:153] == 0) > .99


def test_chromatic_edge_guard_keeps_shifted_blur_outside_subject():
    rng = np.random.default_rng(8)
    background = np.clip(35 + rng.normal(0, 15, (220, 320, 1)), 0, 255).astype(np.uint8)
    background = np.repeat(background, 3, axis=2)
    near = cv2.GaussianBlur(background, (0, 0), 2.5)
    near[45:175, 85:215] = 220
    near[51:169, 91:209] = (180, 30, 30)
    blurred = np.roll(cv2.GaussianBlur(near, (0, 0), 6), 13, axis=1)
    alpha = np.zeros((220, 320), np.float32)
    alpha[45:175, 85:215] = 1
    alpha = np.roll(cv2.GaussianBlur(alpha, (0, 0), 6), 13, axis=1)
    far = np.clip(background * (1 - alpha[..., None]) + blurred * alpha[..., None], 0, 255).astype(np.uint8)
    frames = (near, far)
    regular = build_focus_labels(2, lambda i: frames[i])
    guarded = build_focus_labels(2, lambda i: frames[i], protect_chromatic_edges=True)
    fringe = np.s_[70:150, 216:237]
    assert np.mean(guarded[fringe] == 0) > .99
    assert np.mean(regular[fringe] == 0) < .9
    np.testing.assert_array_equal(guarded[70:150, 110:190], regular[70:150, 110:190])
    assert np.mean(guarded[70:150, 305:315] == 1) > .99


@pytest.mark.parametrize('background,foreground', [(35, 220), (220, 35)])
def test_default_focus_blend_does_not_add_silhouette_overshoot(background, foreground):
    sharp = np.full((240, 320, 3), background, np.uint8)
    sharp[50:190, 100:220] = foreground
    blurred = cv2.GaussianBlur(sharp, (0, 0), 12)
    frames = (sharp, blurred)
    labels = build_focus_labels(2, lambda i: frames[i])
    result = blend_focus_pyramid(2, lambda i: frames[i], labels)
    # A fused edge must not become darker/brighter than both source pixels.
    # Five-level blending previously overshot by 3--5 levels on this scene.
    assert np.all(result >= np.minimum(sharp, blurred))
    assert np.all(result <= np.maximum(sharp, blurred))
    assert np.abs(result.astype(float) - sharp).mean() < 0.3


def test_focus_weights_partition_and_do_not_depend_on_numeric_frame_order():
    _, left, right = thin_details()
    labels = build_focus_labels(2, lambda i: (left, right)[i])
    reverse = build_focus_labels(2, lambda i: (right, left)[i])
    for x in (855, 925):
        roi = np.s_[120:480, x:x+20]
        np.testing.assert_array_equal(labels[roi], 1 - reverse[roi])
    np.testing.assert_allclose(focus_weight(labels, 0) + focus_weight(labels, 1), 1, atol=1e-6)


def test_focus_mask_cancel_stops_before_decoding():
    event = threading.Event()
    event.set()
    def unexpected(index):
        pytest.fail('decoded after cancellation')
    with pytest.raises(RuntimeError, match='cancelled'):
        build_focus_labels(2, unexpected, cancel_event=event)


def test_real_enfuse_uses_padded_full_resolution_masks(tmp_path):
    executable = os.environ.get('FOCUS_STACK_ENFUSE') or shutil.which('enfuse')
    if not executable:
        pytest.skip('Set FOCUS_STACK_ENFUSE to run real Enfuse integration')
    truth, left, right = thin_details()
    paths = [tmp_path / f'frame{i}.tif' for i in range(12)]
    for i, path in enumerate(paths):
        Image.fromarray((left, right)[i % 2]).save(path)
    result = Enfuser(executable).fuse(
        paths, tmp_path / 'result.tif', work_dir=tmp_path / 'work', cleanup_on_success=False,
    )
    assert result.ok
    assert '--load-masks' in result.command_result.command
    assert '--levels=1' in result.command_result.command
    assert (tmp_path / 'work/hardmask-01.tif').is_file()
    assert (tmp_path / 'work/hardmask-12.tif').is_file()
    actual = np.asarray(Image.open(result.output_path).convert('RGB')).astype(float)
    for x in (850, 920):
        roi = np.s_[120:480, x:x+30]
        assert np.abs(actual[roi] - truth[roi]).mean() < 2.0


@pytest.mark.parametrize('config', [
    EnfuseConfig(full_resolution_focus_masks=False),
    EnfuseConfig(exposure_weight=1),
    EnfuseConfig(hard_mask=False),
    EnfuseConfig(extra_args=('--load-masks',)),
])
def test_enfuse_explicit_fusion_settings_bypass_generated_masks(tmp_path, monkeypatch, config):
    from focus_stack_app.fusion import focus_masks
    from focus_stack_app.hugin.process import CommandResult
    def unexpected(*args, **kwargs):
        pytest.fail('replaced explicitly configured fusion masks')
    monkeypatch.setattr(focus_masks, 'build_focus_labels', unexpected)
    paths = [tmp_path / f'frame{i}.tif' for i in range(2)]
    for path in paths:
        Image.new('RGB', (32, 24)).save(path)
    class Runner:
        def run(self, command, **kwargs):
            Image.new('RGB', (32, 24)).save(command[command.index('-o') + 1])
            return CommandResult(tuple(command), 0)
    result = Enfuser('enfuse', config=config, runner=Runner()).fuse(paths, tmp_path/'result.tif', work_dir=tmp_path/'work')
    assert '--levels=1' not in result.command_result.command


@pytest.mark.parametrize('background,foreground', [(35, 220), (220, 35)])
def test_real_enfuse_auto_levels_preserve_tone_stable_silhouette(tmp_path, background, foreground):
    executable = os.environ.get('FOCUS_STACK_ENFUSE') or shutil.which('enfuse')
    if not executable:
        pytest.skip('Set FOCUS_STACK_ENFUSE to run real Enfuse integration')
    sharp = np.full((240, 320, 3), background, np.uint8)
    sharp[50:190, 100:220] = foreground
    blurred = cv2.GaussianBlur(sharp, (0, 0), 12)
    paths = [tmp_path / 'sharp.tif', tmp_path / 'blurred.tif']
    for path, frame in zip(paths, (sharp, blurred)):
        Image.fromarray(frame).save(path)
    result = Enfuser(executable).fuse(paths, tmp_path / 'result.tif', work_dir=tmp_path / 'work')
    actual = np.asarray(Image.open(result.output_path).convert('RGB'))
    assert '--levels=1' in result.command_result.command
    assert np.all(actual >= np.minimum(sharp, blurred))
    assert np.all(actual <= np.maximum(sharp, blurred))
    assert np.abs(actual.astype(float) - sharp).mean() < 0.3


@pytest.mark.parametrize('explicit_levels,expected', [(None, 5), (1, 1)])
def test_real_enfuse_uses_multiband_for_surface_tone_drift_unless_overridden(tmp_path, explicit_levels, expected):
    executable = os.environ.get('FOCUS_STACK_ENFUSE') or shutil.which('enfuse')
    if not executable:
        pytest.skip('Set FOCUS_STACK_ENFUSE to run real Enfuse integration')
    a = np.full((240, 320, 3), (185, 35, 35), np.uint8)
    cv2.putText(a, 'EDGE', (45, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (225, 225, 225), 2)
    b = np.clip(a.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    paths = [tmp_path / 'a.tif', tmp_path / 'b.tif']
    for path, frame in zip(paths, (a, b)):
        Image.fromarray(frame).save(path)
    result = Enfuser(executable, config=EnfuseConfig(focus_blend_levels=explicit_levels)).fuse(
        paths, tmp_path / 'result.tif', work_dir=tmp_path / 'work')
    assert f'--levels={expected}' in result.command_result.command


def test_real_enfuse_tone_drift_does_not_pull_defocused_white_rim_into_red(tmp_path):
    executable = os.environ.get('FOCUS_STACK_ENFUSE') or shutil.which('enfuse')
    if not executable:
        pytest.skip('Set FOCUS_STACK_ENFUSE to run real Enfuse integration')
    sharp = np.full((300, 500, 3), (185, 35, 35), np.uint8)
    sharp[60:240, 180:320] = 220
    # Focus breathing/exposure drift requires multiband blending, while the
    # defocused white feature must not add a pale band to its red surround.
    blurred = np.clip(cv2.GaussianBlur(sharp, (0, 0), 12).astype(np.int16) + 8,
                      0, 255).astype(np.uint8)
    paths = [tmp_path / 'sharp.tif', tmp_path / 'blurred.tif']
    for path, frame in zip(paths, (sharp, blurred)):
        Image.fromarray(frame).save(path)
    result = Enfuser(executable).fuse(paths, tmp_path / 'result.tif', work_dir=tmp_path / 'work')
    actual = np.asarray(Image.open(result.output_path).convert('RGB')).astype(float)
    assert '--levels=5' in result.command_result.command
    # The old seven-pixel support produced ~5 levels of green contamination
    # 12--20 pixels outside the edge and ~4.7 levels of white-feature error.
    assert np.abs(actual[90:210, 160:168, 1] - 35).mean() < 1.0
    assert np.abs(actual[70:230, 190:310] - sharp[70:230, 190:310]).mean() < 1.1


@pytest.mark.parametrize('support_radius', [7, 39])
@pytest.mark.parametrize('reverse', [False, True])
def test_neutral_background_texture_cannot_own_coloured_subject_edge(support_radius, reverse):
    rng = np.random.default_rng(12)
    background = np.clip(110 + rng.normal(0, 35, (300, 500)), 0, 255).astype(np.uint8)
    background = np.repeat(background[..., None], 3, axis=2)
    mask = np.zeros((300, 500), np.float32)
    mask[100:200, 50:450] = 1
    near = cv2.GaussianBlur(background, (0, 0), 7)
    near[mask > 0] = (185, 35, 35)
    alpha = cv2.GaussianBlur(mask, (0, 0), 7)[..., None]
    far = np.uint8(background * (1 - alpha) + np.float32([185, 35, 35]) * alpha)
    frames = (far, near) if reverse else (near, far)
    labels = build_focus_labels(2, lambda i: frames[i], protect_chromatic_edges=True,
                                focus_support_radius=support_radius)
    # The old luminance score picked the background-focused frame throughout
    # this band, carrying its blurred red silhouette into the output.
    assert np.mean(labels[95:106, 100:400] == int(reverse)) > .99
    # Detail well outside the protected edge must still use the sharp backdrop.
    assert np.mean(labels[40:70, 100:400] == int(not reverse)) > .99


def noisy_flat_background():
    rng = np.random.default_rng(23)
    ramp = np.broadcast_to(np.linspace(180, 215, 720), (360, 720))
    frames = []
    for index, offset in enumerate((-8, -4, 0, 4, 8)):
        gray = np.clip(ramp + offset + rng.normal(0, 1.2, ramp.shape), 0, 255)
        frame = np.repeat(gray[..., None], 3, axis=2).astype(np.uint8)
        # A coloured subject and two neutral fine details at different depths.
        frame[150:300, 260:420] = (185 + offset, 35 + offset, 35 + offset)
        for x, owner in ((520, 0), (620, 4)):
            line = np.zeros((360, 720), np.float32)
            line[100:260, x:x + 3] = 9
            if index != owner:
                line = cv2.GaussianBlur(line, (0, 0), 5)
            frame = np.clip(frame.astype(float) - line[..., None], 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


@pytest.mark.parametrize('radius', [7, 39])
@pytest.mark.parametrize('order', [(0, 1, 2, 3, 4), (4, 0, 3, 1, 2)])
def test_flat_background_is_stable_without_discarding_neutral_detail(radius, order):
    original = noisy_flat_background()
    frames = [original[i] for i in order]
    labels = build_focus_labels(len(frames), lambda i: frames[i], focus_support_radius=radius)
    # An entire flat region uses the middle exposure, preserving its gradient.
    assert np.mean(labels[30:330, 20:130] == order.index(2)) > .99
    for x, owner in ((520, 0), (620, 4)):
        assert np.mean(labels[130:230, x:x + 3] == order.index(owner)) > .95


def test_real_enfuse_flat_background_preserves_source_gradient(tmp_path):
    executable = os.environ.get('FOCUS_STACK_ENFUSE') or shutil.which('enfuse')
    if not executable:
        pytest.skip('Set FOCUS_STACK_ENFUSE to run real Enfuse integration')
    frames = noisy_flat_background()
    paths = [tmp_path / f'noise_{i}.tif' for i in range(len(frames))]
    for path, frame in zip(paths, frames):
        Image.fromarray(frame).save(path)
    result = Enfuser(executable).fuse(paths, tmp_path/'result.tif', work_dir=tmp_path/'work')
    actual = np.asarray(Image.open(result.output_path).convert('RGB')).astype(float)
    roi = np.s_[40:320, 30:120]
    assert np.abs(actual[roi] - frames[2][roi]).mean() < .5
    # Stabilising a flat background must not flatten the lighting gradient.
    assert actual[150, 110].mean() - actual[150, 40].mean() > 1


@pytest.mark.parametrize('reverse', [False, True])
def test_smooth_backdrop_keeps_sharp_neutral_rim_despite_stronger_blurred_chroma(reverse):
    near = np.full((300, 500, 3), 220, np.uint8)
    near[75:225, 100:400] = 235
    near[80:220, 105:395] = (150, 90, 90)
    cv2.putText(near, 'EDGE', (145, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (235, 235, 235), 3)
    far = np.full_like(near, 220)
    far[75:225, 100:400] = 235
    # Similar luminance but stronger chroma: a colour-only override used to
    # replace the correctly focused thin rim with this blurred exposure.
    far[80:220, 105:395] = (230, 64, 15)
    cv2.putText(far, 'EDGE', (145, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (235, 235, 235), 3)
    far = cv2.GaussianBlur(far, (0, 0), 3)
    frames = (far, near) if reverse else (near, far)
    labels = build_focus_labels(2, lambda i: frames[i], protect_chromatic_edges=True,
                                focus_support_radius=39)
    assert np.mean(labels[70:90, 130:370] == int(reverse)) > .99
    assert np.mean(labels[110:170, 170:320] == int(reverse)) > .98


@pytest.mark.parametrize('reverse', [False, True])
def test_chromatic_guard_cannot_defocus_dark_metal_rim(reverse):
    rng = np.random.default_rng(9)
    background = np.repeat(np.clip(220 + rng.normal(0, 15, (300, 500)), 0, 255)
                           .astype(np.uint8)[..., None], 3, axis=2)
    near = cv2.GaussianBlur(background, (0, 0), 4)
    near[75:225, 100:400] = 30
    near[83:217, 108:392] = (150, 90, 90)
    far = near.copy()
    far[83:217, 108:392] = (230, 64, 15)
    far = cv2.GaussianBlur(far, (0, 0), 3)
    foreground = np.zeros((300, 500), np.uint8)
    foreground[55:245, 80:420] = 1
    far[foreground == 0] = background[foreground == 0]
    frames = (far, near) if reverse else (near, far)
    labels = build_focus_labels(2, lambda i: frames[i], protect_chromatic_edges=True,
                                focus_support_radius=39)
    # External texture makes the colour guard eligible. Stronger chroma in
    # the far frame still must not override an already sharp dark metal rim.
    # Without the local edge check, this band selected the far frame entirely.
    assert np.mean(labels[72:90, 130:370] == int(reverse)) > .99


def test_dark_body_and_pale_rim_keep_separate_sharp_focus_planes():
    rng = np.random.default_rng(31)
    truth = np.full((300, 600, 3), 240, np.uint8)
    truth[125:145] = 185
    truth[127:143, ::9] = 125
    texture = rng.integers(-25, 26, (75, 600, 1), dtype=np.int16)
    truth[145:220] = np.clip(65 + texture, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(truth, (0, 0), 5)
    body = truth.copy()
    body[115:145] = blurred[115:145]
    rim = blurred.copy()
    rim[115:145] = truth[115:145]
    far = cv2.GaussianBlur(truth, (0, 0), 10)
    frames = (body, rim, far)
    guard = _CoherentNeutralEdges(body)
    for frame in frames:
        guard.observe(frame)
    # Fragmented old winners selected an out-of-focus plane throughout.
    labels = np.full(truth.shape[:2], 2, np.uint16)
    guard.apply(labels, gray_frames={0: cv2.cvtColor(body, cv2.COLOR_RGB2GRAY)},
                load_aligned=lambda i: frames[i])
    assert np.mean(labels[165:200, 40:560] == 0) > .95
    assert np.mean(labels[127:140, 40:560] == 1) > .95


def test_translucent_lip_does_not_inherit_coloured_body_focus():
    height, width = 520, 800
    rng = np.random.default_rng(92)
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    top = 275 + np.rint(16 * np.sin(xx * 2 * np.pi / 125)).astype(int)
    body = yy >= top
    lip = (yy >= top - 35) & (yy < top - 3)
    line = (yy >= top - 3) & (yy < top)
    truth = np.full((height, width, 3), 240, np.uint8)
    truth[body] = (20, 205, 210)
    texture = rng.integers(-20, 21, (height, width), dtype=np.int16)
    shade = np.uint8(np.clip(175 + texture, 0, 255))
    truth[lip] = np.repeat(shade[:, :, None], 3, axis=2)[lip]
    truth[line] = (85, 85, 85)
    blurred = cv2.GaussianBlur(truth, (0, 0), 5)
    body_frame = truth.copy()
    body_frame[lip | line] = blurred[lip | line]
    line_frame = blurred.copy()
    line_frame[line] = truth[line]
    lip_frame = blurred.copy()
    lip_frame[lip] = truth[lip]
    frames = (body_frame, line_frame, lip_frame)
    guard = _CoherentNeutralEdges(body_frame)
    assert guard.pale_rims
    for frame in frames:
        guard.observe(frame)
    labels = np.zeros((height, width), np.uint16)
    guard.apply(labels, load_aligned=lambda index: frames[index])
    middle = slice(150, 650)
    assert np.mean(labels[top[0, middle] - 2, np.arange(150, 650)] == 1) > .7
    assert np.mean(labels[top[0, middle] - 30, np.arange(150, 650)] == 2) > .7

