import threading
import os
import shutil

import cv2
import numpy as np
import pytest
from PIL import Image

from focus_stack_app.fusion.backends import OpenCVFusionBackend
from focus_stack_app.fusion.focus_masks import build_focus_labels, focus_weight, blend_focus_pyramid
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


def test_opencv_preserves_thin_details_at_adjacent_focus_depths(tmp_path):
    truth, left, right = thin_details()
    paths = [tmp_path / 'near.png', tmp_path / 'far.png']
    for path, pixels in zip(paths, (left, right)):
        Image.fromarray(pixels).save(path)
    result = OpenCVFusionBackend().fuse(
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
    assert '--levels=5' in result.command_result.command
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
    assert '--levels=5' not in result.command_result.command

