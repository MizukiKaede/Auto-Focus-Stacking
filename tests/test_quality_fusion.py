from __future__ import annotations

import threading

import cv2
import numpy as np
from PIL import Image

from focus_stack_app.fusion.backends import HuginEnfuseBackend, QualityFusionBackend
from focus_stack_app.fusion.quality_fusion import stabilize_neutral_labels
from focus_stack_app.hugin.output_encoder import OutputConfig
from focus_stack_app.pipeline.merge_worker import StackMergeService


def test_neutral_label_islands_are_removed_without_changing_coloured_detail():
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[28:31, 28:31] = 1
    labels[10:13, 45:48] = 1
    reference = np.full((64, 64, 3), 90, dtype=np.uint8)
    reference[:, 40:] = (20, 180, 200)

    stable = stabilize_neutral_labels(labels, reference)

    assert np.all(stable[28:31, 28:31] == 0)
    assert np.all(stable[10:13, 45:48] == 1)


def test_only_two_modes_route_to_their_backends(tmp_path):
    assert isinstance(StackMergeService(tmp_path / 'quality', archive_enabled=False).backend,
                      QualityFusionBackend)
    assert isinstance(StackMergeService(tmp_path / 'experimental', archive_enabled=False,
                                        fusion_backend='experimental').backend, HuginEnfuseBackend)
    assert isinstance(StackMergeService(tmp_path / 'old-fast', archive_enabled=False,
                                        fusion_backend='opencv').backend, QualityFusionBackend)


def test_quality_mode_fuses_full_resolution_edges_without_hugin_tiffs(tmp_path):
    truth = np.full((240, 360, 3), 235, np.uint8)
    truth[40:205, 70:105] = (35, 40, 45)
    truth[120:210, 105:310] = (20, 195, 205)
    cv2.putText(truth, "A", (185, 185), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (250, 250, 250), 3)
    blurred = cv2.GaussianBlur(truth, (0, 0), 4)
    near = truth.copy()
    near[:, 180:] = blurred[:, 180:]
    far = truth.copy()
    far[:, :180] = blurred[:, :180]
    paths = [tmp_path / "near.png", tmp_path / "far.png"]
    for path, pixels in zip(paths, (near, far)):
        Image.fromarray(pixels).save(path)

    identity = np.eye(3).tolist()
    result = QualityFusionBackend().fuse(
        {}, {"selected_paths": paths, "selected_indices": [0, 1],
             "preview_reference": paths[0], "preview_reference_index": 0,
             "preview_transforms": [identity, identity],
             "analysis_shapes": [[240, 360], [240, 360]],
             "reference_analysis_shape": [240, 360]},
        tmp_path / "quality.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
    )

    with Image.open(result.output_path) as image:
        actual = np.asarray(image.convert("RGB"), dtype=np.int16)
    assert result.actual_backend == "quality"
    assert result.alignment_status == "PREVIEW_TRANSFORMS"
    assert actual.shape == truth.shape
    assert np.abs(actual[:, 70:105] - truth[:, 70:105]).mean() < 4
    assert np.abs(actual[130:200, 185:290] - truth[130:200, 185:290]).mean() < 4
    assert not list((tmp_path / "work").rglob("*.tif"))
