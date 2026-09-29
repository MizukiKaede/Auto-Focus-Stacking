import cv2
import numpy as np
import pytest
from PIL import Image

from focus_stack_app.core.group_analyzer import GroupAnalyzer, GroupAnalyzerConfig
from focus_stack_app.core.types import SceneGroup
from focus_stack_app.fusion import QualityFusionBackend
from focus_stack_app.pipeline.application_controller import ApplicationController, ApplicationOptions
from focus_stack_app.pipeline.analysis_worker import AnalysisJob
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.storage.models import ImageRecord
from focus_stack_app.utils.image_io import load_rgb


def frames(tmp_path, count):
    truth = np.full((240, 360, 3), 235, np.uint8)
    truth[60:180, 40:320] = (170, 25, 35)
    cv2.putText(truth, "FOCUS STACK", (42, 125), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    blurred = cv2.GaussianBlur(truth, (0, 0), 2.5)
    records = []
    for index in range(count):
        image = truth.copy()
        if index % 2:
            image[:, :180] = blurred[:, :180]
        else:
            image[:, 180:] = blurred[:, 180:]
        path = tmp_path / f"frame{index}.jpg"
        Image.fromarray(image).save(path, quality=100, subsampling=0)
        records.append(ImageRecord.from_path(path, sequence_index=index))
    return SceneGroup(1, records)


@pytest.mark.parametrize("count", [2, 3, 4])
def test_four_similar_inputs_required_and_smaller_groups_skip_selection(tmp_path, count):
    group = frames(tmp_path, count)
    result = GroupAnalyzer(GroupAnalyzerConfig(minimum_stack_group_size=4)).analyze_group(group)
    assert result["needs_merge"] is (count >= 4)
    if count < 4:
        assert result["selected_indices"] == []
        assert result["merge_status"] == "NO_MERGE_TOO_SMALL"
        assert result["analysis_skipped"] is True
        assert result["selection_decode_count"] == 0
        merger = StackMergeService(tmp_path / "out", minimum_stack_group_size=4)
        assert merger.process(AnalysisJob(group=group, analysis=result)).output_path is None
    else:
        assert len(result["selected_indices"]) == 2


def test_rgb_controller_input_matches_original_loader(tmp_path):
    group = frames(tmp_path, 4)
    controller = ApplicationController(ApplicationOptions(source_dir=tmp_path, output_dir=tmp_path / "out"))
    path = group.items[0].path
    np.testing.assert_array_equal(controller._analysis_loader(group.items[0], 1280), load_rgb(path, 1280))
    assert isinstance(StackMergeService(tmp_path / "out").backend, QualityFusionBackend)


def test_failed_alignment_cannot_satisfy_four_frame_gate(tmp_path, monkeypatch):
    group = frames(tmp_path, 4)
    from focus_stack_app.core import whole_frame_selection
    original = whole_frame_selection.select_whole_frame

    def select(*args, **kwargs):
        plan = original(*args, **kwargs)
        rejected = next(i for i in range(4) if i not in plan["selected_indices"])
        plan["errors"][rejected] = "alignment rejected"
        return plan

    monkeypatch.setattr(whole_frame_selection, "select_whole_frame", select)
    result = GroupAnalyzer(GroupAnalyzerConfig(minimum_stack_group_size=4)).analyze_group(group)
    assert result["needs_merge"] is False
    assert result["merge_status"] == "NO_MERGE_TOO_SMALL"


@pytest.mark.parametrize("minimum", [2, 3, 4, 5])
def test_threshold_propagates_from_options(tmp_path, minimum):
    controller = ApplicationController(ApplicationOptions(
        source_dir=tmp_path, output_dir=tmp_path / "out", minimum_stack_group_size=minimum,
    ))
    assert controller.config.analysis.minimum_stack_group_size == minimum
    assert controller._build_analyzer()._minimum_stack_group_size() == minimum
    assert controller._build_merger().minimum_stack_group_size == minimum

