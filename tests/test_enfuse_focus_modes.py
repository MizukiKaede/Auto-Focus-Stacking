import numpy as np
import pytest
from PIL import Image

from focus_stack_app.fusion import focus_masks
from focus_stack_app.hugin.enfuse import EnfuseConfig, Enfuser
from focus_stack_app.hugin.process import CommandResult


class _Runner:
    def run(self, command, **kwargs):
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
    [("legacy", True, False), ("quality", False, False), ("gate", False, True)],
)
def test_enfuser_focus_mode_contract(tmp_path, monkeypatch, mode, expected_protect, expects_observer):
    paths = _frames(tmp_path)
    captured = {}
    original = focus_masks.build_focus_labels

    def spy(count, loader, **kwargs):
        captured.update(kwargs)
        return original(count, loader, **kwargs)

    monkeypatch.setattr(focus_masks, "build_focus_labels", spy)
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


def test_enfuser_focus_mode_validates_mode_and_reference(tmp_path):
    paths = _frames(tmp_path)
    enfuser = Enfuser("enfuse", runner=_Runner())
    with pytest.raises(ValueError, match="focus_mask_mode"):
        enfuser.fuse(paths, tmp_path / "bad_mode.tif", focus_mask_mode="unknown")
    with pytest.raises(ValueError, match="focus reference"):
        enfuser.fuse(paths, tmp_path / "bad_reference.tif", focus_reference_index=2)
