import os
import logging
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace

from PIL import Image
import pytest

from focus_stack_app.fusion.backends import HuginEnfuseBackend
from focus_stack_app.hugin.align import AlignConfig, AlignImageStack
from focus_stack_app.hugin.output_encoder import OutputConfig
from focus_stack_app.hugin.process import CommandResult
from focus_stack_app.pipeline.application_controller import cleanup_stale_temp_dirs
from focus_stack_app.pipeline.merge_worker import StackMergeService
from focus_stack_app.utils import temp_storage


def _drive(path):
    return Path(path).drive.rstrip(":").upper()


@pytest.mark.parametrize(
    ("media", "reason"),
    [("SSD", "cache_on_ssd"), ("HDD", "no_detected_ssd_keep_cache"),
     ("UNKNOWN", "no_detected_ssd_keep_cache")],
)
def test_select_hugin_temp_root_respects_detected_media(monkeypatch, tmp_path, media, reason):
    cache_dir = tmp_path / "cache"
    relevant = {
        _drive(cache_dir),
        _drive(tempfile.gettempdir()),
        _drive(Path(temp_storage.__file__).resolve().parents[2]),
    }
    monkeypatch.setattr(
        temp_storage, "windows_drive_media",
        lambda: {drive: media for drive in relevant},
    )

    root, selected_reason, selected_media = temp_storage.select_hugin_temp_root(cache_dir)

    expected = cache_dir / "temp"
    assert root == expected
    assert selected_reason == reason
    assert selected_media == media


def test_select_hugin_temp_root_explicit_path_only_changes_selection_flag(monkeypatch, tmp_path):
    drive = _drive(tmp_path)
    monkeypatch.setattr(temp_storage, "windows_drive_media", lambda: {drive: "HDD"})
    explicit = tmp_path / "private-hugin"

    root, reason, media = temp_storage.select_hugin_temp_root(
        tmp_path / "cache", explicit=explicit,
    )

    assert root == explicit.resolve()
    assert reason == "explicit"
    assert media == "HDD"


def test_private_temp_cleanup_removes_only_old_direct_work_dirs(tmp_path):
    root = tmp_path / "private-temp"
    root.mkdir()
    valid = root / ("7_" + "a" * 32)
    keep = root / ("8_" + "b" * 32)
    invalid = root / "not-a-work-directory"
    nested = root / "nested" / ("9_" + "c" * 32)
    for path in (valid, keep, invalid, nested):
        path.mkdir(parents=True)
        (path / "payload.tif").write_bytes(b"diagnostic")
    (keep / ".keep").touch()

    removed = cleanup_stale_temp_dirs(
        root, max_age_seconds=60, now=time.time() + 3600,
    )

    assert removed == [valid.resolve()]
    assert not valid.exists()
    assert keep.exists() and (keep / "payload.tif").exists()
    assert invalid.exists() and (invalid / "payload.tif").exists()
    assert nested.exists() and (nested / "payload.tif").exists()


def _capture_hugin_command(tmp_path, runtime_config):
    tmp_path.mkdir(parents=True, exist_ok=True)
    anchor = tmp_path / "anchor.jpg"
    other = tmp_path / "other.jpg"
    Image.new("RGB", (32, 32), "red").save(anchor)
    Image.new("RGB", (32, 32), "blue").save(other)
    commands = []

    class Runner:
        def run(self, command, **kwargs):
            commands.append(list(command))
            prefix = Path(command[command.index("-a") + 1])
            for index in range(2):
                path = Path(f"{prefix}{index:04d}.tif")
                Image.new("RGB", (32, 32), "green").save(path)
            return CommandResult(tuple(command), 0)

    class Fusion:
        def fuse(self, paths, output_path, **kwargs):
            output = Path(output_path)
            Image.new("RGB", (32, 32), "green").save(output)
            return SimpleNamespace(output_path=output)

    backend = HuginEnfuseBackend(
        AlignImageStack("align.exe", runner=Runner()), Fusion(),
        runtime_config=runtime_config,
    )
    backend._validate_alignment = lambda *args, **kwargs: (1.0, [])
    backend.fuse(
        {}, {"alignment_order": [anchor, other], "selected_paths": [anchor, other],
             "first_original_path": anchor},
        tmp_path / "result.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
    )
    return commands[0]


def test_hugin_alignment_preset_first_adds_only_explicit_flag(tmp_path):
    legacy = _capture_hugin_command(tmp_path / "legacy", None)
    first = _capture_hugin_command(tmp_path / "first", {"hugin_alignment_preset": "first"})

    assert "--align-to-first" not in legacy
    assert "--align-to-first" in first
    legacy_options = legacy[:legacy.index("-a")]
    first_options = [item for item in first[:first.index("-a")]
                     if item != "--align-to-first"]
    assert first_options == legacy_options


def _reference_first_backend(tmp_path):
    paths = [tmp_path / name for name in ("first.jpg", "mid_reference.jpg", "last.jpg")]
    for path, color in zip(paths, ("red", "green", "blue")):
        Image.new("RGB", (32, 32), color).save(path)
    commands = []

    class Runner:
        def run(self, command, **kwargs):
            commands.append(list(command))
            prefix = Path(command[command.index("-a") + 1])
            input_count = len(command) - command.index("-a") - 2
            for index in range(input_count):
                Image.new("RGB", (32, 32), "white").save(Path(f"{prefix}{index:04d}.tif"))
            return CommandResult(tuple(command), 0)

    class Fusion:
        def fuse(self, paths, output_path, **kwargs):
            output = Path(output_path)
            Image.new("RGB", (32, 32), "white").save(output)
            return SimpleNamespace(output_path=output)

    backend = HuginEnfuseBackend(
        AlignImageStack(
            "align.exe", runner=Runner(),
            config=AlignConfig(extra_args=("--keep-extra",)),
        ),
        Fusion(),
        runtime_config={"hugin_alignment_preset": "reference_first"},
    )
    backend._validate_alignment = lambda *args, **kwargs: (1.0, [])
    return backend, commands, paths


def test_hugin_reference_first_reorders_mid_reference_and_preserves_command_options(tmp_path):
    backend, commands, paths = _reference_first_backend(tmp_path)

    backend.fuse(
        {},
        {
            "alignment_order": paths,
            "selected_paths": paths,
            "preview_reference": paths[1],
            "first_original_path": paths[0],
        },
        tmp_path / "result.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
    )

    command = commands[0]
    output_option = command.index("-a")
    assert command[output_option + 2:] == [str(paths[1]), str(paths[0]), str(paths[2])]
    assert "--align-to-first" in command
    assert "-m" in command and "-C" in command
    assert "--keep-extra" in command


def test_hugin_reference_first_rejects_unselected_preview_reference(tmp_path):
    backend, commands, paths = _reference_first_backend(tmp_path)

    with pytest.raises(ValueError, match="requires a selected preview reference"):
        backend.fuse(
            {},
            {
                "alignment_order": paths,
                "selected_paths": paths,
                "preview_reference": tmp_path / "missing-reference.jpg",
            },
            tmp_path / "result.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
        )

    assert commands == []


def test_hugin_rejects_alignment_order_with_unselected_input(tmp_path):
    backend, commands, paths = _reference_first_backend(tmp_path)
    extra = tmp_path / "unselected.jpg"
    Image.new("RGB", (32, 32), "yellow").save(extra)

    with pytest.raises(ValueError, match="does not contain exactly the selected inputs"):
        backend.fuse(
            {},
            {
                "alignment_order": [paths[0], paths[1], extra],
                "selected_paths": paths,
                "preview_reference": paths[1],
            },
            tmp_path / "result.jpg", tmp_path / "work", OutputConfig(), threading.Event(),
        )

    assert commands == []


def test_stack_service_ssd_mkdir_failure_falls_back_to_private_cache(monkeypatch, tmp_path, caplog):
    cache_dir = tmp_path / "cache"
    candidate = tmp_path / "detected-ssd"
    explicit = tmp_path / "caller-owned"

    def select_root(cache, *, explicit=None):
        if explicit is not None:
            return Path(explicit).resolve(), "explicit", "UNKNOWN"
        return candidate, "detected_ssd", "SSD"

    monkeypatch.setattr(temp_storage, "select_hugin_temp_root", select_root)
    real_mkdir = Path.mkdir

    def deny_detected_ssd(path, *args, **kwargs):
        if path == candidate:
            raise PermissionError("simulated unavailable SSD root")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_detected_ssd)
    caplog.set_level(logging.WARNING)
    service = StackMergeService(
        tmp_path / "output", cache_dir=cache_dir, archive_enabled=False,
        fusion_backend="hugin_enfuse",
    )

    fallback = cache_dir / "temp"
    assert service.temp_root == fallback
    assert "detected Hugin temporary storage is unavailable" in caplog.text

    real_mkdir(fallback, parents=True, exist_ok=True)
    valid = fallback / ("group_" + "a" * 32)
    keep = fallback / ("group_" + "b" * 32)
    unexpected = fallback / "unexpected"
    real_mkdir(valid, parents=True)
    real_mkdir(keep, parents=True)
    real_mkdir(unexpected, parents=True)
    (keep / ".keep").touch()
    removed = cleanup_stale_temp_dirs(fallback, max_age_seconds=0, now=time.time() + 3600)
    assert removed == [valid.resolve()]
    assert not valid.exists() and keep.exists() and unexpected.exists()

    explicit_service = StackMergeService(
        tmp_path / "explicit-output", cache_dir=cache_dir, archive_enabled=False,
        fusion_backend="hugin_enfuse",
        runtime_config={"hugin_temp_directory": str(explicit)},
    )
    assert explicit_service.temp_root == explicit.resolve()
    assert not explicit.exists()
