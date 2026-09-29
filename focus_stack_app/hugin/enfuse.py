"""Enfuse adapter for focus-priority blending."""

from __future__ import annotations

from ..utils.performance import timed

from ..utils.performance import stage

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
from typing import Iterable

from .hugin_locator import HuginLocator
from .output_encoder import OutputCollisionError, OutputConfig, OutputFormat, encode_output
from .process import CommandResult, SubprocessRunner


@dataclass(frozen=True)
class EnfuseConfig:
    """Enfuse weights tuned for focus stacking rather than exposure fusion."""

    contrast_weight: float = 1.0
    exposure_weight: float = 0.0
    saturation_weight: float = 0.0
    entropy_weight: float = 0.0
    hard_mask: bool = True
    extra_args: tuple[str, ...] = ()
    timeout_seconds: float | None = 60 * 60
    jpeg_quality: int = 100
    full_resolution_focus_masks: bool = True
    # Auto keeps single-level silhouette protection for stable tones, and
    # uses five levels when flat surfaces drift in colour between frames.
    # An explicit integer overrides this choice.
    focus_blend_levels: int | None = None
    focus_gray_cache_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if self.focus_blend_levels is not None and not 1 <= self.focus_blend_levels <= 29:
            raise ValueError("focus_blend_levels must be between 1 and 29")
        if int(self.focus_gray_cache_bytes) < 0:
            raise ValueError("focus_gray_cache_bytes cannot be negative")


@dataclass
class EnfuseResult:
    input_paths: tuple[Path, ...]
    output_path: Path
    temporary_output: Path
    work_dir: Path
    command_result: CommandResult

    @property
    def ok(self) -> bool:
        return self.command_result.ok and self.output_path.is_file()


class EnfuseError(RuntimeError):
    def __init__(self, message: str, result: EnfuseResult | None = None):
        self.result = result
        super().__init__(message)


class Enfuser:
    """Run Enfuse and atomically publish the final output."""

    def __init__(
        self,
        executable: os.PathLike[str] | str | None = None,
        *,
        locator: HuginLocator | None = None,
        runner: SubprocessRunner | None = None,
        config: EnfuseConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.executable = Path(executable) if executable else None
        self.locator = locator or HuginLocator(executable)
        self.runner = runner or SubprocessRunner(logger=logger)
        self.config = config or EnfuseConfig()
        self.logger = logger or logging.getLogger(__name__)

    def executable_path(self) -> Path:
        if self.executable is not None:
            if self.executable.is_dir():
                return self.locator.require("enfuse")
            return self.executable
        return self.locator.require("enfuse")

    def build_command(
        self,
        aligned_paths: Iterable[os.PathLike[str] | str],
        output_path: os.PathLike[str] | str,
    ) -> list[str]:
        paths = tuple(Path(item) for item in aligned_paths)
        if not paths:
            raise ValueError("At least one aligned image is required for enfuse")
        command = [str(self.executable_path())]
        command.extend(
            [
                f"--contrast-weight={self.config.contrast_weight:g}",
                f"--exposure-weight={self.config.exposure_weight:g}",
                f"--saturation-weight={self.config.saturation_weight:g}",
                f"--entropy-weight={self.config.entropy_weight:g}",
            ]
        )
        if self.config.hard_mask:
            command.append("--hard-mask")
        if Path(output_path).suffix.casefold() in {".jpg", ".jpeg"}:
            command.append(f"--compression={self.config.jpeg_quality}")
        command.extend(self.config.extra_args)
        # ``-o`` is supported by all released Enfuse versions and avoids the
        # ambiguity of whether a particular build accepts ``--output FILE``
        # versus ``--output=FILE``.
        command.extend(["-o", str(output_path)])
        command.extend(str(path) for path in paths)
        return command

    @timed("enfuse_inclusive")
    def fuse(
        self,
        aligned_paths: Iterable[os.PathLike[str] | str],
        output_path: os.PathLike[str] | str,
        *,
        work_dir: os.PathLike[str] | str | None = None,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
        cleanup_on_success: bool = True,
        output_config: OutputConfig | None = None,
        image_loader=None,
    ) -> EnfuseResult:
        paths = tuple(Path(item) for item in aligned_paths)
        if not paths:
            raise ValueError("At least one aligned image is required for enfuse")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Enfuse input does not exist: {path}")
        if (
            self.executable is not None
            and not self.executable.is_dir()
            and not self.executable.is_file()
            and isinstance(self.runner, SubprocessRunner)
        ):
            from .hugin_locator import HuginToolNotFound
            raise HuginToolNotFound("enfuse", (self.executable,))
        final = Path(output_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            raise OutputCollisionError(f"Refusing to overwrite existing output: {final}")
        source_original = next((p for p in paths if p.resolve(strict=False) == final.resolve(strict=False)), None)
        if source_original is not None:
            raise OutputCollisionError(f"Output cannot overwrite an input image: {final}")
        work = Path(work_dir) if work_dir is not None else final.parent / ".stack_cache" / "temp" / uuid.uuid4().hex
        work.mkdir(parents=True, exist_ok=True)
        # Always let Enfuse finish into a private path.  This prevents a killed
        # process from leaving a file that looks like a completed deliverable.
        # Lossless intermediate: encode JPEG once with explicit 4:4:4 sampling.
        ext = ".tif"
        temporary = work / f"enfused_{uuid.uuid4().hex}{ext}"
        command = self.build_command(paths, temporary)
        # Keep Enfuse's lossless/pyramid renderer, but supply reliable focus
        # decisions instead of its default local-contrast winner masks.
        # Explicit caller-supplied masks and exposure fusion remain supported.
        mask_options = ("--load-masks", "--save-masks", "--soft-mask")
        if (
            len(paths) > 1 and self.config.full_resolution_focus_masks
            and self.config.hard_mask and self.config.contrast_weight > 0
            and self.config.exposure_weight == self.config.saturation_weight == self.config.entropy_weight == 0
            and not any(arg.startswith(mask_options) for arg in self.config.extra_args)
        ):
            from PIL import Image
            from ..fusion.focus_masks import build_focus_labels, SurfaceToneMonitor
            from ..utils.image_io import load_rgb

            with stage("focus_masks_inclusive", frames=len(paths)):
                aligned_loader = image_loader or (lambda i: load_rgb(paths[i]))
                tone_monitor = SurfaceToneMonitor() if self.config.focus_blend_levels is None else None
                labels = build_focus_labels(
                    len(paths), aligned_loader, cancel_event=cancel_event,
                    protect_chromatic_edges=True,
                    gray_cache_bytes=self.config.focus_gray_cache_bytes, tone_monitor=tone_monitor,
                    # Five-level blending reaches beyond the old seven-pixel
                    # ownership band and can mix a defocused white letter's
                    # rim into red print. Cover an extra 32 source pixels.
                    # A rectangular max filter keeps this linear-time without
                    # storing another full-resolution score/label pyramid.
                    focus_support_radius=7 if self.config.focus_blend_levels == 1 else 39,
                )
                digits = len(str(len(paths)))
                for index in range(len(paths)):
                    if cancel_event is not None and cancel_event.is_set():
                        raise EnfuseError("Enfuse cancelled while generating focus masks")
                    mask = (labels == index).astype("uint8") * 255
                    Image.fromarray(mask).save(work / f"hardmask-{index + 1:0{digits}}.tif", compression="tiff_deflate")
                del labels, mask
            # Use the default relative templates: custom template arguments
            # crash some Windows Enfuse builds. Enfuse pads
            # mask numbers to the number of digits in the input count.
            options = ["--load-masks"]
            if not any(arg == "-l" or arg.startswith(("--levels", "-l")) for arg in self.config.extra_args):
                blend_levels = self.config.focus_blend_levels
                if blend_levels is None:
                    blend_levels = 5 if tone_monitor.needs_multiband else 1
                options.append(f"--levels={blend_levels}")
                self.logger.info("Focus blend levels=%s flat_surface_tone_drift=%s",
                                 blend_levels, bool(tone_monitor and tone_monitor.needs_multiband))
            command[1:1] = options
            self.logger.info("Full-resolution focus masks generated for %s aligned frames", len(paths))
        # Mirror the runner audit here so custom/mock runners still publish
        # command and return-code evidence to the application's shared log.
        self.logger.info("enfuse command: %s", command)
        with stage("fusion", backend="enfuse", output=str(final)):
            command_result = self.runner.run(
                command,
                cwd=work,
                timeout=self.config.timeout_seconds if timeout is None else timeout,
                cancel_event=cancel_event,
                check=False,
            )
        self.logger.info(
            "enfuse finished rc=%s timeout=%s cancelled=%s",
            command_result.returncode,
            command_result.timed_out,
            command_result.cancelled,
        )
        result = EnfuseResult(paths, final, temporary, work, command_result)
        if command_result.cancelled:
            raise EnfuseError("Enfuse cancelled; temporary files kept for inspection", result)
        if command_result.timed_out:
            raise EnfuseError(f"Enfuse timed out after {command_result.elapsed:.1f}s; temporary files kept at {work}", result)
        if command_result.returncode != 0:
            details = command_result.stderr.strip() or command_result.stdout.strip()
            raise EnfuseError(
                f"enfuse failed with exit code {command_result.returncode}: {details}\nTemporary files kept at {work}",
                result,
            )
        if not temporary.is_file():
            raise EnfuseError(f"enfuse reported success but produced no output at {temporary}", result)

        try:
            if final.suffix.casefold() in {".jpg", ".jpeg"}:
                cfg = output_config or OutputConfig(format=OutputFormat.JPG, jpeg_quality=self.config.jpeg_quality)
                encode_output(temporary, final, config=cfg)
            else:
                if final.exists():
                    raise OutputCollisionError(f"Refusing to overwrite existing output: {final}")
                os.replace(temporary, final)
            if cleanup_on_success:
                self._cleanup_temp(work, keep=final)
        except Exception as exc:
            raise EnfuseError(f"Unable to publish Enfuse output: {exc}; temporary files kept at {work}", result) from exc
        return result

    @staticmethod
    def _cleanup_temp(work: Path, *, keep: Path | None = None) -> None:
        """Delete only the private temporary directory contents on success."""

        try:
            for child in work.iterdir():
                if keep is not None and child.resolve(strict=False) == keep.resolve(strict=False):
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
            # Remove empty private levels; never remove the final output dir.
            current = work
            while current.name and current != current.parent:
                try:
                    current.rmdir()
                except OSError:
                    break
                current = current.parent
        except OSError:
            # Cleanup is best effort and should not convert a valid output into
            # a failed job.  The path is logged for manual cleanup.
            logging.getLogger(__name__).warning("Unable to clean Enfuse temp directory %s", work, exc_info=True)

    run = fuse
    execute = fuse


EnfuseRunner = Enfuser


def enfuse_images(aligned_paths: Iterable[os.PathLike[str] | str], output_path: os.PathLike[str] | str, *, executable: os.PathLike[str] | str | None = None, **kwargs: object) -> EnfuseResult:
    """Functional convenience wrapper around :class:`Enfuser`."""

    return Enfuser(executable, config=kwargs.pop("config", None)).fuse(aligned_paths, output_path, **kwargs)  # type: ignore[arg-type]


__all__ = ["EnfuseConfig", "Enfuser", "EnfuseError", "EnfuseResult", "EnfuseRunner", "enfuse_images"]

