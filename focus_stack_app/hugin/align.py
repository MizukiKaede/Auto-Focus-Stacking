"""``align_image_stack`` adapter.

Only low-level command execution belongs here.  Group selection and archive
state stay in the pipeline/storage layers, which makes this module usable with
real Hugin binaries as well as deterministic mock subprocesses in tests.
"""

from __future__ import annotations

from ..utils.performance import timed

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import threading
import time
from typing import Iterable, Sequence
import uuid

from .hugin_locator import HuginLocator, HuginToolNotFound
from .process import CommandResult, SubprocessRunner


@dataclass(frozen=True)
class AlignConfig:
    """Command options for focus-stack alignment."""

    optimize_field_of_view: bool = True
    crop_to_fit: bool = True
    optimize_scale: bool = True
    optimize_centre: bool = False
    extra_args: tuple[str, ...] = ()
    timeout_seconds: float | None = 60 * 60
    output_extension: str = ".tif"


@dataclass
class AlignmentResult:
    input_paths: tuple[Path, ...]
    aligned_paths: tuple[Path, ...]
    work_dir: Path
    output_prefix: Path
    command_result: CommandResult

    @property
    def ok(self) -> bool:
        return self.command_result.ok and bool(self.aligned_paths)

    @property
    def outputs(self) -> tuple[Path, ...]:
        return self.aligned_paths


class AlignmentError(RuntimeError):
    def __init__(self, message: str, result: AlignmentResult | None = None):
        self.result = result
        super().__init__(message)


def _normalise_paths(paths: Iterable[os.PathLike[str] | str]) -> tuple[Path, ...]:
    return tuple(Path(item) for item in paths)


class AlignImageStack:
    """Run Hugin alignment in an isolated working directory."""

    def __init__(
        self,
        executable: os.PathLike[str] | str | None = None,
        *,
        locator: HuginLocator | None = None,
        runner: SubprocessRunner | None = None,
        config: AlignConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.executable = Path(executable) if executable else None
        self.locator = locator or HuginLocator(executable)
        self.runner = runner or SubprocessRunner(logger=logger)
        self.config = config or AlignConfig()
        self.logger = logger or logging.getLogger(__name__)

    def executable_path(self) -> Path:
        if self.executable is not None:
            if self.executable.is_dir():
                return self.locator.require("align_image_stack")
            return self.executable
        return self.locator.require("align_image_stack")

    def build_command(
        self,
        image_paths: Iterable[os.PathLike[str] | str],
        output_prefix: os.PathLike[str] | str,
    ) -> list[str]:
        paths = _normalise_paths(image_paths)
        if len(paths) < 2:
            raise ValueError("At least two images are required for alignment")
        prefix = Path(output_prefix)
        command = [str(self.executable_path())]
        # FOV optimisation compensates focus breathing (magnification).
        # -x is camera translation, not scale; -t requires a CP error limit.
        if self.config.optimize_field_of_view or self.config.optimize_scale:
            command.append("-m")
        if self.config.crop_to_fit:
            command.append("-C")
        if self.config.optimize_centre:
            command.append("-i")
        command.append("--use-given-order")
        command.extend(self.config.extra_args)
        command.extend(["-a", str(prefix)])
        command.extend(str(path) for path in paths)
        return command

    @timed("hugin_alignment")
    def align(
        self,
        image_paths: Iterable[os.PathLike[str] | str],
        *,
        work_dir: os.PathLike[str] | str,
        output_prefix: os.PathLike[str] | str | None = None,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
    ) -> AlignmentResult:
        paths = _normalise_paths(image_paths)
        if len(paths) < 2:
            raise ValueError("At least two images are required for alignment")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Alignment input does not exist: {path}")
        if (
            self.executable is not None
            and not self.executable.is_dir()
            and not self.executable.is_file()
            and isinstance(self.runner, SubprocessRunner)
        ):
            raise HuginToolNotFound("align_image_stack", (self.executable,))
        work = Path(work_dir)
        work.mkdir(parents=True, exist_ok=True)
        prefix = Path(output_prefix) if output_prefix is not None else work / f"aligned_{uuid.uuid4().hex}_"
        if not prefix.is_absolute():
            prefix = work / prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)
        command = self.build_command(paths, prefix)
        # Keep the command audit on the adapter's logger as well as on the
        # default SubprocessRunner.  This is important for injected runners
        # (tests, wrappers, and integrations) that do not have their own
        # logging implementation; StackMergeService binds this logger to the
        # application log.
        self.logger.info("align_image_stack command: %s", command)
        result = self.runner.run(
            command,
            cwd=work,
            timeout=self.config.timeout_seconds if timeout is None else timeout,
            cancel_event=cancel_event,
            check=False,
        )
        self.logger.info(
            "align_image_stack finished rc=%s timeout=%s cancelled=%s",
            result.returncode,
            result.timed_out,
            result.cancelled,
        )
        output_glob = f"{prefix.name}*{self.config.output_extension}"
        outputs = tuple(sorted(work.glob(output_glob), key=lambda path: path.name.casefold()))
        alignment = AlignmentResult(paths, outputs, work, prefix, result)
        if result.cancelled:
            raise AlignmentError("Alignment cancelled", alignment)
        if result.timed_out:
            raise AlignmentError(f"Alignment timed out after {result.elapsed:.1f}s; temporary files kept at {work}", alignment)
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise AlignmentError(
                f"align_image_stack failed with exit code {result.returncode}: {details}\nTemporary files kept at {work}",
                alignment,
            )
        if not outputs:
            raise AlignmentError(
                f"align_image_stack reported success but produced no TIFF files in {work}",
                alignment,
            )
        if len(outputs) != len(paths):
            raise AlignmentError(
                f"Expected {len(paths)} aligned TIFFs, got {len(outputs)}; refusing incomplete fusion",
                alignment,
            )
        return alignment

    run = align
    execute = align


AlignImageStackRunner = AlignImageStack


def align_images(image_paths: Iterable[os.PathLike[str] | str], *, work_dir: os.PathLike[str] | str, executable: os.PathLike[str] | str | None = None, **kwargs: object) -> AlignmentResult:
    """Functional convenience wrapper around :class:`AlignImageStack`."""

    return AlignImageStack(executable, config=kwargs.pop("config", None)).align(image_paths, work_dir=work_dir, **kwargs)  # type: ignore[arg-type]


__all__ = ["AlignConfig", "AlignImageStack", "AlignImageStackRunner", "AlignmentError", "AlignmentResult", "HuginToolNotFound", "align_images"]
