"""Small, testable subprocess runner for Hugin command line tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Mapping, Sequence


@dataclass
class CommandResult:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0
    timed_out: bool = False
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled


class CommandError(RuntimeError):
    def __init__(self, message: str, result: CommandResult | None = None):
        self.result = result
        super().__init__(message)


LineCallback = Callable[[str], None]


class SubprocessRunner:
    """Execute one command without blocking the Qt/UI thread.

    The runner itself is intended for a worker thread.  It polls a child at a
    short interval so cancellation and timeout remain responsive on Windows,
    where ``select`` cannot be used with ordinary pipe handles.
    """

    def __init__(self, *, logger: logging.Logger | None = None, poll_interval: float = 0.10):
        self.logger = logger or logging.getLogger(__name__)
        self.poll_interval = max(0.01, poll_interval)

    @staticmethod
    def _terminate(process: subprocess.Popen[str], *, force: bool = False) -> None:
        try:
            if os.name == "nt":
                # taskkill handles descendants spawned by wrapper .bat files;
                # CREATE_NEW_PROCESS_GROUP is not always honoured by Hugin.
                if force:
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    process.terminate()
            elif force:
                process.kill()
            else:
                process.terminate()
        except OSError:
            pass

    def run(
        self,
        command: Sequence[os.PathLike[str] | str],
        *,
        cwd: os.PathLike[str] | str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        cancel_event: threading.Event | None = None,
        on_output: LineCallback | None = None,
        check: bool = False,
    ) -> CommandResult:
        argv = tuple(os.fspath(item) for item in command)
        if not argv:
            raise ValueError("Cannot run an empty command")
        started = time.monotonic()
        self.logger.info("Hugin command: %s", subprocess.list2cmdline(list(argv)))
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        try:
            try:
                process = subprocess.Popen(
                    list(argv),
                    cwd=os.fspath(cwd) if cwd is not None else None,
                    env=dict(env) if env is not None else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
            except TypeError:
                # Small test doubles and older Python wrappers may not accept
                # Windows-only creationflags/encoding keyword arguments.
                process = subprocess.Popen(
                    list(argv),
                    cwd=os.fspath(cwd) if cwd is not None else None,
                    env=dict(env) if env is not None else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
        except OSError as exc:
            result = CommandResult(argv, None, stderr=str(exc), elapsed=time.monotonic() - started)
            if check:
                raise CommandError(f"Unable to start command: {argv[0]}: {exc}", result) from exc
            return result

        # communicate(timeout=...) works with both Windows and POSIX pipes. It
        # raises TimeoutExpired while retaining output captured so far.
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []

        def append_delta(parts: list[str], chunk: str) -> str:
            """Append only new text from communicate/TimeoutExpired output."""

            if not chunk:
                return ""
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", "replace")
            existing = "".join(parts)
            if existing and chunk.startswith(existing):
                chunk = chunk[len(existing) :]
            parts.append(chunk)
            return chunk
        timed_out = False
        cancelled = False
        while True:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                self._terminate(process)
                try:
                    out, err = process.communicate(timeout=max(1.0, self.poll_interval * 5))
                except subprocess.TimeoutExpired:
                    self._terminate(process, force=True)
                    out, err = process.communicate()
                append_delta(stdout_parts, out or "")
                append_delta(stderr_parts, err or "")
                break
            elapsed = time.monotonic() - started
            if timeout is not None and elapsed >= timeout:
                timed_out = True
                self._terminate(process)
                try:
                    out, err = process.communicate(timeout=max(1.0, self.poll_interval * 5))
                except subprocess.TimeoutExpired:
                    self._terminate(process, force=True)
                    out, err = process.communicate()
                append_delta(stdout_parts, out or "")
                append_delta(stderr_parts, err or "")
                break
            try:
                remaining = self.poll_interval
                if timeout is not None:
                    remaining = min(remaining, max(0.01, timeout - elapsed))
                out, err = process.communicate(timeout=remaining)
                append_delta(stdout_parts, out or "")
                append_delta(stderr_parts, err or "")
                break
            except subprocess.TimeoutExpired as exc:
                # CPython may provide bytes despite text=True on mocked Popen.
                out = exc.output.decode("utf-8", "replace") if isinstance(exc.output, bytes) else (exc.output or "")
                err = exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                out_delta = append_delta(stdout_parts, out)
                if out_delta:
                    if on_output:
                        for line in out_delta.splitlines():
                            on_output(line)
                err_delta = append_delta(stderr_parts, err)
                if err_delta:
                    if on_output:
                        for line in err_delta.splitlines():
                            on_output(line)
                continue

        stdout, stderr = "".join(stdout_parts), "".join(stderr_parts)
        result = CommandResult(
            command=argv,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed=time.monotonic() - started,
            timed_out=timed_out,
            cancelled=cancelled,
        )
        self.logger.info(
            "Hugin command finished rc=%s elapsed=%.2fs timeout=%s cancelled=%s",
            result.returncode,
            result.elapsed,
            result.timed_out,
            result.cancelled,
        )
        if check and not result.ok:
            reason = "timed out" if result.timed_out else "cancelled" if result.cancelled else f"exit code {result.returncode}"
            raise CommandError(f"Command {reason}: {argv[0]}\n{stderr.strip()}", result)
        return result


CommandRunner = SubprocessRunner
SubprocessExecutor = SubprocessRunner


__all__ = ["CommandError", "CommandResult", "CommandRunner", "SubprocessExecutor", "SubprocessRunner"]

