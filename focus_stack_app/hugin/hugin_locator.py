"""Locate Hugin's command line tools with actionable diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable, Mapping


class HuginToolNotFound(FileNotFoundError):
    """Raised when a requested Hugin executable cannot be found."""

    def __init__(self, tool: str, searched: Iterable[os.PathLike[str] | str] = ()):
        self.tool = tool
        self.searched = tuple(str(p) for p in searched)
        detail = "\n".join(f"  - {item}" for item in self.searched)
        message = (
            f"Hugin tool '{tool}' was not found. Install Hugin or choose its bin directory "
            "in settings."
        )
        if detail:
            message += f"\nSearched:\n{detail}"
        super().__init__(message)


@dataclass(frozen=True)
class HuginInstallation:
    """Resolved Hugin command paths."""

    align_image_stack: Path | None = None
    enfuse: Path | None = None
    root: Path | None = None

    @property
    def available(self) -> bool:
        return self.align_image_stack is not None and self.enfuse is not None

    @property
    def align_path(self) -> Path | None:
        return self.align_image_stack

    @property
    def enfuse_path(self) -> Path | None:
        return self.enfuse

    def as_dict(self) -> dict[str, str | None]:
        return {
            "align_image_stack": str(self.align_image_stack) if self.align_image_stack else None,
            "enfuse": str(self.enfuse) if self.enfuse else None,
            "root": str(self.root) if self.root else None,
        }


class HuginLocator:
    """Find ``align_image_stack`` and ``enfuse``.

    Search order is explicit override, Hugin environment variables, the
    project-bundled runtime, PATH, and the small set of conventional Windows
    install directories.  No recursive drive scan is performed, so a missing
    installation is quick to diagnose.
    """

    TOOL_NAMES: Mapping[str, tuple[str, ...]] = {
        "align_image_stack": ("align_image_stack", "align_image_stack.exe"),
        "enfuse": ("enfuse", "enfuse.exe"),
    }

    def __init__(self, explicit_path: os.PathLike[str] | str | None = None, *, environ: Mapping[str, str] | None = None):
        self.explicit_path = Path(explicit_path) if explicit_path else None
        self.environ = dict(os.environ if environ is None else environ)
        # Passing an environment mapping is also the supported way for
        # callers/tests to request fully controlled discovery without using
        # files from the source checkout.
        self._use_bundled_runtime = environ is None
        self._searched: list[Path] = []

    @staticmethod
    def bundled_bin() -> Path:
        """Return the project-local Hugin runtime directory."""

        return Path(__file__).resolve().parents[2] / ".runtime-deps" / "hugin" / "bin"

    def _candidate_files(self, tool: str) -> list[Path]:
        if tool not in self.TOOL_NAMES:
            raise ValueError(f"Unsupported Hugin tool: {tool}")
        names = self.TOOL_NAMES[tool]
        candidates: list[Path] = []

        def add(path: Path) -> None:
            path = path.expanduser()
            self._searched.append(path)
            if path not in candidates:
                candidates.append(path)

        explicit = self.explicit_path
        if explicit:
            if explicit.suffix.lower() in {".exe", ".com", ".bat", ".cmd"}:
                # If an explicit executable is supplied, use it for the
                # matching tool only; derive a sibling for the other tool.
                add(explicit if explicit.name.casefold().startswith(tool.casefold()) else explicit.parent / names[-1])
            else:
                for name in names:
                    add(explicit / name)

        for key in ("HUGIN_BIN", "HUGIN_PATH", "HUGIN_HOME"):
            value = self.environ.get(key)
            if not value:
                continue
            base = Path(value)
            if base.suffix.lower() in {".exe", ".com", ".bat", ".cmd"}:
                add(base if base.name.casefold().startswith(tool.casefold()) else base.parent / names[-1])
            else:
                for name in names:
                    add(base / name)
                # HUGIN_HOME usually points one level above bin.
                add(base / "bin" / names[-1])

        if self._use_bundled_runtime:
            bundled = self.bundled_bin()
            for name in names:
                add(bundled / name)

        for name in names:
            found = shutil.which(name)
            if found:
                add(Path(found))

        # Keep this intentionally shallow and deterministic.  Environment
        # variables are read at construction so tests can inject fake roots.
        roots: list[Path] = []
        for key in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            value = self.environ.get(key)
            if value:
                roots.append(Path(value))
        if os.name != "nt":
            roots.extend([Path("/usr/bin"), Path("/usr/local/bin")])
        for root in roots:
            for relative in (
                Path("Hugin") / "bin",
                Path("Hugin"),
                Path("Hugin 2024") / "bin",
                Path("Hugin 2023") / "bin",
            ):
                for name in names:
                    add(root / relative / name)
        return candidates

    def find(self, tool: str) -> Path | None:
        """Return an executable path, or ``None`` when unavailable."""

        for candidate in self._candidate_files(tool):
            try:
                # Windows determines executable type by extension, while POSIX
                # tests commonly use a temporary fake file without chmod +x.
                if candidate.is_file() and (os.name == "nt" or candidate.suffix.casefold() in {".exe", ".bat", ".cmd", ".com"} or os.access(candidate, os.X_OK)):
                    return candidate
            except OSError:
                continue
        return None

    locate = find

    def require(self, tool: str) -> Path:
        path = self.find(tool)
        if path is None:
            raise HuginToolNotFound(tool, self._searched)
        return path

    def find_installation(self, *, require: bool = False) -> HuginInstallation:
        align = self.find("align_image_stack")
        enfuse = self.find("enfuse")
        root = None
        for item in (align, enfuse):
            if item is not None:
                root = item.parent.parent if item.parent.name.casefold() == "bin" else item.parent
                break
        installation = HuginInstallation(align_image_stack=align, enfuse=enfuse, root=root)
        if require and not installation.available:
            missing = "align_image_stack" if align is None else "enfuse"
            raise HuginToolNotFound(missing, self._searched)
        return installation

    # Common aliases used by UI settings code.
    locate_all = find_installation
    diagnose = find_installation


def locate_hugin(explicit_path: os.PathLike[str] | str | None = None, *, require: bool = False) -> HuginInstallation:
    """Functional locator used by settings/CLI integrations."""

    return HuginLocator(explicit_path).find_installation(require=require)


find_hugin = locate_hugin


__all__ = ["HuginInstallation", "HuginLocator", "HuginToolNotFound", "find_hugin", "locate_hugin"]

