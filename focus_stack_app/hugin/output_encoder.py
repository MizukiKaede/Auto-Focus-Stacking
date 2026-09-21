"""Final output naming and loss-aware JPEG/TIFF encoding."""

from __future__ import annotations

from ..utils.performance import timed

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


class OutputFormat(str, Enum):
    JPG = "jpg"
    TIFF = "tiff"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value: str | "OutputFormat") -> "OutputFormat":
        if isinstance(value, cls):
            return value
        value = str(value).casefold().lstrip(".")
        if value in {"jpg", "jpeg"}:
            return cls.JPG
        if value in {"tif", "tiff"}:
            return cls.TIFF
        raise ValueError(f"Unsupported output format: {value}")


@dataclass(frozen=True)
class OutputConfig:
    format: OutputFormat | str = OutputFormat.JPG
    jpeg_quality: int = 100
    jpeg_subsampling: int | str = 0
    optimize: bool = True
    output_suffix: str = "_stack"
    overwrite: bool = False
    tiff_compression: str = "tiff_deflate"

    def __post_init__(self) -> None:
        fmt = OutputFormat.parse(self.format)
        object.__setattr__(self, "format", fmt)
        if not 1 <= int(self.jpeg_quality) <= 100:
            raise ValueError("JPEG quality must be between 1 and 100")


class OutputEncodingError(RuntimeError):
    """Output encoding failed or the optional image backend is unavailable."""


class OutputCollisionError(FileExistsError):
    """Raised when publishing would overwrite an existing file."""


def output_path_for(
    first_original: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    output_format: OutputFormat | str = OutputFormat.JPG,
    *,
    suffix: str | None = None,
) -> Path:
    """Return the canonical output name for a stack's first original.

    JPEG uses ``<stem>_stack.jpg`` by default so the original ``.JPG`` can
    remain beside it.  TIFF follows the V1 plan and uses ``<stem>.tif``.
    """

    fmt = OutputFormat.parse(output_format)
    original = Path(first_original)
    if suffix is None:
        suffix = "_stack" if fmt is OutputFormat.JPG else ""
    extension = ".jpg" if fmt is OutputFormat.JPG else ".tif"
    return Path(output_dir) / f"{original.stem}{suffix}{extension}"


def _same_path(left: Path, right: Path) -> bool:
    try:
        if left.exists() and right.exists() and os.path.samefile(left, right):
            return True
    except OSError:
        pass
    return os.path.abspath(os.fspath(left)).casefold() == os.path.abspath(os.fspath(right)).casefold()


def _atomic_copy(source: Path, destination: Path) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temp = Path(name)
    try:
        shutil.copy2(source, temp)
        if destination.exists():
            raise OutputCollisionError(f"Refusing to overwrite existing output: {destination}")
        os.replace(temp, destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _load_pillow() -> Any:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise OutputEncodingError(
            "Pillow is required to convert a TIFF/PNG Enfuse result to JPEG. "
            "Install Pillow or ask Enfuse to produce JPEG directly."
        ) from exc
    return Image


def _save_encoded_image(image: Any, destination: Path, cfg: OutputConfig) -> None:
    """Share identical format/quality/metadata handling for both input APIs."""
    fmt = OutputFormat.parse(cfg.format)
    exif = image.info.get("exif")
    icc_profile = image.info.get("icc_profile")
    converted = None
    try:
        if fmt is OutputFormat.JPG:
            if image.mode not in {"RGB", "L", "CMYK"}:
                converted = image.convert("RGB")
                image = converted
            kwargs: dict[str, Any] = {
                "format": "JPEG", "quality": int(cfg.jpeg_quality),
                "subsampling": getattr(cfg, "jpeg_subsampling", 0),
                "optimize": bool(getattr(cfg, "optimize", True)),
            }
            if exif:
                kwargs["exif"] = exif
        else:
            kwargs = {"format": "TIFF"}
            compression = getattr(cfg, "tiff_compression", "tiff_deflate")
            if compression:
                kwargs["compression"] = compression
        if icc_profile:
            kwargs["icc_profile"] = icc_profile
        image.save(destination, **kwargs)
    finally:
        if converted is not None:
            converted.close()


@timed("encoding")
def encode_image_output(
    image: Any,
    destination: os.PathLike[str] | str,
    *,
    config: OutputConfig | None = None,
    original_path: os.PathLike[str] | str | None = None,
) -> Path:
    """Encode an in-memory Pillow image without an intermediate TIFF.

    Validate the completed temporary JPEG/TIFF before atomic publication;
    errors leave an existing destination intact and remove partial files.
    The caller retains ownership of the supplied image.
    """
    cfg = config or OutputConfig()
    dst = Path(destination)
    if original_path is not None and _same_path(Path(original_path), dst):
        raise OutputCollisionError(f"Output cannot overwrite original image: {dst}")
    if dst.exists() and not cfg.overwrite:
        raise OutputCollisionError(f"Refusing to overwrite existing output: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fmt = OutputFormat.parse(cfg.format)
    Image = _load_pillow()
    fd, name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent)
    os.close(fd)
    temp = Path(name)
    try:
        _save_encoded_image(image, temp, cfg)
        with Image.open(temp) as check:
            if check.size != image.size or check.format != ("JPEG" if fmt is OutputFormat.JPG else "TIFF"):
                raise OutputEncodingError("Encoded image dimensions or format are invalid")
            check.verify()
        with Image.open(temp) as check:
            check.load()
        if cfg.overwrite:
            os.replace(temp, dst)
        elif os.name == "nt":
            # Windows rename refuses an existing destination, including one
            # another worker published after the initial existence check.
            os.rename(temp, dst)
        else:
            os.link(temp, dst)
            temp.unlink()
    except Exception as exc:
        temp.unlink(missing_ok=True)
        if isinstance(exc, FileExistsError):
            raise OutputCollisionError(f"Refusing to overwrite existing output: {dst}") from exc
        if isinstance(exc, (OutputCollisionError, OutputEncodingError)):
            raise
        raise OutputEncodingError(f"Unable to encode image as {fmt.value}: {exc}") from exc
    return dst


@timed("encoding")
def encode_output(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    config: OutputConfig | None = None,
    original_path: os.PathLike[str] | str | None = None,
) -> Path:
    """Encode *source* and atomically publish *destination*.

    Existing files are rejected unless ``config.overwrite`` is explicitly
    enabled.  Even then, an ``original_path`` (or a source path equal to the
    destination) can never be overwritten.
    """

    cfg = config or OutputConfig()
    src, dst = Path(source), Path(destination)
    if not src.is_file():
        raise FileNotFoundError(f"Encoding source does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _same_path(src, dst):
        raise OutputCollisionError(f"Output cannot overwrite source image: {dst}")
    if original_path is not None and _same_path(Path(original_path), dst):
        raise OutputCollisionError(f"Output cannot overwrite original image: {dst}")
    if dst.exists() and not cfg.overwrite:
        raise OutputCollisionError(f"Refusing to overwrite existing output: {dst}")

    # If the source is already in the requested format, preserving the
    # Enfuse-produced bytes is preferable to a needless decode/re-encode.  In
    # particular this fallback lets headless installations without Pillow use
    # an Enfuse ``--compression=100`` JPEG result.
    fmt = OutputFormat.parse(cfg.format)
    wanted_exts = {".jpg", ".jpeg"} if fmt is OutputFormat.JPG else {".tif", ".tiff"}
    if src.suffix.casefold() in wanted_exts:
        if dst.exists() and cfg.overwrite:
            dst.unlink()
        _atomic_copy(src, dst)
        return dst

    Image = _load_pillow()
    fd, name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".partial", dir=dst.parent)
    os.close(fd)
    temp = Path(name)
    try:
        with Image.open(src) as image:
            _save_encoded_image(image, temp, cfg)
        if dst.exists() and not cfg.overwrite:
            raise OutputCollisionError(f"Refusing to overwrite existing output: {dst}")
        os.replace(temp, dst)
    except Exception as exc:
        temp.unlink(missing_ok=True)
        if isinstance(exc, (OutputCollisionError, OutputEncodingError)):
            raise
        raise OutputEncodingError(f"Unable to encode {src} as {fmt.value}: {exc}") from exc
    return dst


encode = encode_output


def encode_jpeg(
    source: os.PathLike[str] | str,
    destination: os.PathLike[str] | str,
    *,
    quality: int = 100,
    subsampling: int | str = 0,
    optimize: bool = True,
    overwrite: bool = False,
    original_path: os.PathLike[str] | str | None = None,
) -> Path:
    """Explicit JPEG convenience API retaining V1's quality defaults."""

    return encode_output(
        source,
        destination,
        config=OutputConfig(
            format=OutputFormat.JPG,
            jpeg_quality=quality,
            jpeg_subsampling=subsampling,
            optimize=optimize,
            overwrite=overwrite,
        ),
        original_path=original_path,
    )


__all__ = [
    "OutputCollisionError",
    "OutputConfig",
    "OutputEncodingError",
    "OutputFormat",
    "encode",
    "encode_jpeg",
    "encode_output",
    "encode_image_output",
    "output_path_for",
]
