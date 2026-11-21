"""Bounded image I/O helpers.

Metadata functions in this module only read JPEG headers.  Pixel-loading
functions are explicit and decode one image at a time, optionally resizing it
before returning.  Keeping these paths separate makes it difficult for a
directory scanner to accidentally materialize a whole shoot in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import threading
from pathlib import Path
from typing import Any


class ImageIOError(RuntimeError):
    """Base error for optional image-decoding operations."""


_QUALITY_JPEG_READ_SEMAPHORE = threading.BoundedSemaphore(1)


def load_quality_rgb(path: str | Path, *, pass_index: int | None = None) -> Any:
    """Read one whole JPEG under the process-wide disk limit; decode outside it."""
    import cv2
    import numpy as np
    from .performance import stage

    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return load_rgb(path)
    with stage("jpeg_read_wait", path=str(path), pass_index=pass_index):
        _QUALITY_JPEG_READ_SEMAPHORE.acquire()
    try:
        with stage("jpeg_read", path=str(path), bytes=path.stat().st_size,
                   read_concurrency_limit=1, pass_index=pass_index):
            encoded = path.read_bytes()
    finally:
        _QUALITY_JPEG_READ_SEMAPHORE.release()
    with stage("image_decode", path=str(path), decoder="opencv_imdecode", pass_index=pass_index):
        bgr = cv2.imdecode(np.frombuffer(encoded, np.uint8),
                           cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        del encoded
        if bgr is None:
            raise ImageIOError(f"unable to decode JPEG {path}")
        try:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=bgr)
        except TypeError:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int


# Baseline/progressive/lossless JPEG SOF markers carrying width and height.
_SOF_MARKERS = {
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
}


def _parse_jpeg_size_from_buffer(data: bytes) -> ImageSize | None:
    """Fast in-memory scan of a JPEG header chunk for an SOF marker."""
    length_data = len(data)
    if length_data < 4 or data[:2] != b"\xff\xd8":
        return None
    pos = 2
    while pos < length_data:
        pos = data.find(b"\xff", pos)
        if pos < 0 or pos + 1 >= length_data:
            return None
        while pos + 1 < length_data and data[pos + 1] == 0xFF:
            pos += 1
        if pos + 1 >= length_data:
            return None
        marker = data[pos + 1]
        pos += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            if marker == 0xD9:
                return None
            continue
        if pos + 2 > length_data:
            return None
        seg_len = int.from_bytes(data[pos : pos + 2], "big")
        if seg_len < 2 or marker == 0xDA:
            return None
        if marker in _SOF_MARKERS:
            if pos + 7 > length_data:
                return None
            height = int.from_bytes(data[pos + 3 : pos + 5], "big")
            width = int.from_bytes(data[pos + 5 : pos + 7], "big")
            if width and height:
                return ImageSize(width=width, height=height)
        pos += seg_len
    return None


def read_jpeg_size(path: str | Path) -> ImageSize:
    """Read dimensions from a JPEG SOF marker without decoding pixels."""

    path = Path(path)
    try:
        with path.open("rb") as stream:
            head = stream.read(65536)
            if len(head) < 2 or head[:2] != b"\xff\xd8":
                raise ImageIOError(f"not a JPEG file: {path}")
            parsed = _parse_jpeg_size_from_buffer(head)
            if parsed is not None:
                return parsed
            # Fall back to seeking across segments > 64 KiB (e.g. huge ICC/XMP).
            stream.seek(2)
            while True:
                byte = stream.read(1)
                if not byte:
                    break
                if byte != b"\xff":
                    continue
                marker_byte = stream.read(1)
                while marker_byte == b"\xff":
                    marker_byte = stream.read(1)
                if not marker_byte:
                    break
                marker = marker_byte[0]
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    if marker == 0xD9:
                        break
                    continue
                length_bytes = stream.read(2)
                if len(length_bytes) != 2:
                    break
                length = int.from_bytes(length_bytes, "big")
                if length < 2:
                    break
                if marker == 0xDA:
                    break
                if marker in _SOF_MARKERS:
                    sof = stream.read(min(length - 2, 5))
                    if len(sof) >= 5:
                        height = int.from_bytes(sof[1:3], "big")
                        width = int.from_bytes(sof[3:5], "big")
                        if width and height:
                            return ImageSize(width=width, height=height)
                    stream.seek(max(length - 2 - len(sof), 0), os.SEEK_CUR)
                else:
                    stream.seek(length - 2, os.SEEK_CUR)
    except OSError as exc:
        raise ImageIOError(f"unable to read JPEG header {path}: {exc}") from exc
    raise ImageIOError(f"JPEG dimensions not found: {path}")


def read_image_size(path: str | Path) -> ImageSize:
    """Read dimensions; currently V1 accepts only JPEG/JPEG-compatible files."""

    suffix = Path(path).suffix.lower()
    if suffix not in {".jpg", ".jpeg"}:
        raise ImageIOError(f"unsupported image extension: {suffix or '<none>'}")
    return read_jpeg_size(path)


def image_fingerprint(path: str | Path) -> str:
    """Return a cheap cache key based on canonical path and file stat.

    Reading file contents to hash a 15 MB JPEG for every cache lookup would be
    needlessly expensive.  Size + nanosecond mtime detects normal edits while
    keeping scanning O(number of files).
    """

    path = Path(path)
    try:
        stat = path.stat()
    except OSError as exc:
        raise ImageIOError(f"unable to stat image {path}: {exc}") from exc
    payload = "\0".join(
        (
            os.path.normcase(os.path.abspath(os.fspath(path))),
            str(stat.st_size),
            str(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        )
    ).encode("utf-8", errors="surrogatepass")
    return hashlib.sha256(payload).hexdigest()


def _resize_dimensions(width: int, height: int, long_edge: int | None) -> tuple[int, int]:
    if long_edge is None or long_edge <= 0 or max(width, height) <= long_edge:
        return width, height
    scale = long_edge / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _opencv_reduced_mode(cv2: Any, path: Path, long_edge: int | None, grayscale: bool) -> int:
    if long_edge and long_edge > 0 and path.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            size = read_jpeg_size(path)
            ratio = max(size.width, size.height) / long_edge
            if ratio >= 8:
                return cv2.IMREAD_REDUCED_GRAYSCALE_8 if grayscale else cv2.IMREAD_REDUCED_COLOR_8
            if ratio >= 4:
                return cv2.IMREAD_REDUCED_GRAYSCALE_4 if grayscale else cv2.IMREAD_REDUCED_COLOR_4
            if ratio >= 2:
                return cv2.IMREAD_REDUCED_GRAYSCALE_2 if grayscale else cv2.IMREAD_REDUCED_COLOR_2
        except ImageIOError:
            pass
    return cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR


def load_image(
    path: str | Path,
    *,
    long_edge: int | None = None,
    grayscale: bool = False,
    max_long_edge: int | None = None,
) -> Any:
    """Decode one image, resizing it immediately when requested.

    OpenCV is preferred because the rest of the application uses NumPy/OpenCV
    conventions (BGR arrays).  Pillow is a fallback and returns an RGB or
    grayscale NumPy array.  Neither dependency is imported during metadata
    scanning.
    """

    if long_edge is not None and max_long_edge is not None and long_edge != max_long_edge:
        raise ValueError("long_edge and max_long_edge disagree")
    if long_edge is None:
        long_edge = max_long_edge
    path = Path(path)
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        cv2 = None
    if cv2 is not None:
        mode = _opencv_reduced_mode(cv2, path, long_edge, grayscale)
        image = cv2.imread(str(path), mode)
        if image is None:
            try:
                import numpy as np
                encoded = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(encoded, mode)
            except (OSError, ValueError):
                image = None
        if image is None:
            raise ImageIOError(f"unable to decode image: {path}")
        if long_edge and max(image.shape[:2]) > long_edge:
            height, width = image.shape[:2]
            target_width, target_height = _resize_dimensions(width, height, long_edge)
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return image

    try:
        from PIL import Image  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImageIOError(
            "preview decoding requires opencv-python or Pillow + NumPy; "
            "metadata scanning does not"
        ) from exc
    try:
        with Image.open(path) as image:
            target_mode = "L" if grayscale else "RGB"
            if long_edge:
                image.draft(target_mode, (long_edge, long_edge))
            if image.mode != target_mode:
                image = image.convert(target_mode)
            if long_edge and max(image.size) > long_edge:
                target = _resize_dimensions(image.width, image.height, long_edge)
                image = image.resize(target, Image.Resampling.LANCZOS)
            return np.asarray(image)
    except Exception as exc:
        raise ImageIOError(f"unable to decode image {path}: {exc}") from exc


def load_preview(
    path: str | Path,
    long_edge: int | None = None,
    *,
    grayscale: bool = False,
    max_long_edge: int | None = None,
) -> Any:
    if long_edge is None and max_long_edge is None:
        long_edge = 512
    return load_image(path, long_edge=long_edge, grayscale=grayscale, max_long_edge=max_long_edge)


def load_analysis_image(
    path: str | Path,
    long_edge: int | None = None,
    *,
    grayscale: bool = False,
    max_long_edge: int | None = None,
) -> Any:
    if long_edge is None and max_long_edge is None:
        long_edge = 1600
    return load_image(path, long_edge=long_edge, grayscale=grayscale, max_long_edge=max_long_edge)


def load_rgb(path: str | Path, long_edge: int | None = None) -> Any:
    """Load RGB pixels through a shared, non-domain-specific I/O API."""
    try:
        from PIL import Image  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        image = load_image(path, long_edge=long_edge)
        try:
            import cv2  # type: ignore[import-not-found]
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except ImportError:
            return image
    from .performance import stage
    from io import BytesIO
    # Preserve Pillow's decoder and RGB conversion while separating JPEG
    # sequential read from decode for the baseline. Other formats retain
    # their existing file reader, especially large aligned TIFFs.
    source = path
    if long_edge is None and Path(path).suffix.lower() in {".jpg", ".jpeg"}:
        with stage("jpeg_read", path=str(path), bytes=Path(path).stat().st_size):
            source = BytesIO(Path(path).read_bytes())
    with stage("tiff_read_decode" if Path(path).suffix.lower() in {".tif", ".tiff"} else "image_decode",
               path=str(path)), Image.open(source) as image:
        if long_edge and long_edge <= 320:
            image.draft("RGB", (long_edge, long_edge))
        if image.mode != "RGB":
            image = image.convert("RGB")
        if long_edge:
            # convert("RGB") previously forced decode here. Keep thumbnail()
            # from implicitly selecting a different JPEG DCT scale.
            image.load()
            image.thumbnail((long_edge, long_edge))
        return np.asarray(image)


def load_rgb_with_previews(
    path: str | Path,
    long_edges: tuple[int, ...] = (1280, 640),
) -> tuple[Any, dict[int, Any]]:
    """Decode RGB once and make each preview independently from the source.

    Avoids redundant full-resolution copies while matching Pillow's Lanczos
    downsampling behavior.
    """
    try:
        from PIL import Image  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImageIOError("RGB preview decoding requires Pillow + NumPy") from exc

    from .performance import stage
    with stage("tiff_validation_read_decode", path=str(path)), Image.open(path) as image:
        source = image if image.mode == "RGB" else image.convert("RGB")
        full_resolution = np.asarray(source)
        previews: dict[int, Any] = {}
        width, height = source.size
        for edge in dict.fromkeys(int(value) for value in long_edges):
            if edge and max(width, height) > edge:
                target = _resize_dimensions(width, height, edge)
                preview = source.resize(target, Image.Resampling.LANCZOS, reducing_gap=2.0)
                previews[edge] = np.asarray(preview)
            else:
                previews[edge] = full_resolution.copy()
        return full_resolution, previews


# Compatibility aliases for worker code and small scripts.
get_image_size = read_image_size
get_jpeg_size = read_jpeg_size
fingerprint = image_fingerprint

__all__ = [
    "ImageIOError",
    "ImageSize",
    "read_jpeg_size",
    "read_image_size",
    "get_image_size",
    "get_jpeg_size",
    "image_fingerprint",
    "fingerprint",
    "load_image",
    "load_preview",
    "load_analysis_image",
    "load_rgb",
    "load_rgb_with_previews",
]
