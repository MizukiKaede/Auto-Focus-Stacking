"""Read a small, safe subset of JPEG EXIF without decoding pixels.

Pillow is used when available because it handles vendor-specific EXIF details
well.  A dependency-free TIFF/IFD fallback keeps metadata scanning usable in a
fresh Python installation and is intentionally bounded to JPEG APP1 segments
(at most 64 KiB each).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import struct
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExifMetadata:
    """Metadata relevant to grouping and display."""

    capture_time: str | None = None
    camera: str | None = None
    lens: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def capture_datetime(self) -> datetime | None:
        if not self.capture_time:
            return None
        try:
            return datetime.fromisoformat(self.capture_time)
        except ValueError:
            return None


def normalize_capture_time(value: Any) -> str | None:
    """Normalize common EXIF date forms to an ISO-like sortable string."""

    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    text = str(value).strip().strip("\x00")
    if not text:
        return None
    # EXIF normally uses ``YYYY:MM:DD HH:MM:SS``.  Preserve sub-second and
    # timezone suffixes if present while making the date lexically sortable.
    if len(text) >= 19 and text[4] == ":" and text[7] == ":":
        text = f"{text[:4]}-{text[5:7]}-{text[8:10]}{text[10:]}"
    return text


def _read_jpeg_app1(path: str | Path) -> bytes | None:
    """Return the first Exif APP1 payload, never reading image pixel data."""

    try:
        with Path(path).open("rb") as stream:
            if stream.read(2) != b"\xff\xd8":
                return None
            while True:
                byte = stream.read(1)
                if not byte:
                    return None
                if byte != b"\xff":
                    continue
                marker_byte = stream.read(1)
                while marker_byte == b"\xff":
                    marker_byte = stream.read(1)
                if not marker_byte:
                    return None
                marker = marker_byte[0]
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    if marker == 0xD9:
                        return None
                    continue
                raw_length = stream.read(2)
                if len(raw_length) != 2:
                    return None
                length = int.from_bytes(raw_length, "big")
                if length < 2:
                    return None
                payload = stream.read(length - 2)
                if len(payload) != length - 2:
                    return None
                if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
                    return payload[6:]
                if marker == 0xDA:  # compressed scan: no more APP metadata
                    return None
    except (OSError, ValueError):
        return None


def _read_scalar(data: bytes, offset: int, type_id: int, count: int, endian: str) -> Any:
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
    size = sizes.get(type_id)
    if size is None or count < 0:
        return None
    total = size * count
    if offset < 0 or offset + total > len(data):
        return None
    chunk = data[offset : offset + total]
    fmt = f"{endian}{count}"
    try:
        if type_id == 1:  # BYTE
            return chunk[0] if count == 1 else list(chunk)
        if type_id == 2:  # ASCII
            return chunk.rstrip(b"\x00").decode("utf-8", errors="replace")
        if type_id == 3:  # SHORT
            values = struct.unpack(fmt + "H", chunk) if count == 1 else struct.unpack(fmt + "H" * count, chunk)
            return values[0] if count == 1 else list(values)
        if type_id == 4:  # LONG
            values = struct.unpack(fmt + "I", chunk) if count == 1 else struct.unpack(fmt + "I" * count, chunk)
            return values[0] if count == 1 else list(values)
        if type_id == 5:  # RATIONAL, retain a useful float representation
            values: list[float] = []
            for index in range(count):
                numerator, denominator = struct.unpack_from(endian + "II", chunk, index * 8)
                values.append(numerator / denominator if denominator else 0.0)
            return values[0] if count == 1 else values
        if type_id == 7:  # UNDEFINED
            return chunk
        if type_id == 9:  # SLONG
            values = struct.unpack(fmt + "i", chunk) if count == 1 else struct.unpack(fmt + "i" * count, chunk)
            return values[0] if count == 1 else list(values)
        if type_id == 10:  # SRATIONAL
            values = []
            for index in range(count):
                numerator, denominator = struct.unpack_from(endian + "ii", chunk, index * 8)
                values.append(numerator / denominator if denominator else 0.0)
            return values[0] if count == 1 else values
    except (struct.error, UnicodeError):
        return None
    return None


def _parse_ifd(data: bytes, tiff_start: int, ifd_offset: int, endian: str) -> dict[int, Any]:
    absolute = tiff_start + ifd_offset
    if absolute < 0 or absolute + 2 > len(data):
        return {}
    try:
        count = struct.unpack_from(endian + "H", data, absolute)[0]
    except struct.error:
        return {}
    # A malformed file could advertise a huge count.  APP1 is tiny, and this
    # bound prevents pathological allocations/loops while retaining normal
    # EXIF (hundreds of entries at most).
    count = min(count, 4096)
    result: dict[int, Any] = {}
    for index in range(count):
        entry = absolute + 2 + index * 12
        if entry + 12 > len(data):
            break
        try:
            tag, type_id, value_count = struct.unpack_from(endian + "HHI", data, entry)
        except struct.error:
            continue
        size = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}.get(type_id)
        if size is None:
            continue
        total = size * value_count
        value_offset = entry + 8 if total <= 4 else _safe_u32(data, entry + 8, endian)
        if value_offset is None:
            continue
        result[tag] = _read_scalar(data, value_offset, type_id, value_count, endian)
    return result


def _safe_u32(data: bytes, offset: int, endian: str) -> int | None:
    try:
        return struct.unpack_from(endian + "I", data, offset)[0]
    except struct.error:
        return None


def parse_exif_tiff(data: bytes) -> dict[int, Any]:
    """Parse an Exif TIFF payload into a flat tag/value mapping."""

    if len(data) < 8:
        return {}
    byte_order = data[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return {}
    try:
        if struct.unpack_from(endian + "H", data, 2)[0] != 42:
            return {}
        first_ifd = struct.unpack_from(endian + "I", data, 4)[0]
    except struct.error:
        return {}
    tags = _parse_ifd(data, 0, first_ifd, endian)
    exif_ifd = tags.get(0x8769)
    if isinstance(exif_ifd, int):
        tags.update(_parse_ifd(data, 0, exif_ifd, endian))
    gps_ifd = tags.get(0x8825)
    if isinstance(gps_ifd, int):
        tags.update({0x8825: _parse_ifd(data, 0, gps_ifd, endian)})
    return tags


def _from_pillow(path: str | Path) -> ExifMetadata | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return ExifMetadata()
            raw = {str(tag): value for tag, value in exif.items()}
            capture = next((exif.get(tag) for tag in (36867, 36868, 306) if exif.get(tag)), None)
            camera = exif.get(272) or exif.get(271)
            lens = exif.get(42036) or exif.get(0xA434)
            return ExifMetadata(
                capture_time=normalize_capture_time(capture),
                camera=str(camera) if camera else None,
                lens=str(lens) if lens else None,
                raw=raw,
            )
    except Exception:
        # Metadata must not prevent the rest of a directory from scanning.
        return ExifMetadata()


def read_exif(path: str | Path) -> ExifMetadata:
    """Read capture date, camera and lens; malformed/missing EXIF is benign."""

    pillow_result = _from_pillow(path)
    if pillow_result is not None and (pillow_result.raw or pillow_result.capture_time):
        return pillow_result
    payload = _read_jpeg_app1(path)
    tags = parse_exif_tiff(payload or b"")
    capture = next((tags.get(tag) for tag in (0x9003, 0x9004, 0x0132) if tags.get(tag)), None)
    camera = tags.get(0x0110) or tags.get(0x010F)
    lens = tags.get(0xA434) or tags.get(0xA433)
    return ExifMetadata(
        capture_time=normalize_capture_time(capture),
        camera=str(camera) if camera else None,
        lens=str(lens) if lens else None,
        raw={str(tag): value for tag, value in tags.items()},
    )


# Friendly alias used by scanner callers.
read_exif_metadata = read_exif


def get_capture_time(path: str | Path) -> str | None:
    """Convenience accessor used by metadata-only sorters."""

    return read_exif(path).capture_time

__all__ = [
    "ExifMetadata",
    "normalize_capture_time",
    "parse_exif_tiff",
    "read_exif",
    "read_exif_metadata",
    "get_capture_time",
]

