"""Canonical fusion mode names and migration of older option values."""

QUALITY = "quality"
EXPERIMENTAL = "hugin_enfuse"

_ALIASES = {
    "standard": QUALITY,
    "opencv": QUALITY,
    "fast": QUALITY,
    "hugin": EXPERIMENTAL,
    "experimental": EXPERIMENTAL,
}


def normalize_fusion_backend(value: str | None) -> str:
    name = str(value or QUALITY).strip().casefold()
    name = _ALIASES.get(name, name)
    if name not in {QUALITY, EXPERIMENTAL}:
        raise ValueError("fusion_backend must be quality or hugin_enfuse")
    return name


__all__ = ["QUALITY", "EXPERIMENTAL", "normalize_fusion_backend"]
