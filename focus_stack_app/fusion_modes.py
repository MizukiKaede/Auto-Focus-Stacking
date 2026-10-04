"""Canonical fusion mode names and migration of older option values."""

QUALITY = "quality"
FAST = "fast"
EXPERIMENTAL = "hugin_enfuse"

_ALIASES = {
    "standard": QUALITY,
    "opencv": QUALITY,
    "speed": FAST,
    "fast_fusion": FAST,
    "hugin": EXPERIMENTAL,
    "experimental": EXPERIMENTAL,
}


def normalize_fusion_backend(value: str | None) -> str:
    name = str(value or QUALITY).strip().casefold()
    name = _ALIASES.get(name, name)
    if name not in {QUALITY, FAST, EXPERIMENTAL}:
        raise ValueError("fusion_backend must be quality, fast or hugin_enfuse")
    return name


__all__ = ["QUALITY", "FAST", "EXPERIMENTAL", "normalize_fusion_backend"]
