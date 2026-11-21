"""Canonical fusion mode names and migration of older option values."""

AI = "ai"
QUALITY = "quality"
EXPERIMENTAL = "hugin_enfuse"

_ALIASES = {
    "standard": QUALITY,
    "opencv": QUALITY,
    "fast": AI,
    "speed": AI,
    "fast_fusion": AI,
    "hugin": EXPERIMENTAL,
    "experimental": EXPERIMENTAL,
    "stackmff": AI,
    "stackmff_v3": AI,
    "ai_fusion": AI,
}


def normalize_fusion_backend(value: str | None) -> str:
    name = str(value or AI).strip().casefold()
    name = _ALIASES.get(name, name)
    if name not in {AI, QUALITY, EXPERIMENTAL}:
        raise ValueError("fusion_backend must be ai, quality or hugin_enfuse")
    return name


__all__ = ["AI", "QUALITY", "EXPERIMENTAL", "normalize_fusion_backend"]
