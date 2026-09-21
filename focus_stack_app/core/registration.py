"""Bounded-memory, low-resolution image registration helpers.

Registration here is intentionally an analysis aid.  The result is suitable
for putting focus maps into a common coordinate system; final full-resolution
alignment remains the responsibility of Hugin.  Functions accept either an
already decoded image array or a path and resize before feature extraction so
callers never need to keep the original pixels around.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Iterator

from .types import RegistrationResult
from .registration_cache import cached


@dataclass(slots=True)
class RegistrationConfig:
    """Tunable registration settings.

    The object intentionally mirrors the project's duck-typed configuration
    convention: every public function also accepts an arbitrary object with
    matching attributes.  Missing attributes use these defaults.
    """

    analysis_long_edge: int = 1600
    max_features: int = 1600
    ratio_test: float = 0.75
    min_matches: int = 8
    min_inliers: int = 6
    min_inlier_ratio: float = 0.35
    ransac_reprojection_threshold: float = 3.0
    use_affine_fallback: bool = True
    use_ecc_fallback: bool = True
    ecc_min_correlation: float = 0.90
    ecc_refine_features: bool = True
    ecc_iterations: int = 80
    ecc_epsilon: float = 1e-5
    max_translation_fraction: float = 0.25
    min_scale: float = 0.70
    max_scale: float = 1.40


def _setting(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    value = getattr(config, name, None)
    if value is None and name == "analysis_long_edge":
        value = getattr(config, "focus_analysis_long_edge", None)
    if value is None:
        section = getattr(config, "analysis", None)
        if section is not None:
            value = getattr(section, name, None)
            if value is None and name == "analysis_long_edge":
                value = getattr(section, "focus_analysis_long_edge", None)
    return default if value is None else value


def resolve_image_path(value: Any) -> str | PathLike[str] | None:
    """Resolve a path from a Path/string or lightweight image metadata row."""

    if isinstance(value, (str, bytes, PathLike, Path)):
        return value
    if isinstance(value, dict):
        for name in ("current_path", "path", "original_path", "filename"):
            candidate = value.get(name)
            if isinstance(candidate, (str, bytes, PathLike, Path)):
                return candidate
    for name in ("current_path", "path", "original_path", "filename"):
        candidate = getattr(value, name, None)
        if isinstance(candidate, (str, bytes, PathLike, Path)):
            return candidate
    return None


def _np() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "registration requires NumPy; install the project's image "
            "dependencies before running analysis"
        ) from exc
    return np


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in minimal envs
        raise RuntimeError(
            "registration requires OpenCV (opencv-python) for ECC/AKAZE"
        ) from exc
    return cv2


def _as_array(image: Any) -> Any:
    np = _np()
    source_path = resolve_image_path(image)
    if source_path is not None:
        return load_image(source_path)
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("image is empty")
    if arr.ndim not in (2, 3):
        raise ValueError("image must be a 2-D grayscale or 3-D colour array")
    return arr


def load_image(path: str | PathLike[str], max_long_edge: int | None = None,
               grayscale: bool = False) -> Any:
    """Load one image, optionally downsampled before it is returned.

    OpenCV is preferred because it performs a direct JPEG decode.  Pillow is
    a small compatibility fallback for environments that do not ship cv2.
    ``max_long_edge`` is useful for scene previews and registration and keeps
    the expensive full-resolution array out of the caller's long-lived state.
    """

    np = _np()
    resolved_path = resolve_image_path(path)
    if resolved_path is None:
        raise TypeError("path must be a filesystem path or image metadata object")
    path = str(resolved_path)
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(path, flag)
        # ``imdecode(fromfile(...))`` handles non-ASCII Windows paths on
        # OpenCV builds where ``imread`` still uses the narrow Win32 API.
        if image is None:
            try:
                encoded = np.fromfile(path, dtype=np.uint8)
                image = cv2.imdecode(encoded, flag)
            except (OSError, ValueError):
                image = None
        if image is None:
            raise ValueError(f"unable to decode image: {path}")
    else:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "image loading requires OpenCV or Pillow"
            ) from exc
        with Image.open(path) as pil_image:
            pil_image = pil_image.convert("L" if grayscale else "RGB")
            image = np.asarray(pil_image)

    if max_long_edge:
        image = resize_for_analysis(image, int(max_long_edge))
    return image


def resize_for_analysis(image: Any, long_edge: int = 1600) -> Any:
    """Return an aspect-preserving image whose longest edge is ``long_edge``.

    Images already smaller than the requested edge are copied, avoiding an
    accidental upsample.  This function never mutates the input array.
    """

    np = _np()
    arr = _as_array(image) if resolve_image_path(image) is not None else np.asarray(image)
    if arr.size == 0:
        raise ValueError("image is empty")
    if long_edge is None or int(long_edge) <= 0:
        return arr.copy()
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest <= int(long_edge):
        return arr.copy()
    scale = float(long_edge) / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    try:
        cv2 = _cv2()
    except RuntimeError:
        cv2 = None
    if cv2 is not None:
        return cv2.resize(arr, new_size, interpolation=cv2.INTER_AREA)
    # Nearest-neighbour fallback keeps this helper usable with NumPy alone.
    ys = np.minimum((np.arange(new_size[1]) / scale).astype(int), h - 1)
    xs = np.minimum((np.arange(new_size[0]) / scale).astype(int), w - 1)
    return arr[ys[:, None], xs[None, :]].copy()


def _gray_u8(image: Any) -> Any:
    return cached("gray", image, lambda: _gray_u8_uncached(image))


def _gray_u8_uncached(image: Any) -> Any:
    np = _np()
    cv2 = None
    try:
        cv2 = _cv2()
    except RuntimeError:
        pass
    arr = np.asarray(image)
    if arr.ndim == 3:
        if cv2 is not None:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        else:
            # This fallback also handles RGB arrays reasonably well; exact
            # channel ordering is immaterial for correlation/ORB-free paths.
            arr = np.mean(arr[..., :3], axis=2)
    if arr.dtype != np.uint8:
        arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
        if float(arr.max(initial=0.0)) <= 1.0:
            arr *= 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def _normalized_similarity(a: Any, b: Any) -> float:
    """Cheap fallback similarity, robust to exposure shifts."""

    np = _np()
    cv2 = None
    try:
        cv2 = _cv2()
    except RuntimeError:
        pass
    a = _gray_u8(a)
    b = _gray_u8(b)
    if cv2 is not None and a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    elif a.shape != b.shape:
        ys = np.minimum((np.arange(a.shape[0]) * b.shape[0] / a.shape[0]).astype(int), b.shape[0] - 1)
        xs = np.minimum((np.arange(a.shape[1]) * b.shape[1] / a.shape[1]).astype(int), b.shape[1] - 1)
        b = b[ys[:, None], xs[None, :]]
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    af -= af.mean()
    bf -= bf.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
    corr = float(np.dot(af, bf) / denom) if denom > 1e-6 else 1.0
    corr = (corr + 1.0) * 0.5
    mad = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
    return float(np.clip(0.65 * corr + 0.35 * (1.0 - mad), 0.0, 1.0))


def _identity_result(message: str = "identity fallback", confidence: float = 0.0) -> RegistrationResult:
    np = _np()
    return RegistrationResult(
        matrix=np.eye(3, dtype=np.float32),
        model="identity",
        valid=False,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        message=message,
    )


def _result_from_matrix(matrix: Any, model: str, valid_matches: int,
                        inliers: int, inlier_ratio: float,
                        image_shape: tuple[int, int], config: Any,
                        reprojection_error: float | None = None,
                        message: str = "") -> RegistrationResult:
    np = _np()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        h = np.eye(3, dtype=np.float32)
        h[:2] = matrix
        matrix = h
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return _identity_result("invalid transform")
    if abs(float(matrix[2, 2])) > 1e-8:
        matrix = matrix / matrix[2, 2]
    a = matrix[:2, :2]
    # For a homography this is an approximation around the image centre.  It
    # is still useful for sanity checks and telemetry.
    sx = float(np.linalg.norm(a[:, 0]))
    sy = float(np.linalg.norm(a[:, 1]))
    scale = (sx + sy) * 0.5
    rotation = math.degrees(math.atan2(float(a[1, 0]), float(a[0, 0])))
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    h, w = image_shape
    max_translation = float(_setting(config, "max_translation_fraction", 0.25)) * max(h, w)
    min_scale = float(_setting(config, "min_scale", 0.70))
    max_scale = float(_setting(config, "max_scale", 1.40))
    geometry_ok = (
        valid_matches >= int(_setting(config, "min_matches", 8))
        and inliers >= int(_setting(config, "min_inliers", 6))
        and inlier_ratio >= float(_setting(config, "min_inlier_ratio", 0.35))
        and min_scale <= scale <= max_scale
        and abs(tx) <= max_translation
        and abs(ty) <= max_translation
    )
    confidence = 0.65 * float(np.clip(inlier_ratio, 0.0, 1.0))
    confidence += 0.35 * float(np.clip(inliers / 30.0, 0.0, 1.0))
    if not (min_scale <= scale <= max_scale):
        confidence *= 0.35
    if abs(tx) > max_translation or abs(ty) > max_translation:
        confidence *= 0.35
    return RegistrationResult(
        matrix=matrix,
        model=model,
        valid=bool(geometry_ok),
        valid_matches=int(valid_matches),
        inliers=int(inliers),
        inlier_ratio=float(np.clip(inlier_ratio, 0.0, 1.0)),
        scale=float(scale if np.isfinite(scale) else 1.0),
        rotation_degrees=float(rotation if np.isfinite(rotation) else 0.0),
        translation=(tx, ty),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        reprojection_error=reprojection_error,
        message=message,
    )


def _match_akaze(reference: Any, image: Any, config: Any) -> RegistrationResult:
    np = _np()
    cv2 = _cv2()
    ref_gray = _gray_u8(reference)
    img_gray = _gray_u8(image)
    detector = cv2.AKAZE_create()
    kp_ref, des_ref = cached("akaze", ref_gray, lambda: detector.detectAndCompute(ref_gray, None))
    kp_img, des_img = cached("akaze", img_gray, lambda: detector.detectAndCompute(img_gray, None))
    if des_ref is None or des_img is None or len(kp_ref) < 4 or len(kp_img) < 4:
        return _identity_result("not enough AKAZE features")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw = matcher.knnMatch(des_img, des_ref, k=2)
    ratio = float(_setting(config, "ratio_test", 0.75))
    good = [m for pair in raw if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    min_matches = int(_setting(config, "min_matches", 8))
    if len(good) < min_matches:
        return _identity_result(f"only {len(good)} AKAZE matches")
    src = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    threshold = float(_setting(config, "ransac_reprojection_threshold", 3.0))
    matrix, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold
    )
    model = "similarity"
    if (matrix is None or mask is None) and bool(_setting(config, "use_affine_fallback", True)):
        matrix, mask = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold
        )
        model = "affine"
    if matrix is None or mask is None:
        return _identity_result("AKAZE RANSAC could not estimate transform")
    inlier_count = int(np.count_nonzero(mask))
    ratio_value = inlier_count / max(1, len(good))
    error = None
    try:
        projected = cv2.transform(src, matrix)
        error = float(np.mean(np.linalg.norm(projected - dst, axis=2)))
    except Exception:
        pass
    return _result_from_matrix(
        matrix, model, len(good), inlier_count, ratio_value,
        ref_gray.shape[:2], config, reprojection_error=error,
        message=f"AKAZE + {model} RANSAC",
    )


def _match_ecc(reference: Any, image: Any, config: Any,
               initial: RegistrationResult | None = None) -> RegistrationResult:
    """Optional translation/affine ECC fallback for texture-poor ORB input."""

    np = _np()
    cv2 = _cv2()
    ref = cached("ecc_float", reference, lambda: _gray_u8(reference).astype(np.float32) / 255.0)
    img = cached("ecc_float", image, lambda: _gray_u8(image).astype(np.float32) / 255.0)
    if ref.shape != img.shape:
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    if initial is not None and initial.matrix is not None:
        candidate = np.asarray(initial.matrix, dtype=np.float32)
        if candidate.shape == (3, 3):
            candidate = candidate[:2]
        if candidate.shape == (2, 3):
            # Stored transforms are source -> reference; ECC's initial warp
            # uses the inverse mapping expected with WARP_INVERSE_MAP.
            warp = cv2.invertAffineTransform(candidate)
    motion = cv2.MOTION_AFFINE
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(_setting(config, "ecc_iterations", 80)),
        float(_setting(config, "ecc_epsilon", 1e-5)),
    )
    try:
        correlation, warp = cv2.findTransformECC(ref, img, warp, motion, criteria)
    except cv2.error:
        return _identity_result("ECC could not estimate transform")
    # findTransformECC returns the matrix used with WARP_INVERSE_MAP in the
    # OpenCV examples (reference/destination -> source). The rest of this
    # module consistently stores source -> reference transforms.
    warp = cv2.invertAffineTransform(warp)
    result = _result_from_matrix(
        warp, "ecc-affine", int(max(1.0, correlation * 20.0)),
        int(max(1.0, correlation * 20.0)), float(np.clip(correlation, 0.0, 1.0)),
        ref.shape[:2], config, message="ECC affine",
    )
    if correlation < float(_setting(config, "ecc_min_correlation", 0.90)):
        result.valid = False
        result.message = f"ECC correlation below threshold ({correlation:.4f})"
    return result


def register_images(reference: Any, image: Any,
                    config: Any = None) -> RegistrationResult:
    """Register ``image`` into ``reference`` coordinates at analysis scale.

    Affine ECC is attempted first.  When it fails or its correlation is too
    low, AKAZE descriptor matching and similarity/affine RANSAC are used. If neither method can
    establish a safe transform, an identity matrix is returned with
    ``valid=False``; callers can still use it for a best-effort comparison.
    """

    config = config or RegistrationConfig()
    edge = int(_setting(config, "analysis_long_edge", 1600))
    ref = cached(("resize", edge), reference, lambda: resize_for_analysis(_as_array(reference), edge))
    img = cached(("resize", edge), image, lambda: resize_for_analysis(_as_array(image), edge))
    try:
        result = _match_ecc(ref, img, config)
    except RuntimeError as exc:
        # Keep analysis usable in a minimal install (for example while the
        # optional OpenCV wheel is being provisioned).  Scene detection can
        # still use low-frequency evidence, and callers get an explicit
        # invalid/identity result rather than an opaque import crash.
        similarity = _normalized_similarity(ref, img)
        result = _identity_result(
            f"ECC unavailable: {exc}", confidence=similarity * 0.45,
        )
    except ValueError:
        raise
    except Exception as exc:
        result = _identity_result(f"ECC error: {exc}")
    if result.valid:
        return result
    try:
        feature_result = _match_akaze(ref, img, config)
    except Exception as exc:
        feature_result = _identity_result(f"AKAZE error: {exc}")
    if feature_result.valid:
        if bool(_setting(config, "ecc_refine_features", True)):
            refined = _match_ecc(ref, img, config, initial=feature_result)
            if refined.valid and refined.confidence >= feature_result.confidence * 0.8:
                refined.message = feature_result.message + " + ECC refinement"
                return refined
        return feature_result
    similarity = _normalized_similarity(ref, img)
    if similarity >= 0.985:
        feature_result.confidence = max(feature_result.confidence, float(similarity * 0.65))
        feature_result.message = "high pixel similarity; identity transform"
    return feature_result


def register(reference: Any, image: Any, config: Any = None) -> RegistrationResult:
    """Alias retained for concise callers."""

    return register_images(reference, image, config)


def warp_image(image: Any, registration: RegistrationResult | Any,
               output_shape: tuple[int, int] | None = None) -> Any:
    """Warp one analysis image using a registration result.

    ``output_shape`` is ``(height, width)``.  By default the input shape is
    retained.  An invalid/identity result returns a copy of the input.
    """

    np = _np()
    cv2 = _cv2()
    arr = _as_array(image)
    matrix = getattr(registration, "matrix", registration)
    if matrix is None:
        return arr.copy()
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (2, 3):
        hmat = np.eye(3, dtype=np.float32)
        hmat[:2] = matrix
        matrix = hmat
    if matrix.shape != (3, 3):
        raise ValueError("registration matrix must be 3x3 or 2x3")
    h, w = arr.shape[:2]
    oh, ow = output_shape or (h, w)
    return cv2.warpPerspective(arr, matrix, (int(ow), int(oh)), flags=cv2.INTER_LINEAR)


def register_sequence(images: Iterable[Any], config: Any = None) -> Iterator[tuple[Any, RegistrationResult]]:
    """Yield ``(image, result)`` while retaining only the reference image.

    This helper is intentionally a generator: callers can read, register,
    compute a focus map, and release each array before advancing the iterator.
    The first image is the reference and receives an identity result.
    """

    iterator = iter(images)
    try:
        reference = next(iterator)
    except StopIteration:
        return
    config = config or RegistrationConfig()
    ref_analysis = resize_for_analysis(_as_array(reference), int(_setting(config, "analysis_long_edge", 1600)))
    yield reference, RegistrationResult(
        matrix=_np().eye(3, dtype=_np().float32), model="identity", valid=True,
        valid_matches=0, inliers=0, inlier_ratio=1.0, confidence=1.0,
        message="reference image",
    )
    for image in iterator:
        yield image, register_images(ref_analysis, image, config)


estimate_registration = register_images
RegistrationSettings = RegistrationConfig

__all__ = [
    "RegistrationConfig", "RegistrationSettings", "RegistrationResult", "load_image",
    "resolve_image_path", "resize_for_analysis", "register_images", "register",
    "estimate_registration", "warp_image", "register_sequence",
]
