"""Preserved pre-refactor whole-frame renderer; opt-in comparison interface.

Keep the preview-scale decisions (focus_score + sigma 8), affine inverse
warps and sigma-6 feathering independent of the newer fusion backends.
The application still defaults to the refactored Hugin/Enfuse pipeline.
"""
from pathlib import Path
import threading

import cv2
import numpy as np

from ..utils.image_io import load_rgb
from ..core.whole_frame_selection import focus_score, warp_preview


def render_plan(plan, destination, *, cancel_event=None, progress=None):
    """Original plan interface: paths, reference_index, selected_indices,
    preview_shape and matrices (2x3 reference-to-source inverse warps).
    """
    from PIL import Image
    cancel_event = cancel_event or threading.Event()
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    paths = plan["paths"]
    reference = load_rgb(paths[plan["reference_index"]])
    height, width = reference.shape[:2]
    ph, pw = plan["preview_shape"]
    preview = load_rgb(paths[plan["reference_index"]], max(ph, pw))
    selected = plan["selected_indices"]
    maps = []
    for index in selected:
        if cancel_event.is_set():
            raise RuntimeError("fusion cancelled")
        rgb = load_rgb(paths[index], max(ph, pw))
        aligned = warp_preview(rgb, plan["matrices"][index], preview.shape)
        maps.append(cv2.GaussianBlur(focus_score(aligned), (0, 0), 8.0))
    labels = np.argmax(np.stack(maps), axis=0).astype(np.float32)
    labels = cv2.medianBlur(labels, 5)
    numerator = np.zeros((height, width, 3), np.float32)
    denominator = np.zeros((height, width), np.float32)
    for local, index in enumerate(selected):
        if cancel_event.is_set():
            raise RuntimeError("fusion cancelled")
        region = cv2.GaussianBlur((labels == local).astype(np.float32), (0, 0), 6.0)
        weight = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
        matrix = np.array(plan["matrices"][index], np.float32)
        matrix[0, 1] *= (width / pw) / (height / ph)
        matrix[1, 0] *= (height / ph) / (width / pw)
        matrix[0, 2] *= width / pw
        matrix[1, 2] *= height / ph
        aligned = warp_preview(load_rgb(paths[index]), matrix, reference.shape)
        numerator += aligned.astype(np.float32) * weight[..., None]
        denominator += weight
        if progress:
            progress(local + 1, len(selected))
    result = np.clip(np.rint(numerator / np.maximum(denominator[..., None], 1e-6)), 0, 255).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if cancel_event.is_set():
        raise RuntimeError("fusion cancelled")
    with Image.open(paths[plan["reference_index"]]) as source:
        metadata = {key: source.info[key] for key in ("icc_profile",) if key in source.info}
    if destination.suffix.lower() in {".tif", ".tiff"}:
        Image.fromarray(result).save(destination, compression="tiff_deflate", **metadata)
    else:
        Image.fromarray(result).save(destination, quality=100, subsampling=0, **metadata)
    return destination
