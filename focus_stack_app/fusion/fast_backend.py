"""Independent Fast backend; existing backends remain unchanged."""
from pathlib import Path
from typing import Mapping
from .backends import FusionBackend, FusionResult, _value
from ..hugin.output_encoder import encode_image_output, OutputCollisionError, _same_path
from ..utils.performance import profiled_fusion, stage, diagnostic

class FastFusionBackend(FusionBackend):
    """Fast proxy ownership with one full-resolution C++ render pass."""

    name = "fast"

    def __init__(self, *, runtime_config=None):
        self.runtime_config = (runtime_config.get("runtime", runtime_config)
                               if isinstance(runtime_config, Mapping)
                               else getattr(runtime_config, "runtime", runtime_config))

    @profiled_fusion
    def fuse(self, group, analysis, output_path, work_dir, output_config, cancel_event):
        import numpy as np
        from PIL import Image
        from .fast_fusion import compute_proxy_depth_map, render_fast_stream, VERSION, _check_cancel

        _check_cancel(cancel_event)
        paths = [Path(p) for p in _value(analysis, "selected_paths", ())]
        indices = list(_value(analysis, "selected_indices", ()))
        transforms = list(_value(analysis, "preview_transforms", ()))
        shapes = list(_value(analysis, "analysis_shapes", ()))
        reference_index = int(_value(analysis, "preview_reference_index", -1))
        reference_value = _value(analysis, "preview_reference", None)
        reference_shape = _value(analysis, "reference_analysis_shape", None)
        if len(paths) < 2 or len(paths) > 65535 or len(paths) != len(indices):
            raise ValueError("Fast fusion requires 2..65535 selected paths with matching indices")
        if len(set(indices)) != len(indices) or reference_index not in indices or reference_value is None:
            raise ValueError("Fast fusion requires unique indices and a selected reference")
        reference_local = indices.index(reference_index)
        reference = Path(reference_value)
        if not _same_path(reference, paths[reference_local]):
            raise ValueError("Reference path does not match the selected reference index")
        if not reference_shape:
            raise ValueError("Missing reference coordinate shape; alignment is not rebuilt implicitly")
        for index in indices:
            if index < 0 or index >= len(transforms) or index >= len(shapes) or not shapes[index]:
                raise ValueError("Missing source coordinate shape or transform")
            matrix = np.asarray(transforms[index], dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError("Invalid or singular source transform")
        destination = Path(output_path)
        if any(_same_path(destination, p) for p in paths):
            raise OutputCollisionError("Fast fusion cannot overwrite an input")
        if destination.exists() and not _value(output_config, "overwrite", False):
            raise OutputCollisionError(f"Refusing to overwrite existing output: {destination}")
        with Image.open(reference) as header:
            full_shape = (header.height, header.width)
        proxy_edge = int(_value(self.runtime_config, "fast_proxy_long_edge", 1600))
        radius = int(_value(self.runtime_config, "fast_seam_radius", 2))
        if proxy_edge < 32 or radius not in (1, 2, 3):
            raise ValueError("Invalid fast proxy size or seam radius")
        diagnostic("fast_input_configuration", version=VERSION, input_order=[str(p) for p in paths],
                   selected_indices=indices, reference=str(reference), reference_local=reference_local,
                   reference_global=reference_index, full_shape=list(full_shape),
                   analysis_rebuilt=False, proxy_long_edge=proxy_edge,
                   reference_coordinate_shape=list(reference_shape),
                   source_coordinate_shapes=[list(shapes[i]) for i in indices])
        encoded = {}
        render_context = {}
        from .fast_surface_tone import FastSurfaceTone
        tone_model = FastSurfaceTone(reference_local, full_shape)
        try:
            labels, proxy_shape = compute_proxy_depth_map(
                paths, indices, transforms, shapes, reference_shape, reference_local, full_shape,
                proxy_long_edge=proxy_edge, cancel_event=cancel_event, encoded_sources=encoded,
                tone_model=tone_model, render_context=render_context)
            result = render_fast_stream(
                paths, indices, transforms, shapes, reference_shape, reference_local, full_shape, labels,
                encoded_sources=encoded, seam_radius=radius, cancel_event=cancel_event,
                tone_model=tone_model, render_context=render_context)
        finally:
            encoded.clear()
        _check_cancel(cancel_event)
        with stage("encoding"), Image.fromarray(result) as image:
            final = encode_image_output(image, destination, config=output_config, original_path=reference)
        return FusionResult(Path(final), self.name, alignment_status="PREVIEW_TRANSFORMS",
                            diagnostics=(VERSION, "PROXY_FOCUS_1600_DEFAULT", "CUBIC_WARP",
                                         "ONE_FULL_RGB_PASS", "REFERENCE_FLAT_LOCK", "PAIRED_MATERIAL_TONE"))


