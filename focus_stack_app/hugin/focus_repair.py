"""Source ownership and material tone repair for Hugin's external masks.

The Quality renderer is unchanged. Hugin applies the existing statistics to
its own aligned TIFFs, then passes corrected lossless inputs to Enfuse.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.performance import diagnostic, stage


HUGIN_FOCUS_REPAIR_VERSION = "hugin-material-ownership-v6-stack-consistent-local-detail"


class HuginFocusRepair:
    def __init__(self, reference_index, *, edge_ownership=True, surface_tone=True,
                 stabilize_texture=False, edge_mode="localized"):
        self.boundary = self.printed = self.tone = None
        self.texture = None
        self.exterior = None
        self.target_detail = None
        if stabilize_texture:
            from ..fusion.quality_fusion import FlatTextureStatistics
            self.texture = FlatTextureStatistics(reference_index, edge_mode=edge_mode)
        if edge_ownership:
            from ..fusion.quality_fusion import PrintedEdgeOwnership, SurfaceBoundaryOwnership
            self.boundary = SurfaceBoundaryOwnership()
            self.printed = PrintedEdgeOwnership()
            from .exterior_ownership import ExteriorRimOwnership
            self.exterior = ExteriorRimOwnership()
            from .target_detail import NeutralTargetDetail
            self.target_detail = NeutralTargetDetail()
        if surface_tone:
            from ..fusion.surface_tone import SurfaceToneHarmonizer
            self.tone = SurfaceToneHarmonizer(reference_index)
        diagnostic("hugin_focus_repair_configuration", reference_index=reference_index,
                   edge_ownership=bool(edge_ownership), surface_tone=bool(surface_tone),
                   stabilize_texture=bool(stabilize_texture),
                   version=HUGIN_FOCUS_REPAIR_VERSION)

    def observe(self, index, rgb, gray, score):
        if self.texture is not None:
            self.texture.observe(index, rgb, gray, score)
        if self.boundary is not None:
            self.boundary.observe(index, rgb, score)
            self.printed.observe(index, rgb, gray, score)
            self.exterior.observe(index, rgb)
            self.target_detail.observe(index, rgb, gray, score)
        if self.tone is not None:
            self.tone.observe(index, rgb)

    def apply(self, labels, *, load_aligned=None):
        raw_labels = labels.copy()
        if self.texture is not None:
            labels, protected = self.texture.apply(labels)
            del protected
        # Run after Hugin's texture regularizer. Ink and its adjacent
        # paint fringe must retain one continuous source at the final step.
        if self.boundary is not None:
            labels = self.boundary.apply(labels)
            labels = self.printed.apply(
                labels, protected_texture=self.boundary.texture_protection(labels.shape),
            )
        if self.exterior is not None:
            labels = self.exterior.apply(labels)
        if self.target_detail is not None and load_aligned is not None:
            labels = self.target_detail.apply(labels, raw_labels, load_aligned)
        return labels

    def corrected_inputs(self, paths, load_aligned, work_dir, *, cancel_event=None, force_rewrite=False):
        if self.tone is None and not force_rewrite:
            return paths
        from PIL import Image
        from .enfuse import EnfuseError

        corrected_paths = []
        output_dir = Path(work_dir) / "tone_inputs"
        for index, path in enumerate(paths):
            if cancel_event is not None and cancel_event.is_set():
                raise EnfuseError("Enfuse cancelled while correcting material tone")
            rgb = load_aligned(index)
            corrected = self.tone.correct(rgb, index) if self.tone is not None else rgb
            if corrected is rgb and not force_rewrite:
                corrected_paths.append(path)
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"frame-{index:04d}.tif"
            with stage("hugin_surface_tone_tiff_write", frame=index):
                Image.fromarray(corrected).save(target, compression="tiff_deflate")
            corrected_paths.append(target)
        diagnostic("hugin_surface_tone_inputs", frames=len(paths),
                   corrected_tiff_count=sum(a != b for a, b in zip(paths, corrected_paths)),
                   drifting_materials=self.tone.drifting_materials if self.tone is not None else [],
                   version=HUGIN_FOCUS_REPAIR_VERSION)
        return tuple(corrected_paths)
