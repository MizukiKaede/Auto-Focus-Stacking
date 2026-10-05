"""Bounded residual affine refinement of already aligned Hugin TIFFs."""
from __future__ import annotations

import cv2
import numpy as np

from ..utils.performance import diagnostic


class HuginAlignmentRefiner:
    def __init__(self, loader, reference_index, long_edge=1600, *, cpu_budget=12):
        from .parallel import validate_budget
        self.cpu_budget = validate_budget(cpu_budget)
        self.loader = loader
        self.reference_index = reference_index
        reference = loader(reference_index)
        self.shape = reference.shape[:2]
        h, w = self.shape
        scale = min(1.0, long_edge / max(h, w))
        self.size = (max(1, round(w * scale)), max(1, round(h * scale)))
        self.reference = self._gray(reference)
        self.matrices = {reference_index: np.eye(3, dtype=np.float32)}
        self.events = {}
        self.prepared_count = None

    def _gray(self, rgb):
        if rgb.shape[:2] != self.shape:
            raise ValueError("Hugin residual refinement requires matching aligned dimensions")
        small = cv2.resize(rgb, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return cv2.GaussianBlur(gray, (0, 0), 1.0)

    def _resolve(self, index, rgb):
        if index not in self.matrices:
            _, matrix, event = self._estimate((index, self._gray(rgb)))
            self.matrices[index] = matrix
            self.events[index] = event

    def _estimate(self, item):
        index, gray = item
        h, w = self.shape
        matrix = np.eye(3, dtype=np.float32)
        accepted, correlation, displacement = False, None, None
        try:
            correlation, affine = cv2.findTransformECC(
                self.reference, gray, np.eye(2, 3, dtype=np.float32),
                cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6),
            )
            small = np.eye(3, dtype=np.float32); small[:2] = affine
            scaling = np.diag([w / self.size[0], h / self.size[1], 1.0])
            # ECC maps template to source; the renderer needs source to template.
            candidate = (scaling @ np.linalg.inv(small) @ np.linalg.inv(scaling)).astype(np.float32)
            corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
            warped = cv2.perspectiveTransform(corners[None], candidate)[0]
            displacement = float(np.linalg.norm(warped - corners, axis=1).max())
            singular = np.linalg.svd(candidate[:2, :2], compute_uv=False)
            accepted = bool(correlation >= 0.98 and displacement <= 32
                            and singular.min() >= 0.98 and singular.max() <= 1.02)
            if accepted and displacement >= 0.5:
                matrix = candidate
        except (cv2.error, np.linalg.LinAlgError):
            pass
        return index, matrix, dict(frame=int(index), accepted=accepted,
                                   correlation=correlation, max_corner_displacement=displacement)

    def _emit(self, index):
        event = self.events[index]
        diagnostic("hugin_residual_affine", **event,
                   source_to_reference=self.matrices[index].tolist(),
                   version="hugin-residual-affine-v2-stack-consistent")

    def prepare(self, frame_count, *, cancel_event=None):
        """Validate the complete coordinate change before observing any frame.

        A hard per-frame rejection mixes original and refined coordinate
        systems. At the acceptance boundary even adjacent focus planes can
        jump by tens of pixels. If any estimate fails the existing bounds,
        retain the original Hugin geometry for the complete stack. No rejected
        transform is extrapolated and no frame is dropped from focus selection.
        """
        if self.prepared_count is not None:
            if self.prepared_count != frame_count:
                raise ValueError("Hugin refinement stack size cannot change after preparation")
            return
        # Restore the original caller-thread scan. Independent stacks may
        # progress concurrently; ECC no longer enters the shared mask budget.
        # OpenCV retains the application-level internal thread setting.
        from .parallel import _check_cancel
        diagnostic('hugin_residual_execution', execution='serial_per_stack',
                   shared_preparation_budget=False,
                   opencv_threads=max(1, cv2.getNumThreads()))
        matrices, events = {}, {}
        for index in range(frame_count):
            _check_cancel(cancel_event)
            if index != self.reference_index and index not in self.matrices:
                gray = self._gray(self.loader(index))
                _check_cancel(cancel_event)
                _, matrix, event = self._estimate((index, gray))
                _check_cancel(cancel_event)
                matrices[index] = matrix
                events[index] = event
        # Cancel/failure leaves no partially prepared coordinate system.
        self.matrices.update(matrices)
        self.events.update(events)
        rejected = [i for i, event in self.events.items() if not event['accepted']]
        if rejected:
            for index in range(frame_count):
                self.matrices[index] = np.eye(3, dtype=np.float32)
            for event in self.events.values():
                event['candidate_accepted'] = event['accepted']
                event['accepted'] = False
        self.prepared_count = frame_count
        diagnostic("hugin_residual_stack_gate", frames=frame_count,
                   refinement_enabled=not rejected, rejected_frames=rejected,
                   fallback="original_hugin_stack" if rejected else None,
                   version="hugin-residual-affine-v2-stack-consistent")
        for index in sorted(self.events):
            self._emit(index)

    def load(self, index):
        rgb = self.loader(index)
        if self.prepared_count is not None and not 0 <= index < self.prepared_count:
            raise IndexError("Hugin frame outside the prepared stack")
        if index not in self.matrices:
            self._resolve(index, rgb)
            self._emit(index)
        h, w = self.shape
        matrix = self.matrices[index]
        if np.array_equal(matrix, np.eye(3, dtype=np.float32)):
            return rgb
        return cv2.warpPerspective(rgb, matrix, (w, h), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT_101)
