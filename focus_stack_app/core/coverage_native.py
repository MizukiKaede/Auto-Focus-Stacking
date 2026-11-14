"""Packed sharp-mask counting; old DLLs retain the boolean implementation."""
import ctypes as C
import operator
from pathlib import Path

import numpy as np

from ..fusion.native_runtime import NativeLibrary
from ..utils.performance import diagnostic

P, N = C.c_void_p, C.c_int64
_library = NativeLibrary(
    Path(__file__).resolve().parents[1] / 'fusion' / 'statistics_native.dll',
    {'coverage_counts': ([P, P, P, P, N, N], None),
     'coverage_add': ([P, P, N], N)},
    abi_name='statistics_core_abi', expected_abi=1, optional_abi=True,
)


class CoverageMasks:
    def __init__(self, scores, threshold, useful):
        self.count = len(scores)
        self.native = _library.available
        self.covered_count = 0
        if self.native:
            self.words = (useful.size + 63) // 64
            self.masks = np.zeros((self.count, self.words), np.uint64)
            for index, score in enumerate(scores):
                packed = np.packbits(((score >= threshold) & useful).ravel(), bitorder='little')
                self.masks[index].view(np.uint8)[:packed.size] = packed
            self.covered = np.zeros(self.words, np.uint64)
        else:
            self.masks = [(score >= threshold) & useful for score in scores]
            self.covered = np.zeros_like(useful)
        diagnostic('selection_coverage_execution', native=self.native,
                   mask_bytes=(self.masks.nbytes if self.native else sum(m.nbytes for m in self.masks)),
                   frames=self.count, pixels=useful.size, **_library.info())

    def counts(self, options):
        indices = np.ascontiguousarray(options, dtype=np.int64)
        if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= self.count):
            raise ValueError('coverage source index outside frame range')
        if self.native:
            counts = np.empty(indices.size, np.int64)
            _library.call('coverage_counts', self.masks.ctypes.data, self.covered.ctypes.data,
                          indices.ctypes.data, counts.ctypes.data, self.words, indices.size)
            return dict(zip(options, map(int, counts)))
        uncovered = ~self.covered
        return {index: int(np.count_nonzero(self.masks[index] & uncovered)) for index in options}

    def add(self, index):
        index = operator.index(index)
        if not 0 <= index < self.count:
            raise ValueError('coverage source index outside frame range')
        if self.native:
            self.covered_count = int(_library.call(
                'coverage_add', self.masks[index].ctypes.data, self.covered.ctypes.data, self.words))
        else:
            self.covered |= self.masks[index]
            self.covered_count = int(np.count_nonzero(self.covered))

