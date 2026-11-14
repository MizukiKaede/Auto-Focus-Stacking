"""Share identical compact RGB features within one observation only."""
import cv2

from .statistics_native import chroma_map


class FrameFeatures:
    def __init__(self, rgb):
        self.source = rgb
        self._rgb = {}
        self._gray = {}
        self._chroma = {}

    def rgb(self, size):
        size = tuple(size)
        if size not in self._rgb:
            self._rgb[size] = cv2.resize(self.source, size, interpolation=cv2.INTER_AREA)
        return self._rgb[size]

    def gray(self, size):
        size = tuple(size)
        if size not in self._gray:
            self._gray[size] = cv2.cvtColor(self.rgb(size), cv2.COLOR_RGB2GRAY)
        return self._gray[size]

    def chroma(self, size):
        size = tuple(size)
        if size not in self._chroma:
            self._chroma[size] = chroma_map(self.rgb(size))
        return self._chroma[size]
