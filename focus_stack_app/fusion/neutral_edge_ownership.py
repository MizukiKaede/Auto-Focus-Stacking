"""Keep a long neutral edge on a coherent, locally sharp source."""

import cv2
import numpy as np

from ..utils.performance import diagnostic, timed


class NeutralEdgeOwnership:
    """Choose one source along each straight edge before feathering focus labels.

    Aligned frames can still disagree by a few pixels on a physical edge.
    Keep the complete luminance transition on one source, with a narrow collar
    around it. Only retain compact line profiles and a proxy colour mask.
    """

    def __init__(self, reference_index):
        self.reference_index = reference_index
        self.size = None
        self.lines = []
        self.colour = None
        self.silhouette_gray = None

    def _find_lines(self, gray, chroma, full_shape):
        height, width = gray.shape
        if min(height, width) < 32:
            return
        segments = cv2.createLineSegmentDetector().detect(gray)[0]
        if segments is None:
            return
        for start, end in segments.reshape(-1, 2, 2):
            length = np.linalg.norm(end - start)
            if length < 16:
                continue
            direction = (end - start) / length
            normal = direction[::-1] * (-1, 1)
            along = np.linspace(.05, .95, max(7, int(length / 3)), dtype=np.float32)
            centres = start + along[:, None] * (end - start)
            offsets = np.arange(-8, 9, dtype=np.float32)
            profile = centres + offsets[:, None, None] * normal
            x, y = profile.transpose(2, 0, 1).astype(np.float32)
            if x.min() < 1 or y.min() < 1 or x.max() >= width - 1 or y.max() >= height - 1:
                continue
            luma = cv2.remap(gray, x, y, cv2.INTER_LINEAR).astype(np.float32)
            sides = np.stack((luma[:3].mean(axis=0), luma[-3:].mean(axis=0)))
            contrast = sides[1] - sides[0]
            polarity = np.sign(np.median(contrast))
            colour = cv2.remap(chroma, x, y, cv2.INTER_LINEAR).max(axis=0)
            support = (contrast * polarity >= 18) & (colour < 38)
            if np.mean(support) < .75:
                continue
            # Cross-sections agree on edge polarity and have neutral support.
            # Remap their pixel centres back to the full-resolution label grid.
            x, y = x[:, support], y[:, support]
            full_x = np.rint((x + .5) * full_shape[1] / width - .5).astype(np.intp)
            full_y = np.rint((y + .5) * full_shape[0] / height - .5).astype(np.intp)
            # On the silhouette against a bright backdrop, even a short
            # interruption in a long edge is conspicuous. The strong step
            # also makes the one-source sharpness comparison reliable.
            silhouette = np.mean((np.abs(contrast[support]) >= 75)
                                 & (sides[:, support].max(axis=0) >= 175)) >= .75
            self.lines.append((length, start, end, polarity, silhouette, along[support],
                               x, y, full_x, full_y, {}))
        self.lines.sort(key=lambda line: line[0])

    @timed("neutral_edge_observe")
    def observe(self, index, rgb, valid, features):
        height, width = rgb.shape[:2]
        first = self.size is None
        if first:
            scale = min(.5, 2048 / max(height, width))
            self.size = (max(1, round(width * scale)), max(1, round(height * scale)))
        elif not self.lines:
            return
        gray = features.gray(self.size)
        chroma = features.chroma(self.size)
        if first:
            self._find_lines(gray, chroma, (height, width))
            self.colour = np.zeros(gray.shape, np.uint8)
            self.silhouette_gray = gray
        self.colour |= chroma >= 38
        for _, _, _, polarity, _, _, x, y, full_x, full_y, scores in self.lines:
            strip = cv2.remap(gray, x, y, cv2.INTER_LINEAR).astype(np.float32)
            # Peak normal gradient near the line, across a two-pixel baseline.
            gradient = (strip[5:14] - strip[3:12]) * polarity
            peak = gradient.max(axis=0)
            column = np.arange(peak.size)
            position = gradient.argmax(axis=0)
            left = gradient[np.maximum(position - 1, 0), column]
            right = gradient[np.minimum(position + 1, len(gradient) - 1), column]
            curvature = np.maximum(2 * peak - left - right, 1)
            location = position + .5 * (right - left) / curvature
            peak[~valid[full_y, full_x].all(axis=0)] = 0
            scores[index] = peak, location

    def _join_silhouette(self, owners, endpoints, untouched):
        """Continue the same focused source through short corner bevels."""
        joined = 0
        height, width = owners.shape
        for source, points in endpoints.items():
            points = np.asarray(points, dtype=np.float32)
            for index, start in enumerate(points):
                nearby = points[index + 1:]
                lengths = np.linalg.norm(nearby - start, axis=1)
                close = (lengths > 8) & (lengths <= 22)
                for end, length in zip(nearby[close], lengths[close]):
                    direction = (end - start) / length
                    normal = direction[::-1] * (-1, 1)
                    centres = start + np.linspace(.2, .8, 7)[:, None] * (end - start)
                    strip = centres + np.arange(-6, 7, dtype=np.float32)[:, None, None] * normal
                    x, y = strip.transpose(2, 0, 1).astype(np.float32)
                    luma = cv2.remap(self.silhouette_gray, x, y, cv2.INTER_LINEAR)
                    sides = np.stack((luma[:2].mean(axis=0), luma[-2:].mean(axis=0)))
                    contrast = sides[1] - sides[0]
                    support = ((contrast * np.sign(np.median(contrast)) >= 60)
                               & (sides.max(axis=0) >= 175))
                    # A short bevel may end at another strong edge. Verify
                    # the same bright-side step along the middle of the join.
                    if np.count_nonzero(support) < 6:
                        continue
                    lo = np.maximum(np.floor(np.minimum(start, end)).astype(int) - 5, 0)
                    hi = np.minimum(np.ceil(np.maximum(start, end)).astype(int) + 6, (width, height))
                    region = owners[lo[1]:hi[1], lo[0]:hi[0]]
                    bridge = np.zeros(region.shape, np.uint8)
                    cv2.line(bridge, tuple(np.rint(start).astype(int) - lo),
                             tuple(np.rint(end).astype(int) - lo), 1, thickness=9)
                    new_pixels = (bridge != 0) & (region == untouched)
                    region[new_pixels] = source
                    joined += bool(np.any(new_pixels))
        return joined

    @timed("neutral_edge_ownership")
    def apply(self, labels):
        if not self.lines:
            return
        untouched = np.iinfo(np.uint16).max
        owners = np.full(self.colour.shape, untouched, np.uint16)
        selected_lines = 0
        silhouette_endpoints = {}
        for length, start, end, _, silhouette, along, _, _, full_x, full_y, scores in self.lines:
            indices = sorted(scores)
            strengths, locations = (np.stack(values) for values in zip(*(
                scores[index] for index in indices)))
            best = strengths.max(axis=0)
            # A single source must stay sharp along this edge. If focus varies
            # substantially along its length, leave the local decisions in place.
            if np.mean(best >= 12) < .9:
                continue
            coverage = (strengths >= .78 * best).mean(axis=1)
            average = strengths.mean(axis=1)
            eligible = coverage >= .85
            if not np.any(eligible):
                continue
            winner = int(np.argmax(np.where(eligible, average, -1)))
            reference = indices.index(self.reference_index)
            if eligible[reference] and average[reference] >= .92 * average[winner]:
                winner = reference
            current = np.searchsorted(indices, labels[full_y[6:11], full_x[6:11]])
            if silhouette and np.unique(current, return_counts=True)[1].max() < .94 * current.size:
                cv2.line(owners, tuple(np.rint(start).astype(int)),
                         tuple(np.rint(end).astype(int)), indices[winner], thickness=9)
                silhouette_endpoints.setdefault(indices[winner], []).extend((start, end))
                selected_lines += 1
                continue
            columns = np.arange(along.size)
            offsets = locations[current, columns] - locations[winner]
            # A handful of changes at one end of a long edge can be visible
            # even when most of that edge already has a single owner.
            bounds = np.linspace(0, 1, max(2, int(np.ceil(length / 64)) + 1))
            draw_line = False
            for begin, finish in zip(bounds, bounds[1:]):
                local = (along >= begin) & (along <= finish)
                if np.count_nonzero(local) < 4:
                    continue
                selected = current[:, local]
                if np.unique(selected, return_counts=True)[1].max() >= .94 * selected.size:
                    continue
                low, high = np.percentile(offsets[:, local], (8, 92))
                if high - low < .35 or np.mean(strengths[winner, local] >= .78 * best[local]) < .85:
                    continue
                a = tuple(np.rint(start + begin * (end - start)).astype(int))
                b = tuple(np.rint(start + finish * (end - start)).astype(int))
                cv2.line(owners, a, b, indices[winner], thickness=9)
                if silhouette:
                    silhouette_endpoints.setdefault(indices[winner], []).extend((a, b))
                draw_line = True
            selected_lines += draw_line
        if not selected_lines:
            return
        joined = self._join_silhouette(owners, silhouette_endpoints, untouched)
        # Preserve nearby coloured hardware, printing and their transition.
        colour = cv2.dilate(self.colour, np.ones((9, 9), np.uint8))
        owners[colour != 0] = untouched
        full_owners = cv2.resize(owners, (labels.shape[1], labels.shape[0]),
                                 interpolation=cv2.INTER_NEAREST_EXACT)
        guard = full_owners != untouched
        np.copyto(labels, full_owners, where=guard)
        diagnostic("neutral_straight_edge_ownership", lines=selected_lines, corner_joins=joined,
                   guarded_pixels=int(np.count_nonzero(guard)))
