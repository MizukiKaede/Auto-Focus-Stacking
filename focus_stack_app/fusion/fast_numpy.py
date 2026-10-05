"""Bounded NumPy equivalents of the Fast v3 pixel kernels."""
import cv2
import numpy as np
from .native_runtime import chunks

F = np.float32


def structure_tensor_texture(xx, yy, xy, energy):
    result = np.empty(xx.shape, bool)
    x, y, cross, e, out = (a.ravel() for a in (xx, yy, xy, energy, result))
    for p in chunks(xx.size):
        diff = x[p] - y[p]
        coherence = np.sqrt(diff * diff + F(4) * cross[p] * cross[p]) / np.maximum(x[p] + y[p], F(1e-8))
        out[p] = (e[p] > F((6.0 / 255.0) ** 2)) & (coherence < F(0.6))
    return result


def hard_mask(labels, index):
    result = np.empty(labels.shape, np.uint8)
    source, out = labels.ravel(), result.ravel()
    for p in chunks(labels.size):
        np.equal(source[p], index, out=out[p])
        out[p] *= 255
    return result


def chroma(rgb):
    result = np.empty(rgb.shape[:2], np.uint8)
    source, out = rgb.reshape(-1, 3), result.ravel()
    for part in chunks(result.size):
        out[part] = source[part].max(axis=1)-source[part].min(axis=1)
    return result


def independent(detail, broad, gradient, floor):
    result = np.empty(detail.shape, bool)
    d, b, g, out = (a.ravel() for a in (detail, broad, gradient, result))
    for p in chunks(detail.size):
        out[p] = (d[p] > F(floor)) & ((d[p] > F(2)*b[p]) | (d[p] > F(16)*g[p]))
    return result


def focus_fields(xx, yy, xy, gradient, variance, detail, floor):
    base, sharp = (np.empty(detail.shape, np.float32) for _ in range(2))
    texture = np.empty(detail.shape, bool)
    x, y, cross, g, v, d, b, s, t = (a.ravel() for a in (xx, yy, xy, gradient, variance, detail, base, sharp, texture))
    for p in chunks(detail.size):
        trace, difference = x[p]+y[p], x[p]-y[p]
        b[p] = F(.65)*g[p]+F(.35)*trace+F(.10)*np.maximum(v[p], F(0))
        minor = F(.5)*(trace-np.sqrt(np.maximum(difference*difference+F(4)*cross[p]*cross[p], F(0))))
        t[p] = (minor > F(.12)*trace) & (d[p] > F(floor)) & (trace > F(2e-5))
        s[p] = np.where((g[p] > F(1e-4)) & (d[p] > F(floor)), g[p]*np.sqrt(np.maximum(d[p], F(0))), F(0))
    return base, sharp, texture


def proxy_winners(valid, evidence, texture, score, detail, index, best, labels,
                  detail_best, detail_owner, local_best, local_owner, textured):
    result = np.empty(valid.shape, bool)
    v, e, t, s, d, b, l, db, do, lb, lo, tx, out = (a.ravel() for a in
        (valid,evidence,texture,score,detail,best,labels,detail_best,detail_owner,local_best,local_owner,textured,result))
    for p in chunks(valid.size):
        usable = v[p] != 0
        use = usable & (s[p] > b[p]); b[p][use] = s[p][use]; l[p][use] = index
        use = usable & (d[p] > db[p]); db[p][use] = d[p][use]; do[p][use] = index
        use = usable & (e[p] != 0) & (d[p] > lb[p])
        out[p] = use; lb[p][use] = d[p][use]; lo[p][use] = index
        tx[p][usable] |= t[p][usable].astype(tx.dtype, copy=False)
    return result


def preserve_detail(labels, propagated, propagated_detail, local_best, local_owner, active):
    result, independent = labels.copy(), np.empty(labels.shape, bool)
    l, pr, pd, lb, lo, a, out = (v.ravel() for v in (result, propagated, propagated_detail, local_best, local_owner, active, independent))
    for p in chunks(labels.size):
        use = a[p] != 0
        out[p] = use & (lb[p] > F(0)) & (pd[p] < F(.7)*lb[p])
        l[p][use] = np.where(out[p][use], lo[p][use], pr[p][use])
    return result, independent


def nearest_support(strength, edge, radius):
    edge = np.ascontiguousarray(edge != 0)
    if not np.any(edge):
        return np.zeros_like(strength)
    present = edge.astype(np.float32)
    density = cv2.GaussianBlur(present, (0,0), 2.0)
    averaged = cv2.GaussianBlur(strength*present, (0,0), 2.0)
    distance, nearest = cv2.distanceTransformWithLabels(np.uint8(~edge), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    values = np.zeros(int(nearest.max())+1, np.float32)
    values[nearest[edge]] = averaged[edge]/np.maximum(density[edge], F(1e-6))
    result = np.empty_like(strength)
    out, d, near = result.ravel(), distance.ravel(), nearest.ravel()
    denominator = max(F(2), F(radius)/F(4))
    for p in chunks(strength.size):
        weight = np.clip((F(radius)+F(1)-d[p])/denominator, F(0), F(1))
        out[p] = values[near[p]]*weight
    return result


def neutral_filter(positions, details, local_best, chroma, foreground, material):
    result = np.empty(positions.shape, bool)
    pos, d, out = positions.ravel(), details.ravel(), result.ravel()
    lb, c, fg, m = (a.ravel() for a in (local_best,chroma,foreground,material))
    for p in chunks(positions.size):
        i = pos[p]
        out[p] = (lb[i] > F(0)) & (c[i] < 65) & (fg[i] != 0) & (m[i] != 0) & (d[p] >= F(.7)*lb[i])
    return result


def neutral_update(targets, material, strength, best, owner, chroma, index):
    t, m, s, b, o, c = (a.ravel() for a in (targets,material,strength,best,owner,chroma))
    for p in chunks(best.size):
        use = (t[p] != 0) & (m[p] != 0) & (s[p] > b[p])
        b[p][use] = s[p][use]; o[p][use] = index; c[p][use] = 0


def neutral_apply(labels, owner, best, chroma, protected):
    result = labels.copy(); guarded = changed = 0
    l, o, b, c, pr = (a.ravel() for a in (result,owner,best,chroma,protected))
    for p in chunks(labels.size):
        use = (b[p] > F(1e-6)) & (c[p] < 65) & (pr[p] == 0)
        guarded += int(np.count_nonzero(use))
        changed += int(np.count_nonzero(l[p][use] != o[p][use]))
        l[p][use] = o[p][use]
    return result, guarded, changed


def contour_energy(gradient, detail, seeds):
    present, weighted = (np.empty(gradient.shape, np.float32) for _ in range(2))
    g, d, s, p, w = (a.ravel() for a in (gradient,detail,seeds,present,weighted))
    for part in chunks(gradient.size):
        p[part] = s[part] != 0
        w[part] = g[part]*np.sqrt(np.maximum(d[part], F(0)))*p[part]
    density = cv2.GaussianBlur(present, (0,0), 8.0)
    coherent = cv2.GaussianBlur(weighted, (0,0), 8.0)
    out, d = coherent.ravel(), density.ravel()
    for part in chunks(gradient.size):
        out[part] /= np.maximum(d[part], F(1e-6))
    return coherent


def region(tile):
    return np.s_[tile['y']:tile['y1'], tile['x']:tile['x1']]


def rank(tile, valid, score, index, reference):
    usable = (tile['mask'] != 0) & (valid[region(tile)] != 0)
    better = usable & (score > tile['best'])
    tile['best'][better] = score[better]; tile['owner'][better] = index
    if reference:
        tile['reference_score'][usable] = score[usable]
    return better


def capture(tile, rgb, better, reference):
    source = rgb[region(tile)]
    use = better != 0
    tile['rgb'][use] = source[use]
    if reference:
        tile['reference_rgb'][:] = source


def select(tile, covered, reference):
    use = (tile['mask'] != 0) & (tile['best'] > F(0))
    use_reference = use & (tile['reference_score'] >= F(.9)*tile['best'])
    tile['rgb'][use_reference] = tile['reference_rgb'][use_reference]
    tile['owner'][use_reference] = reference
    replaced = np.count_nonzero(use & (tile['owner'] != tile['labels']))
    covered[region(tile)] = use
    return int(np.count_nonzero(use_reference)), int(replaced)


def blend(tile, output, covered, distance, fade):
    rect = region(tile)
    target, source = output[rect], tile['rgb']
    # Row blocks keep temporaries bounded even for unusually large tiles.
    rows = max(1, 262144 // target.shape[1])
    for start in range(0, target.shape[0], rows):
        p = slice(start, start+rows)
        use = covered[rect][p] != 0
        t = np.minimum(distance[rect][p]/F(fade), F(1))
        a = (t*t*(F(3)-F(2)*t))[..., None]
        pixels = np.rint(np.clip(target[p].astype(np.float32)*(F(1)-a)+source[p].astype(np.float32)*a, F(0), F(255))).astype(np.uint8)
        target[p][use] = pixels[use]


class Render:
    def __init__(self, labels, band):
        self.labels, self.band = labels.ravel().copy(), band.ravel().copy()
        self.copied = np.zeros(labels.size, np.uint8)
        self.seam = np.flatnonzero(self.band)
        self.colors = np.zeros((self.seam.size,3), np.float32)
        self.weights = np.zeros(self.seam.size, np.float32)

    def owner(self, index, shape):
        mask = (self.labels == index).astype(np.uint8).reshape(shape)
        return mask, int(np.count_nonzero(mask))

    def add(self, index, rgb, valid, feather, output):
        source, out, v = rgb.reshape(-1,3), output.reshape(-1,3), valid.ravel()
        for p in chunks(self.labels.size):
            use = (self.labels[p] == index) & (self.band[p] == 0) & (v[p] != 0)
            out[p][use] = source[p][use]; self.copied[p][use] = 1
        if feather is None:
            return
        f = feather.ravel()
        for p in chunks(self.seam.size):
            i = self.seam[p]
            weight = f[i]*v[i].astype(np.float32)
            self.colors[p] += source[i].astype(np.float32)*weight[:,None]
            self.weights[p] += weight

    def finish(self, output):
        out = output.reshape(-1,3)
        for p in chunks(self.seam.size):
            use = self.weights[p] > F(1e-8)
            i = self.seam[p][use]
            out[i] = np.rint(np.clip(self.colors[p][use]/self.weights[p][use,None], F(0), F(255))).astype(np.uint8)
            self.copied[i] = 1
        return int(np.count_nonzero(self.copied == 0))
