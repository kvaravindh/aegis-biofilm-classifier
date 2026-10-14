"""
bfcore.py — portable core of the biofilm classifier.

Every operation here is written out explicitly (own Gaussian kernels, own reflect
padding, own Otsu, own connected components, own disk sums) so that the browser
version (biofilm_tool.html) can mirror it line for line and give the same numbers.
"""
from __future__ import annotations

import numpy as np

SIGMAS = (1.0, 2.0, 4.0, 8.0)
NORM_SIGMA = 16.0
N_FEATURES = 1 + 8 * len(SIGMAS)
BSUB, ECOLI = 1, 0


# ----------------------------------------------------------------------------- filtering

def reflect_index(i: np.ndarray, n: int) -> np.ndarray:
    """Half-sample symmetric extension (d c b a | a b c d | d c b a), periodic 2n."""
    i = np.mod(i, 2 * n)
    return np.where(i >= n, 2 * n - 1 - i, i)


def kernel(sigma: float, order: int) -> np.ndarray:
    r = int(4.0 * sigma + 0.5)
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * x * x / (sigma * sigma))
    g = g / g.sum()
    if order == 0:
        return g
    if order == 1:
        return x / (sigma * sigma) * g
    if order == 2:
        return (x * x / sigma ** 4 - 1.0 / (sigma * sigma)) * g
    raise ValueError(order)


def corr1d(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """out[i] = sum_j k[j] * a[reflect(i + j - r)], accumulated in order j = 0..2r."""
    r = (len(k) - 1) // 2
    n = a.shape[axis]
    idx = reflect_index(np.arange(-r, n + r), n)
    p = np.take(a, idx, axis=axis)
    out = np.zeros(a.shape, np.float64)
    for j in range(len(k)):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(j, j + n)
        out = out + k[j] * p[tuple(sl)]
    return out


def filt(a, sigma, oy=0, ox=0):
    """Separable Gaussian (derivative) filter: rows (axis 0) first, then columns."""
    return corr1d(corr1d(a, kernel(sigma, oy), 0), kernel(sigma, ox), 1)


# ----------------------------------------------------------------------------- stain + mask

def percentile_u8(v: np.ndarray, q: float) -> float:
    s = np.sort(v.ravel().astype(np.float64))
    pos = q / 100.0 * (len(s) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (pos - lo) * (s[hi] - s[lo]))


def stain_od(rgb: np.ndarray) -> np.ndarray:
    g = rgb[..., 1].astype(np.float64)
    bg = max(percentile_u8(g, 99.5), 1.0)
    return -np.log10(np.clip(g / bg, 1e-3, 1.0))


def otsu(a: np.ndarray, nbins: int = 256) -> float:
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return lo
    w = (hi - lo) / nbins
    b = np.minimum(((a.ravel() - lo) / (hi - lo) * nbins).astype(np.int64), nbins - 1)
    hist = np.bincount(b, minlength=nbins).astype(np.float64)
    centers = lo + (np.arange(nbins) + 0.5) * w
    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * centers) / np.maximum(w1, 1e-300)
    m2 = (np.cumsum((hist * centers)[::-1]) / np.maximum(w2[::-1], 1e-300))[::-1]
    v = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return float(centers[int(np.argmax(v))])


def label4(m: np.ndarray):
    """4-connected component labels and sizes (sizes don't depend on label order,
    so this matches the JS flood fill)."""
    from scipy import ndimage as ndi
    lab, n = ndi.label(m)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0
    return lab, sizes


def biomass_mask(od: np.ndarray, min_size: int = 30):
    sm = filt(od, 1.0)
    thr = min(max(otsu(sm), 0.08), 0.25)
    m = sm > thr
    lab, sz = label4(m)
    m = m & (sz[lab] >= min_size)
    lab, sz = label4(~m)
    m = m | ((~m) & (sz[lab] < min_size))
    return m, thr


# ----------------------------------------------------------------------------- features

def features(od: np.ndarray) -> np.ndarray:
    """(H, W, 33) float32 morphology features (no hue, no absolute stain level)."""
    mu = filt(od, NORM_SIGMA)
    var = filt((od - mu) ** 2, NORM_SIGMA)
    x = (od - mu) / np.sqrt(var + 1e-3)
    out = [x]
    ix = iy = None
    for s in SIGMAS:
        gs = filt(x, s)
        gy = filt(x, s, 1, 0)
        gx = filt(x, s, 0, 1)
        if ix is None:
            ix, iy = gx, gy  # derivative at the finest scale, reused for the structure tensor
        hyy = filt(x, s, 2, 0)
        hxx = filt(x, s, 0, 2)
        hxy = filt(x, s, 1, 1)
        tr = hxx + hyy
        disc = np.sqrt(((hxx - hyy) * 0.5) ** 2 + hxy * hxy)
        l1 = tr * 0.5 + disc
        l2 = tr * 0.5 - disc
        axx = filt(ix * ix, 2 * s)
        ayy = filt(iy * iy, 2 * s)
        axy = filt(ix * iy, 2 * s)
        coh = np.sqrt((axx - ayy) ** 2 + 4 * axy * axy) / (axx + ayy + 1e-6)
        tex = np.sqrt(np.maximum(filt(x * x, s) - gs * gs, 0.0))
        s2 = s * s
        out += [gs, np.sqrt(gx * gx + gy * gy), tr * s2, l1 * s2, l2 * s2,
                (np.abs(l1) - np.abs(l2)) / (np.abs(l1) + np.abs(l2) + 1e-6), coh, tex]
    return np.stack(out, -1).astype(np.float32)


# ----------------------------------------------------------------------------- local fraction

def disk_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum of a over the disk x^2 + y^2 <= r^2 around each pixel, reflect padding."""
    H, W = a.shape
    yi = reflect_index(np.arange(-r, H + r), H)
    xi = reflect_index(np.arange(-r, W + r), W)
    p = a[np.ix_(yi, xi)].astype(np.float64)
    cs = np.concatenate([np.zeros((p.shape[0], 1)), np.cumsum(p, 1)], 1)  # row prefix sums
    out = np.zeros((H, W))
    for dy in range(-r, r + 1):
        hw = int(np.floor(np.sqrt(r * r - dy * dy)))
        rows = cs[r + dy: r + dy + H]
        xs = np.arange(W) + r
        out += rows[:, xs + hw + 1] - rows[:, xs - hw]
    return out


# ----------------------------------------------------------------------------- forest

class Forest:
    """Flat random-forest representation shared with the browser tool."""

    def __init__(self, feat, thr, left, right, roots):
        self.feat, self.thr, self.left, self.right, self.roots = feat, thr, left, right, roots

    @classmethod
    def from_sklearn(cls, rf):
        ci = list(rf.classes_).index(BSUB)
        F, T, L, R, roots = [], [], [], [], []
        off = 0
        for est in rf.estimators_:
            t = est.tree_
            n = t.node_count
            roots.append(off)
            leaf = t.children_left == -1
            val = t.value[:, 0, :]
            pb = val[:, ci] / val.sum(1)
            F.append(np.where(leaf, 255, t.feature).astype(np.uint8))
            T.append(np.where(leaf, pb, t.threshold).astype(np.float64))
            L.append(np.where(leaf, 0, t.children_left + off).astype(np.uint32))
            R.append(np.where(leaf, 0, t.children_right + off).astype(np.uint32))
            off += n
        return cls(np.concatenate(F), np.concatenate(T), np.concatenate(L), np.concatenate(R),
                   np.array(roots, np.uint32))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(B. subtilis) for rows of X (float32), averaged over trees."""
        X = X.astype(np.float32)
        acc = np.zeros(len(X))
        rows = np.arange(len(X))
        for r in self.roots:
            node = np.full(len(X), r, np.int64)
            while True:
                f = self.feat[node]
                inner = f != 255
                if not inner.any():
                    break
                ni = node[inner]
                go_left = X[rows[inner], f[inner]].astype(np.float64) <= self.thr[ni]
                node[inner] = np.where(go_left, self.left[ni], self.right[ni])
            acc += self.thr[node]
        return acc / len(self.roots)


# ----------------------------------------------------------------------------- call

def classify(forest: Forest, od, mask, window=15, overlap_thr=0.25, tpr=1.0, fpr=0.0, X=None):
    H, W = od.shape
    if X is None:
        X = features(od)
    prob = np.zeros(H * W)
    idx = np.flatnonzero(mask.ravel())
    if len(idx):
        prob[idx] = forest.predict(X.reshape(-1, X.shape[-1])[idx])
    prob = prob.reshape(H, W)
    hard = (prob >= 0.5) & mask
    nb = disk_sum(hard.astype(np.float64), window)
    nm = disk_sum(mask.astype(np.float64), window)
    fb = np.where(nm > 0, nb / np.maximum(nm, 1e-9), 0.0)
    if tpr - fpr > 0.2:
        fb = np.clip((fb - fpr) / (tpr - fpr), 0.0, 1.0)
    lab = np.zeros((H, W), np.uint8)
    lab[mask & (fb >= 1 - overlap_thr)] = 1
    lab[mask & (fb <= overlap_thr)] = 2
    lab[mask & (fb > overlap_thr) & (fb < 1 - overlap_thr)] = 3
    n = int(mask.sum())
    w = od * mask
    wn = w.sum()
    pct = lambda c: 100.0 * c / n if n else float("nan")
    stats = dict(
        biomass_area_pct_of_image=100.0 * n / mask.size,
        bsub_only_pct=pct((lab == 1).sum()),
        ecoli_only_pct=pct((lab == 2).sum()),
        overlap_pct=pct((lab == 3).sum()),
        bsub_total_pct=pct(fb[mask].sum()),
        ecoli_total_pct=pct((1 - fb[mask]).sum()),
        bsub_total_pct_stain_weighted=100.0 * (fb * w).sum() / wn if wn else float("nan"),
        ecoli_total_pct_stain_weighted=100.0 * ((1 - fb) * w).sum() / wn if wn else float("nan"),
        mean_confidence=float(np.mean(np.abs(prob[mask] - 0.5) * 2)) if n else float("nan"),
    )
    return dict(prob=prob, frac_bsub=fb, labels=lab, stats=stats)
