"""Slanted-edge MTF50 estimator used for the fixed chart (cycles/pixel)."""
import numpy as np


def esf_mtf(roi):
    """ISO-12233-style slant vertical-edge e-SFR -> (freq cy/px, MTF, mtf50)."""
    H, W = roi.shape
    rows = np.arange(H)
    g = np.abs(np.gradient(roi, axis=1))
    edge = g.argmax(1).astype(float)
    a, b = np.polyfit(rows, edge, 1)
    good = np.abs(edge - (a * rows + b)) < 6
    a, b = np.polyfit(rows[good], edge[good], 1)
    OS = 4
    cols = np.arange(W)
    pos = (cols[None, :] - (a * rows[:, None] + b)) * OS
    sel = np.abs(pos) < 30 * OS
    pos, val = pos[sel].ravel(), roi[sel].ravel()
    order = np.argsort(pos); pos, val = pos[order], val[order]
    bins = np.arange(pos.min(), pos.max(), 1.0)
    idx = np.digitize(pos, bins)
    esf = np.array([val[idx == k].mean() if (idx == k).any() else np.nan for k in range(1, len(bins))])
    m = ~np.isnan(esf); esf = np.interp(np.arange(len(esf)), np.where(m)[0], esf[m])
    lsf = np.gradient(esf); lsf *= np.hanning(len(lsf))
    mtf = np.abs(np.fft.rfft(lsf)); mtf /= mtf[0]
    freq = np.fft.rfftfreq(len(lsf), d=1.0 / OS)
    half = freq <= 0.6
    f, M = freq[half], mtf[half]
    k = np.where(M < 0.5)[0][0]
    mtf50 = np.interp(0.5, [M[k], M[k - 1]], [f[k], f[k - 1]])
    return f, M, mtf50
