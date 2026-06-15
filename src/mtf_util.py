# -*- coding: utf-8 -*-
"""Shared MTF utilities for SP03 — ISO-12233 e-SFR (reused from make_gt_figure.py) + Gaussian<->MTF50."""
import numpy as np, cv2

MTF50_K = 0.18738   # Gaussian std (px) = MTF50_K / mtf50  (since MTF=exp(-2 pi^2 sigma^2 f^2))


def imread_u(p, flag=cv2.IMREAD_GRAYSCALE):
    """Unicode-path-safe imread (Korean folder 'MTF 추출')."""
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def sigma_from_mtf50(mtf50):
    """Gaussian-PSF std (px) whose MTF50 equals mtf50."""
    return MTF50_K / mtf50


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
