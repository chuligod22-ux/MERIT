# -*- coding: utf-8 -*-
"""
SP03 C1 — physically-calibrated degradation synthesis.

A high-quality (least-degraded) reference frame is degraded toward a TARGET operating
condition using the MEASURED per-condition parameters, not guesses:
  - anisotropic optical+motion blur   : Gaussian std from measured MTF_h / MTF_v (cam1_analysis.csv)
    (horizontal axis carries along-track motion blur -> larger; vertical = optics)
  - sensor noise                       : additive Gaussian from measured sigma_n(ISO) (cam2_fitness_v2.csv)
All in a Gaussian-cascade "gap" form: sigma_add = sqrt(sigma_target^2 - sigma_ref^2), so the
synthesized frame reproduces the target condition's measured blur/noise starting from the reference.

Validation here: (A) the blur operator HITS a specified MTF50 (e-SFR self-consistency);
                 (B) per-condition degradation table; (C) save one example.
"""
import os, glob
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
import cv2
from mtf_util import imread_u, esf_mtf, sigma_from_mtf50

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
SP01 = r"C:\Users\user\setting\code\tunnelscanning\01_tunnelscanning"
OUT = os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp"))
os.makedirs(OUT, exist_ok=True)


# ---------------- measured parameters ----------------
def load_params():
    df = pd.read_csv(os.path.join(SP01, "03_src", "data", "cam1_analysis.csv"))
    # per-ISO sensor-noise level (sigma_n) measured on the deployment domain
    cam2 = pd.read_csv(os.path.join(NRIQA, "04_data", "cam2_fitness_v2.csv")).dropna(subset=["v2_sigma_n"])
    noise_lut = cam2.groupby("iso")["v2_sigma_n"].mean().to_dict()
    return df, noise_lut


# ---------------- calibrated degradation ----------------
def degrade(clean, tgt, ref, noise_lut, rng):
    """clean: float image; tgt/ref: rows with mtf_h_mean, mtf_v_mean, iso. Returns degraded float image."""
    def gap(mt, mr):                      # added Gaussian std to go from ref-MTF to target-MTF
        st, sr = sigma_from_mtf50(mt), sigma_from_mtf50(mr)
        return float(np.sqrt(max(st * st - sr * sr, 0.0)))
    sh = gap(tgt["mtf_h_mean"], ref["mtf_h_mean"])      # horizontal (along-track motion + optics)
    sv = gap(tgt["mtf_v_mean"], ref["mtf_v_mean"])      # vertical (optics)
    blurred = gaussian_filter(clean, sigma=(sv, sh))    # (axis0=rows=vertical, axis1=cols=horizontal)
    nt = noise_lut[tgt["iso"]]; nr = noise_lut[ref["iso"]]
    n_add = float(np.sqrt(max(nt * nt - nr * nr, 0.0)))
    return blurred + rng.randn(*blurred.shape) * n_add, dict(sigma_h=sh, sigma_v=sv, noise_add=n_add)


# ---------------- run ----------------
df, noise_lut = load_params()
# pseudo-clean reference = operationally best condition: near distance + low ISO + low speed
# (chosen by acquisition logic / highest SP02 Q, NOT by raw MTF which is measurement-noisy and non-monotonic)
ref = df[(df.iso == df.iso.min()) & (df.dist == df.dist.min()) & (df.speed == df.speed.min())].iloc[0].to_dict()
print(f"reference (pseudo-clean) condition: {int(ref['speed'])}km {ref['dist']}m ISO{int(ref['iso'])}  "
      f"MTF_h={ref['mtf_h_mean']:.3f} MTF_v={ref['mtf_v_mean']:.3f} sigma_n={noise_lut[ref['iso']]:.2f}")
print("  (note: per-condition MTF is measurement-noisy/non-monotonic; a fitted MTF(dist,ISO) trend is a planned refinement)")

# load the reference chart ROI (vertical slant edge -> horizontal MTF)
CAM1 = glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", "60km_2.5m_ISO100", "MTF*", "frame_000102.png"))[0]
roi = imread_u(CAM1)[158:554, 937:1133].astype(np.float64)
_, _, m_clean = esf_mtf(roi)
print(f"reference ROI e-SFR MTF50 = {m_clean:.3f} cy/px\n")

# (A) operator hits a specified MTF50 (e-SFR self-consistency, horizontal blur only)
print("(A) blur operator hits target MTF50 (e-SFR metric):")
print(f"    {'target':>8s}{'sigma_add':>11s}{'measured':>10s}{'err':>8s}")
for m_t in [0.065, 0.055, 0.045, 0.035]:
    s_add = np.sqrt(max(sigma_from_mtf50(m_t)**2 - sigma_from_mtf50(m_clean)**2, 0.0))
    deg = gaussian_filter(roi, sigma=(0.0, s_add))
    _, _, m_meas = esf_mtf(deg)
    print(f"    {m_t:8.3f}{s_add:11.2f}{m_meas:10.3f}{(m_meas-m_t):+8.3f}")

# (B) per-condition degradation relative to the reference (sample conditions)
print("\n(B) calibrated degradation per condition (from reference):")
print(f"    {'cond':>18s}{'MTF_h':>7s}{'sig_h':>7s}{'sig_v':>7s}{'noise+':>8s}")
rng = np.random.RandomState(0)
sample = df[(df.speed == 60)].sort_values(["dist", "iso"])
for _, r in sample.iloc[::5].iterrows():
    _, info = degrade(roi, r.to_dict(), ref, noise_lut, rng)
    print(f"    {f'{int(r.speed)}k_{r.dist}m_ISO{int(r.iso)}':>18s}{r.mtf_h_mean:7.3f}"
          f"{info['sigma_h']:7.2f}{info['sigma_v']:7.2f}{info['noise_add']:8.2f}")

# (C) save one example: reference ROI degraded toward the worst condition
worst = df.loc[df["mtf_h_mean"].idxmin()].to_dict()
deg_img, info = degrade(roi, worst, ref, noise_lut, rng)
cv2.imwrite(os.path.join(OUT, "example_clean_roi.png"), np.clip(roi, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(OUT, "example_degraded_roi.png"), np.clip(deg_img, 0, 255).astype(np.uint8))
_, _, m_worst = esf_mtf(deg_img)
print(f"\n(C) example: ref ROI -> worst cond {int(worst['speed'])}k_{worst['dist']}m_ISO{int(worst['iso'])} "
      f"(sig_h={info['sigma_h']:.2f} sig_v={info['sigma_v']:.2f} noise+={info['noise_add']:.2f}) "
      f"-> degraded e-SFR MTF50 {m_worst:.3f}  (saved to 05_tmp/)")
