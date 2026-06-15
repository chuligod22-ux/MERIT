# Pilot — Wiener deconvolution → MTF50 recovery (SP03 de-risk)

**Date**: 2026-06-08 · **Script**: `03_restoration/03_src/pilot_wiener_mtf.py`
**Setup**: real cam1 slant-edge chart ROI (60km_2.5m_ISO100, ROI (937,158,1133,554));
ISO-12233 e-SFR MTF (reused from `make_gt_figure.py`). Sharp baseline MTF50 = **0.077 cy/px**.

## Result

**(A) Blur-only (Gaussian σ=1.8, no noise)** — degraded MTF50 0.061 (79% of sharp)

| Wiener K | MTF50 | vs sharp | recovery of lost MTF |
|---:|---:|---:|---:|
| 3e-5 … 1e-3 | 0.077 | 100% | **100%** |
| 1e-2 | 0.077 | 99% | 96% |

→ Deconvolution **fully recovers objectively-measured MTF50**. The recovery axis works.

**(B) Blur+noise (σ=1.8, noise sd=3.0)** — degraded MTF50 0.045

| Wiener K | MTF50 | flat-noise sd |
|---:|---:|---:|
| 3e-5 | 0.018 | 84.3 |
| 1e-3 | 0.025 | 14.8 |
| 1e-2 | 0.032 | 4.7 |

→ Classical Wiener **cannot recover MTF under noise**: small K recovers high frequency but
explodes noise (sd 84) — corrupting both the image and the e-SFR measurement; large K
suppresses noise but over-smooths. The MTF↔noise trade-off is the wall.

## Interpretation (de-risk verdict: PASS)

1. The **objective MTF50-recovery validation axis is real and works** (proven noiseless).
2. **Noise-coupled blur traps classical deconvolution** in an MTF↔noise trade-off — quantified.
3. This **motivates the learned, noise-disentangled restorer** (SOTA Mamba + measured σ_n/nc_ratio
   conditioning): deconvolve without amplifying noise. The paper's motivation is now data-grounded.

**Caveat**: under heavy noise the e-SFR edge detection is itself corrupted (the 0.018 is partly a
measurement artifact) — the full method must couple denoising with deconvolution and harden the
MTF measurement on restored frames. This too points to the deep approach.

**Next**: implement the learned restorer; keep this classical Wiener as the baseline that shows the wall.
