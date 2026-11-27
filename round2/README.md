# MERIT reproducibility package for the second revision

Version **v2.1.0-round2** provides the measurements, scoring inputs and analysis scripts for the second revision of *MERIT: Measurement-Fidelity Restoration for High-Speed Tunnel-Lining Crack Inspection*. The four-tunnel field results concern these sampled tunnels. Apparent crack width is referenced to the expert-mask chord and expressed in image pixels.

## 1. Verify the download

Unzip the release asset, change to the directory containing this README, and run the following commands with Python 3.10 or later. The first command uses the standard library to check every packaged file against its SHA-256 digest. The remaining commands also reproduce the statistical and chart checks on a CPU.

```bash
python verify.py
python -m pip install numpy scipy pillow opencv-python-headless
python verify.py --analyses
```

The optical guide describes the additional dependencies and upstream weights for rescoring LPIPS or rerunning SD-x4. Each module also runs independently.

## 2. Find the evidence

| Research question | Start here | What can be checked |
|---|---|---|
| Does restoration change detector response on acquired and degraded images? | [Detection](detection/README.md) | Per-span scores, routing, matched loss ablations, degradation severity, and bootstrap intervals |
| How large is the framework effect relative to the choice of backbone? | [Factorial rows](factorial/confirmation_rows.csv) and [script](factorial/verify.py) | Five point contrasts from 336 previously released rows, weighting the four tunnels equally. The archived confidence limits are supplied in estimates.json. |
| Does the optical gate preserve chart resolution on held-out acquisition conditions? | [Optics](optics/README.md) | Frame and ROI measurements from 479 controlled frames, held-out model selection, and comparison with always applying restoration |
| How do the methods score on the same input chart? | [Fixed chart](optics/chart/fixed_chart.csv) | The common decoded 8-bit input, fixed scoring regions, outputs, provenance, and Figures 5–7. The current SD-x4 result is 34.61 dB. |
| How does apparent-width error change, and which cross-sections are measurable? | [Field width and sampling](field_width/README.md) | 1,057 frozen cross-sections, 741 paired measurements, width bands, peak location, mask sensitivity, and sampling across 14 spans |
| Where are the other updated plots? | [Figure 4](figures/fig4.pdf), [Figure 13](figures/fig13.pdf), [Figure 15](figures/fig15.pdf) | Resolution and fidelity, apparent field width, and sensitivity to added degradation |

## 3. Acquisition inputs, models and access

The companion releases provide the [cam1 optical crops](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-cam1-optical-crops-v2), [cam2 controlled crops](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-cam2-crops-v2), [printed-width targets](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-width-inputs-v2), [six author-trained restoration checkpoints and one frozen detector](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-models-essential-v2), and [previously released analyses](https://github.com/chuligod22-ux/MERIT/releases/tag/revision-analysis-v2).

The public field tables contain derived measurements. Access to the 1,703 source tiles and their expert masks requires owner approval and an agreement on data use, as described in [the access instructions](ACCESS_REQUEST.md). The optical chart images are from controlled acquisitions. Obtain third-party models and weights from their original sources. The corrected fixed-input chart uses [the recorded SD-x4 model revision](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/572c99286543a273bfd17fac263db5a77be12c4c).

Author-developed code is released under the [MIT License](LICENSE-CODE). The [public data license](LICENSE-DATA) covers the public measurements and author-supplied figures. Cite the specific [MERIT release](https://github.com/chuligod22-ux/MERIT/releases/tag/v2.1.0-round2) used in your analysis.
