# 1. Third-party components

Third-party repositories, datasets, and pretrained weights are not redistributed in this
package. Obtain them from their official sources and retain the upstream license.

The revised comparison includes author-implemented classical methods and the cited public
implementations of NAFNet, MPRNet, Restormer, DiffBIR, and Stable-Diffusion x4. Detector
families include the cited CNN baselines, SegFormer-B0, and the documented Mamba2 U-Net
hybrid. Exact model IDs, commits, and hashes are recorded in the corresponding frozen
protocol when retained.

## 1.1 Stable Diffusion x4

The second-revision fixed-chart comparison used `stabilityai/stable-diffusion-x4-upscaler` at Hub revision `572c99286543a273bfd17fac263db5a77be12c4c`. Obtain that snapshot under its upstream licence to repeat inference. The run record includes the fixed input and output SHA-256 values, seed, scheduler, guidance, noise level and conversion of the 4× output to the original dimensions. Scoring the archived output only requires the public chart files. [Official model snapshot](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler/tree/572c99286543a273bfd17fac263db5a77be12c4c) · [Second-revision run record](../round2/optics/chart/sdx4_run.json) · [Replay instructions](../round2/optics/README.md)

The separate October revision recorded the model identifier only. Its original Hub revision remains unknown. [October revision materials](https://github.com/chuligod22-ux/MERIT/releases/tag/v2.0.0-revision)
