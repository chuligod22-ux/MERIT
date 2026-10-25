# Third-party components

Third-party repositories, datasets, and pretrained weights are not redistributed in this
package. Obtain them from their official sources and retain the upstream license.

The revised comparison includes author-implemented classical methods and the cited public
implementations of NAFNet, MPRNet, Restormer, DiffBIR, and Stable-Diffusion x4. Detector
families include the cited CNN baselines, SegFormer-B0, and the documented Mamba2 U-Net
hybrid. Exact model IDs, commits, and hashes are recorded in the corresponding frozen
protocol when retained.

For SD-x4, the canonical model identifier is:

```text
stabilityai/stable-diffusion-x4-upscaler
```

The exact Hugging Face Hub revision used by the historical run was not retained and cannot
be reconstructed from the available record. Users should record the resolved revision in
any new execution.
