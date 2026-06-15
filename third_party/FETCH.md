# Third-party components (not redistributed)

These baselines/tools are used by the paper but are **not** included in this repository.
Download them from their original sources under their own licenses, then point the scripts
in `src/` to the resulting paths.

## 1. Crack segmenter (task loss + detection / false-crack metrics)
- UNet-VGG16 crack-segmentation model of the DeepCrack family (public implementation,
  e.g. the `khanhha/crack_segmentation` repository).
- Place the checkpoint where the evaluation scripts expect it (see the `load_unet_vgg16`
  call in `src/eval_detection.py` / `src/eval_falsecrack.py`).
- Reference: Zou et al. (2019), *DeepCrack: Learning hierarchical convolutional features
  for crack detection*, IEEE TIP.

## 2. DiffBIR (generative diffusion baseline)
- Clone the official DiffBIR repository and download the v2.1 weights.
- Run at native resolution with an empty text prompt, e.g.:
  ```bash
  python inference.py --task denoise --upscale 1 --version v2.1 --captioner none \
    --pos_prompt "" --cldm_tiled --cldm_tile_size 512 \
    --input <degraded_dir> --output <out_dir> --device cuda
  ```
- The MERIT scripts read DiffBIR outputs from the `--output` directory.

## 3. Stable-Diffusion ×4 upscaler (SD-x4, generative baseline)
- Pulled automatically by `diffusers` from `stabilityai/stable-diffusion-x4-upscaler`
  (Hugging Face) on first use; run seeded for reproducibility
  (`torch.Generator(device).manual_seed(0)`).

All three are governed by their respective upstream licenses; this repository redistributes
none of their weights.
