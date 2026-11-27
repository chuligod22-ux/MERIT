"""Re-evaluate SD-x4 on the archived degraded chart sent to DiffBIR."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
from diffusers import StableDiffusionUpscalePipeline
import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
REVISION = "572c99286543a273bfd17fac263db5a77be12c4c"
MODEL_ID = "stabilityai/stable-diffusion-x4-upscaler"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.resolve() == HERE:
        parser.error("Choose a separate output directory to preserve the archived run")
    if args.snapshot.name != REVISION:
        raise ValueError(f"Expected local Hub revision {REVISION}")
    source = HERE / "common_degraded.png"
    if sha(source) != "9705bf8d64778ba6ac901158c24fcc502c486ebd97e557e75202b6ba59cef89c":
        raise ValueError("The input differs from the archived fixed-condition chart")
    with Image.open(source) as image:
        rgb = image.convert("RGB")
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        args.snapshot, torch_dtype=torch.float16, local_files_only=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    seed = 5
    with torch.inference_mode():
        output = pipeline(
            prompt="", image=rgb, num_inference_steps=50,
            guidance_scale=0.0, noise_level=20,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
    gray = cv2.resize(np.asarray(output.convert("L")), rgb.size, interpolation=cv2.INTER_AREA)
    dst = args.output_dir / "sdx4_chart.png"
    Image.fromarray(gray).save(dst)
    meta = {
        "model_id": MODEL_ID, "revision": REVISION, "seed": seed,
        "steps": 50, "guidance_scale": 0, "noise_level": 20, "prompt": "",
        "input_sha256": sha(source), "output_sha256": sha(dst),
        "chart_size": list(rgb.size), "native_output_size": list(output.size),
        "scheduler": type(pipeline.scheduler).__name__,
        "grayscale_downsampling": "PIL L then OpenCV INTER_AREA to input dimensions",
    }
    (args.output_dir / "sdx4_run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
