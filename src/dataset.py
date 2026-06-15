# -*- coding: utf-8 -*-
"""
SP03 training data — on-the-fly calibrated degradation of high-quality reference patches.

A bank of clean (best-condition) wall patches is degraded with blur/noise sampled from the
MEASURED envelope (blur from MTF range, noise from sigma_n(ISO) range). The model input carries
the measured degradation as a 2-channel prompt (blur level, noise level) alongside the degraded
image; the target is the clean patch.
"""
import os, glob
import numpy as np, cv2
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))

# measured envelope (from cam1_analysis MTF range -> added-blur sigma, and sigma_n(ISO) range)
BLUR_MAX = 3.0      # px Gaussian std (covers ref->worst gap)
NOISE_MAX = 3.0     # DN std (sigma_n ~0.8 .. 2.7 -> added noise up to ~2.6)


def imread_u(p, flag=cv2.IMREAD_GRAYSCALE):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


EVAL_FRAME = "frame_000102.png"   # held out for MTF-recovery evaluation


def clean_reference_frames():
    """Best-condition pseudo-clean references: cam2 wall texture + cam1 chart edges.
    Excludes the held-out eval frame so MTF-recovery evaluation is not on trained content."""
    fs = []
    for c in ["crack_d25_ISO100_V60", "crack_d25_ISO100_V80",
              "crack_d35_ISO100_V60", "crack_d35_ISO100_V80", "crack_d25_ISO200_V60"]:
        fs += glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam2", c, "*.png"))
    for c in ["60km_2.5m_ISO100", "60km_3.5m_ISO100", "80km_2.5m_ISO100", "80km_3.5m_ISO100"]:
        fs += [f for f in glob.glob(os.path.join(NRIQA, "04_data", "raw", "cam1", c, "frame_*.png"))
               if os.path.basename(f) != EVAL_FRAME]   # top-level chart frames only (not MTF/Results)
    return sorted(fs)


def build_patch_bank(frame_paths, psz=256, per_frame=12, mean_lo=20, mean_hi=235, std_min=6, seed=0):
    """Extract valid wall-texture patches from clean frames into an in-memory bank."""
    rng = np.random.RandomState(seed)
    bank = []
    for p in frame_paths:
        g = imread_u(p)
        if g is None:
            continue
        H, W = g.shape
        got = 0
        for _ in range(per_frame * 6):
            if got >= per_frame:
                break
            y = rng.randint(0, H - psz); x = rng.randint(0, W - psz)
            patch = g[y:y + psz, x:x + psz]
            if mean_lo < patch.mean() < mean_hi and patch.std() > std_min:
                bank.append(patch.copy()); got += 1
    return np.stack(bank).astype(np.float32)


class TunnelRestorationDataset(Dataset):
    def __init__(self, bank, length=2000, seed=0):
        self.bank = bank
        self.length = length
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean = self.bank[self.rng.randint(len(self.bank))]
        # sample calibrated degradation from the measured envelope
        sh = self.rng.uniform(0.2, BLUR_MAX)               # horizontal (motion+optics)
        sv = self.rng.uniform(0.2, BLUR_MAX * 0.8)         # vertical (optics, slightly less)
        nstd = self.rng.uniform(0.0, NOISE_MAX)
        deg = gaussian_filter(clean, sigma=(sv, sh))
        deg = deg + self.rng.randn(*deg.shape).astype(np.float32) * nstd
        # tensors, normalized to [0,1]; 2-channel measured degradation prompt (broadcast maps)
        x_img = np.clip(deg, 0, 255) / 255.0
        blur_map = np.full_like(x_img, (0.5 * (sh + sv)) / BLUR_MAX)
        noise_map = np.full_like(x_img, nstd / NOISE_MAX)
        x = np.stack([x_img, blur_map, noise_map], 0)
        y = (clean / 255.0)[None]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()
