# -*- coding: utf-8 -*-
"""
SP03 D — task-oriented retraining: a frozen crack segmenter in the loop.

Loss = L1 + grad + focal-frequency (fidelity)  +  lambda * L1(seg(restored), seg(clean)).
The detection term preserves crack-detectability where it exists (crack patches) AND suppresses
invented cracks where it does not (wall patches, seg(clean) ~ 0) — aligned with the fidelity thesis.
"""
import os, sys, time
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from model import NAFNetUNet
from dataset import clean_reference_frames, build_patch_bank, BLUR_MAX, NOISE_MAX

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
CKPT = os.path.join(WEIGHTS, "restorer_task.pt")
INIT = os.path.join(WEIGHTS, "restorer_big.pt")            # warm-start from the big model
dev = "cuda"
STEPS, BATCH, LAM = 2500, 4, 5.0
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16

seg = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
for p in seg.parameters():
    p.requires_grad_(False)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def seg_prob(gray01):                                   # gray01: [B,1,H,W] in [0,1] -> crack prob [B,1,H,W]
    rgb = (gray01.repeat(1, 3, 1, 1) - MEAN) / STD
    return torch.sigmoid(seg(rgb))[:, :1]


# ---- 448 patch bank: crack patches + wall patches, with cached clean segmenter maps ----
crack = np.load(os.path.join(DATA, "crack_bank.npy")).astype(np.float32)         # (Nc,448,448)
frames = clean_reference_frames()
wall = build_patch_bank([f for f in frames][:60], psz=448, per_frame=4).astype(np.float32)  # (Nw,448,448)
bank = np.concatenate([crack, wall], 0)
print(f"task bank: crack={len(crack)} wall={len(wall)} total={len(bank)}", flush=True)

# precompute clean segmenter targets (no grad)
seg_tgt = np.zeros_like(bank)
with torch.no_grad():
    for i in range(0, len(bank), 16):
        g = torch.from_numpy(bank[i:i + 16] / 255.0)[:, None].float().to(dev)
        seg_tgt[i:i + 16] = seg_prob(g)[:, 0].cpu().numpy()
print(f"cached clean seg targets; mean crack-prob over bank = {seg_tgt.mean():.4f}", flush=True)

net = NAFNetUNet(width=32).to(dev)
net.load_state_dict(torch.load(INIT, map_location=dev)["state"])    # warm start


def grad_loss(o, t):
    return (o[..., 1:] - o[..., :-1] - (t[..., 1:] - t[..., :-1])).abs().mean() \
         + (o[..., 1:, :] - o[..., :-1, :] - (t[..., 1:, :] - t[..., :-1, :])).abs().mean()


def fft_loss(o, t):
    d = torch.fft.rfft2(o, norm="ortho") - torch.fft.rfft2(t, norm="ortho")
    return (d.abs().detach() * (d.real**2 + d.imag**2)).mean()


opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
l1f = torch.nn.L1Loss()
rs = np.random.RandomState(0)
net.train()
t0 = time.time(); acc = [0.0, 0.0]
for i in range(STEPS):
    idx = rs.randint(0, len(bank), BATCH)
    cl = bank[idx] / 255.0
    st = seg_tgt[idx]
    xs, ys = [], []
    for c in cl:
        shb = rs.uniform(0.2, BLUR_MAX); svb = rs.uniform(0.2, BLUR_MAX * 0.8); nb = rs.uniform(0.0, NOISE_MAX)
        d = gaussian_filter(c, (svb, shb)) + rs.randn(448, 448) * (nb / 255.0)
        xs.append(np.stack([np.clip(d, 0, 1), np.full((448, 448), 0.5 * (shb + svb) / BLUR_MAX),
                            np.full((448, 448), nb / NOISE_MAX)], 0))
        ys.append(c[None])
    x = torch.from_numpy(np.stack(xs)).float().to(dev)
    y = torch.from_numpy(np.stack(ys)).float().to(dev)
    stt = torch.from_numpy(st[:, None]).float().to(dev)
    out = net(x)
    fid = l1f(out, y) + grad_loss(out, y) + 0.1 * fft_loss(out, y)
    task = l1f(seg_prob(out), stt)
    loss = fid + LAM * task
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    acc[0] += fid.item(); acc[1] += task.item()
    if i % 250 == 0 or i == STEPS - 1:
        n = (i % 250) + 1
        print(f"  step {i:4d}/{STEPS}  fid {acc[0]/n:.4f}  task {acc[1]/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        acc = [0.0, 0.0]
    if i > 0 and i % 1000 == 0:
        torch.save({"state": net.state_dict(), "arch": "unet", "width": 32}, CKPT)

torch.save({"state": net.state_dict(), "arch": "unet", "width": 32}, CKPT)
print(f"DONE saved {CKPT}  ({time.time()-t0:.0f}s)", flush=True)
