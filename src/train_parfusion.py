# -*- coding: utf-8 -*-
"""SP03 B (retry) — train the parallel+fusion disentangled restorer (3.93M < single 4.95M)."""
import os, sys, time
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from torch.amp import autocast, GradScaler
from model_parfusion import ParallelFusionRestorer
from dataset import clean_reference_frames, build_patch_bank, BLUR_MAX, NOISE_MAX

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
CKPT = os.path.join(WEIGHTS, "restorer_parfusion.pt")
dev = "cuda"
STEPS, BATCH, LAM, ADEN = 6000, 4, 5.0, 1.0
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16

seg = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
for p in seg.parameters():
    p.requires_grad_(False)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def seg_prob(g01):
    return torch.sigmoid(seg((g01.repeat(1, 3, 1, 1) - MEAN) / STD))[:, :1]


def grad_loss(o, t):
    return (o[..., 1:] - o[..., :-1] - (t[..., 1:] - t[..., :-1])).abs().mean() \
         + (o[..., 1:, :] - o[..., :-1, :] - (t[..., 1:, :] - t[..., :-1, :])).abs().mean()


def fft_loss(o, t):
    d = torch.fft.rfft2(o, norm="ortho") - torch.fft.rfft2(t, norm="ortho")
    return (d.abs().detach() * (d.real**2 + d.imag**2)).mean()


crack = np.load(os.path.join(DATA, "crack_bank.npy")).astype(np.float32)
wall = build_patch_bank([f for f in clean_reference_frames()][:60], psz=448, per_frame=4).astype(np.float32)
bank = np.concatenate([crack, wall], 0)
seg_tgt = np.zeros_like(bank)
with torch.no_grad():
    for i in range(0, len(bank), 16):
        seg_tgt[i:i + 16] = seg_prob(torch.from_numpy(bank[i:i + 16] / 255.0)[:, None].float().to(dev))[:, 0].cpu().numpy()
print(f"bank {bank.shape} (crack={len(crack)} wall={len(wall)})", flush=True)

net = ParallelFusionRestorer(width=20).to(dev)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M (vs single 4.95M)", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
scaler = GradScaler("cuda")
l1f = torch.nn.L1Loss()
rs = np.random.RandomState(0)
net.train()
t0 = time.time(); acc = [0.0, 0.0, 0.0]
for i in range(STEPS):
    idx = rs.randint(0, len(bank), BATCH)
    cl = bank[idx] / 255.0; st = seg_tgt[idx]
    xs, blo, ys = [], [], []
    for c in cl:
        shb = rs.uniform(0.2, BLUR_MAX); svb = rs.uniform(0.2, BLUR_MAX * 0.8); nb = rs.uniform(0.0, NOISE_MAX)
        bo = gaussian_filter(c, (svb, shb))
        d = bo + rs.randn(448, 448) * (nb / 255.0)
        xs.append(np.stack([np.clip(d, 0, 1), np.full((448, 448), 0.5 * (shb + svb) / BLUR_MAX),
                            np.full((448, 448), nb / NOISE_MAX)], 0))
        blo.append(bo[None]); ys.append(c[None])
    x = torch.from_numpy(np.stack(xs)).float().to(dev)
    bo_t = torch.from_numpy(np.stack(blo)).float().to(dev)
    y = torch.from_numpy(np.stack(ys)).float().to(dev)
    stt = torch.from_numpy(st[:, None]).float().to(dev)
    opt.zero_grad()
    with autocast("cuda"):
        out, den = net(x)
    out, den = out.float(), den.float()
    l_den = l1f(den, bo_t)
    l_fid = l1f(out, y) + grad_loss(out, y) + 0.1 * fft_loss(out, y)
    l_task = l1f(seg_prob(out), stt)
    loss = l_fid + LAM * l_task + ADEN * l_den
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    for k, v in enumerate((l_fid, l_task, l_den)):
        acc[k] += v.item()
    if i % 500 == 0 or i == STEPS - 1:
        n = (i % 500) + 1
        print(f"  step {i:4d}/{STEPS}  fid {acc[0]/n:.4f} task {acc[1]/n:.4f} den {acc[2]/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        acc = [0.0, 0.0, 0.0]
    if i > 0 and i % 1500 == 0:
        torch.save({"state": net.state_dict(), "width": 20}, CKPT)

torch.save({"state": net.state_dict(), "width": 20}, CKPT)
print(f"DONE saved {CKPT} ({time.time()-t0:.0f}s)", flush=True)
