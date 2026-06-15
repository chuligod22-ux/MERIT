# -*- coding: utf-8 -*-
"""SP03 heavy-investment training — encoder-decoder NAFNet, AMP, larger data, L1+grad+focal-frequency."""
import os, time
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from dataset import clean_reference_frames, build_patch_bank, TunnelRestorationDataset
from model import NAFNetUNet

CKPT = os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer_big.pt")
dev = "cuda"
STEPS, BATCH = 8000, 8
LG, LF = 1.0, 0.10


def grad_loss(o, t):
    return (o[..., 1:] - o[..., :-1] - (t[..., 1:] - t[..., :-1])).abs().mean() \
         + (o[..., 1:, :] - o[..., :-1, :] - (t[..., 1:, :] - t[..., :-1, :])).abs().mean()


def fft_loss(o, t):
    d = torch.fft.rfft2(o, norm="ortho") - torch.fft.rfft2(t, norm="ortho")
    w = d.abs().detach()
    return (w * (d.real**2 + d.imag**2)).mean()


frames = clean_reference_frames()
cam2 = [f for f in frames if "cam2" in f][:40]; cam1 = [f for f in frames if "cam1" in f][:60]
t0 = time.time()
bank = build_patch_bank(cam2 + cam1, psz=256, per_frame=16)
print(f"patch bank {bank.shape}  cam2={len(cam2)} cam1={len(cam1)}  ({time.time()-t0:.0f}s)", flush=True)

dl = DataLoader(TunnelRestorationDataset(bank, length=STEPS * BATCH), batch_size=BATCH, num_workers=0)
net = NAFNetUNet(width=32).to(dev)
print(f"NAFNetUNet params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  AMP  steps {STEPS} bs {BATCH}", flush=True)
opt = torch.optim.AdamW(net.parameters(), lr=4e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
scaler = GradScaler("cuda")
l1f = torch.nn.L1Loss()
net.train()
t0 = time.time(); acc = [0.0, 0.0, 0.0]
for i, (x, y) in enumerate(dl):
    if i >= STEPS:
        break
    x, y = x.to(dev), y.to(dev)
    opt.zero_grad()
    with autocast("cuda"):
        out = net(x)
    o = out.float()
    l1, lg, lf = l1f(o, y), grad_loss(o, y), fft_loss(o, y)
    loss = l1 + LG * lg + LF * lf
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    for k, v in enumerate((l1, lg, lf)):
        acc[k] += v.item()
    if i % 500 == 0 or i == STEPS - 1:
        n = (i % 500) + 1
        print(f"  step {i:5d}/{STEPS}  L1 {acc[0]/n:.4f} grad {acc[1]/n:.4f} fft {acc[2]/n:.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        acc = [0.0, 0.0, 0.0]
    if i > 0 and i % 2000 == 0:
        torch.save({"state": net.state_dict(), "arch": "unet", "width": 32}, CKPT)
        print(f"  [checkpoint @ {i}]", flush=True)

torch.save({"state": net.state_dict(), "arch": "unet", "width": 32}, CKPT)
print(f"DONE saved {CKPT}  ({time.time()-t0:.0f}s)", flush=True)
