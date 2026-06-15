# -*- coding: utf-8 -*-
"""SP03 B — retrain with L1 + gradient + focal-frequency loss (recover MTF/high-freq, no GAN hallucination)."""
import os, time
import torch
from torch.utils.data import DataLoader
from dataset import clean_reference_frames, build_patch_bank, TunnelRestorationDataset
from model import NAFNetRestorer

CKPT = os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer_freq.pt")
dev = "cuda"
STEPS = 1800
LG, LF = 1.0, 0.10            # gradient, focal-frequency weights


def grad_loss(o, t):
    return (o[..., 1:] - o[..., :-1] - (t[..., 1:] - t[..., :-1])).abs().mean() \
         + (o[..., 1:, :] - o[..., :-1, :] - (t[..., 1:, :] - t[..., :-1, :])).abs().mean()


def fft_loss(o, t):
    d = torch.fft.rfft2(o, norm="ortho") - torch.fft.rfft2(t, norm="ortho")
    w = d.abs().detach()                       # focal frequency weighting
    return (w * (d.real**2 + d.imag**2)).mean()


frames = clean_reference_frames()
cam2 = [f for f in frames if "cam2" in f][:30]; cam1 = [f for f in frames if "cam1" in f][:40]
bank = build_patch_bank(cam2 + cam1, psz=256, per_frame=10)
print(f"patch bank {bank.shape}  cam2={len(cam2)} cam1={len(cam1)}")

dl = DataLoader(TunnelRestorationDataset(bank, length=STEPS * 8), batch_size=8, num_workers=0)
net = NAFNetRestorer(width=48, n_blocks=10).to(dev)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  | loss = L1 + {LG}*grad + {LF}*fft")
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
l1f = torch.nn.L1Loss()
net.train()
t0 = time.time(); acc = [0, 0, 0]
for i, (x, y) in enumerate(dl):
    if i >= STEPS:
        break
    x, y = x.to(dev), y.to(dev)
    o = net(x)
    l1, lg, lf = l1f(o, y), grad_loss(o, y), fft_loss(o, y)
    loss = l1 + LG * lg + LF * lf
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    for k, v in enumerate((l1, lg, lf)):
        acc[k] += v.item()
    if i % 300 == 0 or i == STEPS - 1:
        n = (i % 300) + 1
        print(f"  step {i:4d}  L1 {acc[0]/n:.4f}  grad {acc[1]/n:.4f}  fft {acc[2]/n:.4f}  ({time.time()-t0:.0f}s)")
        acc = [0, 0, 0]

torch.save({"state": net.state_dict(), "width": 48, "n_blocks": 10}, CKPT)
print(f"saved {CKPT}  ({time.time()-t0:.0f}s)")
