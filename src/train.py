# -*- coding: utf-8 -*-
"""SP03 — train the measurement-fidelity NAFNet restorer on calibrated synthetic degradation."""
import os, time
import torch
from torch.utils.data import DataLoader
from dataset import clean_reference_frames, build_patch_bank, TunnelRestorationDataset
from model import NAFNetRestorer

CKPT = os.path.join(os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")), "restorer.pt")
dev = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 1500

frames = clean_reference_frames()
# cap per-domain for a fast first run: ~30 cam2 + ~40 cam1
cam2 = [f for f in frames if "cam2" in f][:30]
cam1 = [f for f in frames if "cam1" in f][:40]
t0 = time.time()
bank = build_patch_bank(cam2 + cam1, psz=256, per_frame=10)
print(f"patch bank {bank.shape}  ({time.time()-t0:.1f}s, cam2={len(cam2)} cam1={len(cam1)})")

ds = TunnelRestorationDataset(bank, length=STEPS * 8)
dl = DataLoader(ds, batch_size=8, num_workers=0)
net = NAFNetRestorer(width=48, n_blocks=10).to(dev)
print(f"NAFNet params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
lossf = torch.nn.L1Loss()
net.train()
t0 = time.time()
run = 0.0
for i, (x, y) in enumerate(dl):
    if i >= STEPS:
        break
    x, y = x.to(dev), y.to(dev)
    loss = lossf(net(x), y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    run += loss.item()
    if i % 200 == 0 or i == STEPS - 1:
        print(f"  step {i:4d}/{STEPS}  L1 {run/(min(i,199)+1):.4f}  ({time.time()-t0:.0f}s)")
        run = 0.0

torch.save({"state": net.state_dict(), "width": 48, "n_blocks": 10}, CKPT)
print(f"saved {CKPT}  ({time.time()-t0:.0f}s total)")
