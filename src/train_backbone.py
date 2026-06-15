# -*- coding: utf-8 -*-
"""
SP03 — unified trainer for the backbone-agnosticism demonstration.

Trains ANY backbone under the MERIT framework (mode=on) or as a plain baseline (mode=off):
  mode=on  : 3ch input (degraded + measured blur/noise prompt) + fidelity (L1+grad+focal-freq)
             + lambda * segmenter-in-loop task loss   == the full MERIT framework
  mode=off : 1ch input (degraded only) + plain L1 only == NO framework (control)

The SAME degradation, patch bank, and schedule are used for every (backbone, mode), so differences
are attributable to backbone (across ON rows) and to framework (ON vs OFF). From scratch.

Usage:  uv run python train_backbone.py --backbone {nafnet|plainunet|restormer} --mode {on|off} --steps N
Saves:  04_data/restorer_<backbone>_<mode>.pt   with {state, arch, width, in_ch, mode}
"""
import os, sys, time, argparse
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from dataset import clean_reference_frames, build_patch_bank, BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
from model_backbones import PlainUNet, RestormerLite, MambaLite

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
dev = "cuda"

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", required=True, choices=["nafnet", "plainunet", "restormer", "mamba"])
ap.add_argument("--mode", required=True, choices=["on", "off"])
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lam", type=float, default=5.0)
ap.add_argument("--lr", type=float, default=2e-4)
args = ap.parse_args()

IN_CH = 3 if args.mode == "on" else 1
WIDTHS = {"nafnet": 32, "plainunet": 48, "restormer": 48, "mamba": 128}
W = WIDTHS[args.backbone]


def make_net():
    if args.backbone == "nafnet":
        return NAFNetUNet(in_ch=IN_CH, width=W)
    if args.backbone == "plainunet":
        return PlainUNet(in_ch=IN_CH, width=W)
    if args.backbone == "mamba":
        return MambaLite(in_ch=IN_CH, width=W)
    return RestormerLite(in_ch=IN_CH, width=W)


CKPT = os.path.join(WEIGHTS, f"restorer_{args.backbone}_{args.mode}.pt")

# ---- segmenter (only needed for mode=on task loss) ----
seg = None
if args.mode == "on":
    CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
    from utils import load_unet_vgg16
    seg = load_unet_vgg16(os.path.join(CSEG, "models", "model_unet_vgg16_best.pt")).cuda().eval()
    for p in seg.parameters():
        p.requires_grad_(False)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def seg_prob(g):
        return torch.sigmoid(seg((g.repeat(1, 3, 1, 1) - MEAN) / STD))[:, :1]

# ---- patch bank (crack + wall), shared across all runs ----
crack = np.load(os.path.join(DATA, "crack_bank.npy")).astype(np.float32)
frames = clean_reference_frames()
wall = build_patch_bank([f for f in frames][:60], psz=448, per_frame=4).astype(np.float32)
bank = np.concatenate([crack, wall], 0)
print(f"[{args.backbone}/{args.mode}] bank crack={len(crack)} wall={len(wall)} in_ch={IN_CH} width={W}", flush=True)

seg_tgt = None
if args.mode == "on":
    seg_tgt = np.zeros_like(bank)
    with torch.no_grad():
        for i in range(0, len(bank), 16):
            g = torch.from_numpy(bank[i:i + 16] / 255.0)[:, None].float().to(dev)
            seg_tgt[i:i + 16] = seg_prob(g)[:, 0].cpu().numpy()

net = make_net().to(dev)
print(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M", flush=True)


def grad_loss(o, t):
    return (o[..., 1:] - o[..., :-1] - (t[..., 1:] - t[..., :-1])).abs().mean() \
         + (o[..., 1:, :] - o[..., :-1, :] - (t[..., 1:, :] - t[..., :-1, :])).abs().mean()


def fft_loss(o, t):
    d = torch.fft.rfft2(o, norm="ortho") - torch.fft.rfft2(t, norm="ortho")
    return (d.abs().detach() * (d.real**2 + d.imag**2)).mean()


opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
scaler = torch.cuda.amp.GradScaler()
l1f = torch.nn.L1Loss()
rs = np.random.RandomState(0)
net.train()
t0 = time.time(); acc = [0.0, 0.0]
for i in range(args.steps):
    idx = rs.randint(0, len(bank), args.batch)
    cl = bank[idx] / 255.0
    xs, ys = [], []
    for c in cl:
        shb = rs.uniform(0.2, BLUR_MAX); svb = rs.uniform(0.2, BLUR_MAX * 0.8); nb = rs.uniform(0.0, NOISE_MAX)
        d = gaussian_filter(c, (svb, shb)) + rs.randn(448, 448) * (nb / 255.0)
        if IN_CH == 3:
            xs.append(np.stack([np.clip(d, 0, 1), np.full((448, 448), 0.5 * (shb + svb) / BLUR_MAX),
                                np.full((448, 448), nb / NOISE_MAX)], 0))
        else:
            xs.append(np.clip(d, 0, 1)[None])
        ys.append(c[None])
    x = torch.from_numpy(np.stack(xs)).float().to(dev)
    y = torch.from_numpy(np.stack(ys)).float().to(dev)
    with torch.cuda.amp.autocast():
        out = net(x)
        if args.mode == "on":
            fid = l1f(out, y) + grad_loss(out, y) + 0.1 * fft_loss(out, y)
            stt = torch.from_numpy(seg_tgt[idx][:, None]).float().to(dev)
            task = l1f(seg_prob(out.float()), stt)
            loss = fid + args.lam * task
        else:
            fid = l1f(out, y); task = torch.zeros((), device=dev); loss = fid
    opt.zero_grad(); scaler.scale(loss).backward()
    scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # stabilise (esp. SSM scan)
    scaler.step(opt); scaler.update(); sched.step()
    acc[0] += float(fid); acc[1] += float(task)
    if i % 500 == 0 or i == args.steps - 1:
        n = (i % 500) + 1
        print(f"  step {i:4d}/{args.steps}  fid {acc[0]/n:.4f}  task {acc[1]/n:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        acc = [0.0, 0.0]

torch.save({"state": net.state_dict(), "arch": args.backbone, "width": W, "in_ch": IN_CH, "mode": args.mode}, CKPT)
print(f"DONE saved {CKPT}  ({time.time()-t0:.0f}s)", flush=True)
