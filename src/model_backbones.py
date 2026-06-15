# -*- coding: utf-8 -*-
"""
SP03 — architecturally-distinct backbones for the backbone-agnosticism demonstration.

Every backbone shares the MERIT interface:
    input  : in_ch  (3 = [degraded, blur-prompt, noise-prompt] for framework-ON; 1 = degraded for OFF)
    output : 1ch restored grayscale via a GLOBAL RESIDUAL on the degraded channel x[:, :1]
so the same training/eval code drives any of them. None use custom CUDA (run on Blackwell/torch2.11).

  PlainUNet     : pure-convolutional U-Net (Conv-GroupNorm-GELU), NO gating / NO attention
  RestormerLite : transformer — MDTA channel self-attention + GDFN (Restormer-style, full-res tractable)
NAFNet lives in model.py (simplified-attention CNN). These three span conv / transformer paradigms.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== PlainUNet (pure conv) =====================
class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())

    def forward(self, x):
        return self.net(x)


class PlainUNet(nn.Module):
    """Textbook convolutional encoder-decoder. No SimpleGate, no channel attention, no self-attention."""
    def __init__(self, in_ch=3, out_ch=1, width=48):
        super().__init__()
        w = width
        self.e1 = ConvBlock(in_ch, w)
        self.e2 = ConvBlock(w, 2 * w)
        self.e3 = ConvBlock(2 * w, 4 * w)
        self.mid = ConvBlock(4 * w, 8 * w)
        self.d3 = ConvBlock(8 * w + 4 * w, 4 * w)
        self.d2 = ConvBlock(4 * w + 2 * w, 2 * w)
        self.d1 = ConvBlock(2 * w + w, w)
        self.out = nn.Conv2d(w, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        m = self.mid(self.pool(s3))
        up = lambda f, s: F.interpolate(f, size=s.shape[-2:], mode="bilinear", align_corners=False)
        d = self.d3(torch.cat([up(m, s3), s3], 1))
        d = self.d2(torch.cat([up(d, s2), s2], 1))
        d = self.d1(torch.cat([up(d, s1), s1], 1))
        return torch.clamp(x[:, :1] + self.out(d), 0.0, 1.0)


# ===================== RestormerLite (transformer) =====================
class MDTA(nn.Module):
    """Multi-Dconv head transposed attention (Restormer): self-attention across CHANNELS (tractable at full res)."""
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.temp = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.qkv_dw = nn.Conv2d(3 * c, 3 * c, 3, padding=1, groups=3 * c)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        h = self.heads
        q = q.reshape(B, h, C // h, H * W)
        k = k.reshape(B, h, C // h, H * W)
        v = v.reshape(B, h, C // h, H * W)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temp        # (B,h,C/h,C/h)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)
        return self.proj(out)


class GDFN(nn.Module):
    """Gated-Dconv feed-forward network (Restormer)."""
    def __init__(self, c, expand=2.0):
        super().__init__()
        hid = int(c * expand)
        self.pw1 = nn.Conv2d(c, 2 * hid, 1)
        self.dw = nn.Conv2d(2 * hid, 2 * hid, 3, padding=1, groups=2 * hid)
        self.pw2 = nn.Conv2d(hid, c, 1)

    def forward(self, x):
        a, b = self.dw(self.pw1(x)).chunk(2, dim=1)
        return self.pw2(F.gelu(a) * b)


class LN2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(c)); self.b = nn.Parameter(torch.zeros(c)); self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True); var = (x - mu).pow(2).mean(1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.w[None, :, None, None] + self.b[None, :, None, None]


class TBlock(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n1 = LN2d(c); self.attn = MDTA(c, heads)
        self.n2 = LN2d(c); self.ffn = GDFN(c)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class RestormerLite(nn.Module):
    """Self-attention restorer (Restormer-style MDTA+GDFN), hierarchical: attention runs at H/4 (tractable).

    intro conv -> 2 stride-2 downsamples -> transformer body at 1/4 res -> 2 pixelshuffle upsamples -> out.
    Genuinely a transformer (channel self-attention via MDTA) and memory-feasible at 448x448.
    """
    def __init__(self, in_ch=3, out_ch=1, width=48, n_blocks=6, heads=4):
        super().__init__()
        w = width
        self.intro = nn.Conv2d(in_ch, w, 3, padding=1)
        self.down1 = nn.Conv2d(w, 2 * w, 2, stride=2)         # H/2
        self.down2 = nn.Conv2d(2 * w, 4 * w, 2, stride=2)     # H/4
        self.body = nn.Sequential(*[TBlock(4 * w, heads) for _ in range(n_blocks)])
        self.up2 = nn.Sequential(nn.Conv2d(4 * w, 8 * w, 1), nn.PixelShuffle(2))   # ->2w, H/2
        self.up1 = nn.Sequential(nn.Conv2d(2 * w, 4 * w, 1), nn.PixelShuffle(2))   # ->w,  H
        self.out = nn.Conv2d(w, out_ch, 3, padding=1)

    def forward(self, x):
        f0 = self.intro(x)
        f1 = self.down1(f0)
        f2 = self.down2(f1)
        f2 = self.body(f2)
        f = self.up2(f2) + f1
        f = self.up1(f) + f0
        return torch.clamp(x[:, :1] + self.out(f), 0.0, 1.0)


# ===================== MambaLite (state-space / selective scan) =====================
def selective_scan_par(u, dt, A, B, C, D):
    """Parallel selective scan (S6) in PURE PyTorch via cumprod/cumsum (no custom CUDA, runs on sm_120).
    u,dt: (b,d,L)   A: (d,n)   B,C: (b,n,L)   D: (d,)   ->  y: (b,d,L)
    Recurrence h_t = dA_t * h_{t-1} + dB_t * u_t ;  y_t = sum_n C_t h_t + D u_t .
    Computed in fp32 even under AMP: the exp/cumprod recurrence overflows in fp16 (-> NaN, skipped steps).
    """
    with torch.autocast(device_type="cuda", enabled=False):
        u, dt, A, B, C, D = (t.float() for t in (u, dt, A, B, C, D))
        d, L = u.shape[1], u.shape[2]
        dt = dt.clamp(max=8.0)
        logdA = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(2)                  # (b,d,L,n) <= 0
        cumlogA = torch.cumsum(logdA, dim=2)                                    # (b,d,L,n) decreasing
        dBu = dt.unsqueeze(-1) * B.permute(0, 2, 1).unsqueeze(1) * u.unsqueeze(-1)   # (b,d,L,n)
        # STABLE scan: h_t = sum_{s<=t} W[t,s] dBu_s,  W[t,s] = exp(cumlogA_t - cumlogA_s) in [0,1].
        diff = cumlogA.unsqueeze(3) - cumlogA.unsqueeze(2)                      # (b,d,L_t,L_s,n)
        mask = torch.tril(torch.ones(L, L, device=u.device)).view(1, 1, L, L, 1)
        W = torch.exp(diff) * mask                                             # entries in [0,1], no overflow
        h = torch.einsum("bdtsn,bdsn->bdtn", W, dBu)                           # (b,d,L,n)
        y = torch.einsum("bdtn,bnt->bdt", h, C)                                # (b,d,L)
        return y + u * D.view(1, d, 1)


def _scan_dir(u, dt, A, B, C, D, flip):
    if flip:
        u, dt, B, C = u.flip(-1), dt.flip(-1), B.flip(-1), C.flip(-1)
    y = selective_scan_par(u, dt, A, B, C, D)
    return y.flip(-1) if flip else y


class SS2D(nn.Module):
    """2D selective scan: horizontal (both dirs) + vertical (both dirs), input-dependent (selective)."""
    def __init__(self, d, n=16):
        super().__init__()
        self.d, self.n = d, n
        self.in_proj = nn.Conv2d(d, d, 1)
        self.dt_proj = nn.Conv2d(d, d, 1)
        self.bc_proj = nn.Conv2d(d, 2 * n, 1)            # shared B,C across the directional scans
        self.A = nn.Parameter(-torch.rand(d, n) - 0.5)   # negative (stable); learned
        self.D = nn.Parameter(torch.ones(d))
        self.out_proj = nn.Conv2d(d, d, 1)

    def _scan_axis(self, x):
        # x: (B,d,H,W) -> scan along W (both directions), return (B,d,H,W)
        Bb, d, H, W = x.shape
        u = x.reshape(Bb * H, d, W)
        dt = torch.nn.functional.softplus(self.dt_proj(x)).reshape(Bb * H, d, W)
        bc = self.bc_proj(x).reshape(Bb * H, 2 * self.n, W)
        Bm, Cm = bc[:, :self.n], bc[:, self.n:]
        A = -torch.nn.functional.softplus(self.A)        # ensure negative
        y = _scan_dir(u, dt, A, Bm, Cm, self.D, False) + _scan_dir(u, dt, A, Bm, Cm, self.D, True)
        return y.reshape(Bb, d, H, W)

    def forward(self, x):
        x = self.in_proj(x)
        yh = self._scan_axis(x)                          # horizontal
        yv = self._scan_axis(x.transpose(-1, -2)).transpose(-1, -2)   # vertical
        return self.out_proj(yh + yv)


class VSSBlock(nn.Module):
    def __init__(self, d, n=16):
        super().__init__()
        self.n1 = LN2d(d); self.ss = SS2D(d, n)
        self.n2 = LN2d(d); self.ffn = GDFN(d)
        # zero-init layerscale: block starts as identity, scan/ffn contribution ramps up (stabilises SSM training)
        self.g1 = nn.Parameter(torch.zeros(1, d, 1, 1))
        self.g2 = nn.Parameter(torch.zeros(1, d, 1, 1))

    def forward(self, x):
        x = x + self.g1 * self.ss(self.n1(x))
        return x + self.g2 * self.ffn(self.n2(x))


class MambaLite(nn.Module):
    """State-space (selective-scan / Mamba-class) restorer in pure PyTorch. Selective scan runs at 1/8 res
    with a constant channel width (keeps the (B*H, C, L, N) scan tensors small enough for sm_120)."""
    def __init__(self, in_ch=3, out_ch=1, width=112, n_blocks=5, n_state=4, n_down=4):
        super().__init__()
        C = width
        self.intro = nn.Conv2d(in_ch, C, 3, padding=1)
        self.down = nn.ModuleList([nn.Conv2d(C, C, 2, stride=2) for _ in range(n_down)])   # H -> H/16
        self.body = nn.Sequential(*[VSSBlock(C, n_state) for _ in range(n_blocks)])
        self.up = nn.ModuleList([nn.Conv2d(C, C, 3, padding=1) for _ in range(n_down)])
        self.out = nn.Conv2d(C, out_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start as identity (output=degraded)

    def forward(self, x):
        f = self.intro(x)
        skips = []
        for dn in self.down:
            skips.append(f); f = dn(f)
        f = self.body(f)
        for up, sk in zip(self.up, reversed(skips)):
            f = torch.nn.functional.interpolate(f, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            f = up(f) + sk
        return torch.clamp(x[:, :1] + self.out(f), 0.0, 1.0)


BACKBONES = {"plainunet": PlainUNet, "restormer": RestormerLite, "mamba": MambaLite}


if __name__ == "__main__":
    for name, M in BACKBONES.items():
        for ic in (3, 1):
            m = M(in_ch=ic)
            n = sum(p.numel() for p in m.parameters())
            out = m(torch.randn(2, ic, 128, 128))
            print(f"{name:12s} in_ch={ic} params {n/1e6:.2f}M  out {tuple(out.shape)}")
