# -*- coding: utf-8 -*-
"""F3 (display Fig. 4) redo with better visibility: two panels vs added blur sigma.
(a) recovered MTF50 with a shaded clean +-band; (b) PSNR-to-clean (fidelity / hallucination axis).
The PSNR panel separates MERIT (stable ~45 dB) from Wiener (collapses at low blur) where the
MTF50 curves overlap. Data: degraded / oracle-Wiener (best-K by MTF50) / MERIT, mean over 3 seeds."""
import os, glob, csv
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA  = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
dev = "cuda"
ck = torch.load(os.path.join(WEIGHTS, "restorer_task.pt"), map_location=dev)
net = NAFNetUNet(width=ck["width"]).to(dev).eval(); net.load_state_dict(ck["state"])

CAM1 = glob.glob(os.path.join(NRIQA,"04_data","raw","cam1","60km_2.5m_ISO100","MTF*","frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64); Mg = 48
big = img[158-Mg:554+Mg, 937-Mg:1133+Mg]
inner = lambda a: a[Mg:Mg+396, Mg:Mg+196]
clean = inner(big); m_sharp = esf_mtf(clean)[2]
psnr = lambda a, b: 10*np.log10(255.0**2 / max(np.mean((np.clip(a,0,255)-b)**2), 1e-9))

def gpsf(sh, sv):
    nh=int(2*np.ceil(3*sh)+1); nv=int(2*np.ceil(3*sv)+1)
    kh=np.exp(-(np.arange(nh)-nh//2)**2/(2*sh**2)); kv=np.exp(-(np.arange(nv)-nv//2)**2/(2*sv**2))
    k=np.outer(kv,kh); return k/k.sum()
def wiener(im, psf, K):
    pad=np.zeros_like(im); kh,kw=psf.shape; pad[:kh,:kw]=psf
    pad=np.roll(pad,(-(kh//2),-(kw//2)),(0,1)); H=np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(im)*np.conj(H)/(np.abs(H)**2+K)))
def merit(im, sh, sv, nstd):
    xi=np.clip(im,0,255)/255.0; b=np.full_like(xi,0.5*(sh+sv)/BLUR_MAX); n=np.full_like(xi,nstd/NOISE_MAX)
    x=torch.from_numpy(np.stack([xi,b,n],0)[None]).float().to(dev)
    H,W=x.shape[-2:]; ph,pw=(8-H%8)%8,(8-W%8)%8
    xp=torch.nn.functional.pad(x,(0,pw,0,ph),mode="reflect")
    with torch.no_grad(): return net(xp)[0,0,:H,:W].cpu().numpy()*255.0

SIG=[1.0,1.5,2.0,2.5,3.0]
md,mw,mm,pd,pw_,pm=[],[],[],[],[],[]
for sb in SIG:
    sh,sv=sb,sb*0.9; psf=gpsf(sh,sv)
    dd,ww,mmm,pdd,pww,pmm=[],[],[],[],[],[]
    for s in range(3):
        deg=gaussian_filter(big,(sv,sh))+np.random.RandomState(s).randn(*big.shape)*2.5
        dd.append(esf_mtf(inner(deg))[2]); pdd.append(psnr(inner(deg),clean))
        bestK=max([wiener(deg,psf,K) for K in [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]],key=lambda r:esf_mtf(inner(r))[2])
        ww.append(esf_mtf(inner(bestK))[2]); pww.append(psnr(inner(bestK),clean))
        r=merit(deg,sh,sv,2.5); mmm.append(esf_mtf(inner(r))[2]); pmm.append(psnr(inner(r),clean))
    md.append(np.mean(dd)); mw.append(np.mean(ww)); mm.append(np.mean(mmm))
    pd.append(np.mean(pdd)); pw_.append(np.mean(pww)); pm.append(np.mean(pmm))
    print(f"sig {sb}: MTF deg {md[-1]:.3f} Wie {mw[-1]:.3f} MERIT {mm[-1]:.3f} | PSNR deg {pd[-1]:.1f} Wie {pw_[-1]:.1f} MERIT {pm[-1]:.1f}")

plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
GRY,ORN,GRN="#7a7a7a","#e08214","#2c7a2c"
fig,(axA,axB)=plt.subplots(1,2,figsize=(10.2,4.2))

axA.axhspan(m_sharp-0.002,m_sharp+0.002,color="0.80",alpha=0.5,zorder=0)
axA.axhline(m_sharp,ls="--",color="k",lw=1.2,label=f"clean ({m_sharp:.3f})",zorder=1)
axA.plot(SIG,md,"o:",color=GRY,lw=1.8,ms=7,label="degraded")
axA.plot(SIG,mw,"s--",color=ORN,lw=1.8,ms=7,label="Wiener (oracle)")
axA.plot(SIG,mm,"*-",color=GRN,lw=2.6,ms=13,label="MERIT (ours)")
axA.annotate("Wiener\nnoise blow-up",xy=(1.0,mw[0]),xytext=(1.35,mw[0]+0.012),
             fontsize=9,color=ORN,arrowprops=dict(arrowstyle="->",color=ORN,lw=1.2))
axA.set_xlabel("added blur $\\sigma$ (px)"); axA.set_ylabel("recovered MTF50 (cy/px)")
axA.set_ylim(0.02,0.085); axA.set_xticks(SIG)
axA.legend(frameon=False,fontsize=9.5,loc="lower right")

axB.plot(SIG,pd,"o:",color=GRY,lw=1.8,ms=7,label="degraded")
axB.plot(SIG,pw_,"s--",color=ORN,lw=1.8,ms=7,label="Wiener (oracle)")
axB.plot(SIG,pm,"*-",color=GRN,lw=2.6,ms=13,label="MERIT (ours)")
axB.annotate(f"{pw_[0]:.1f} dB",xy=(1.03,pw_[0]),xytext=(1.4,pw_[0]+0.7),fontsize=9,color=ORN,
             va="center",ha="left",arrowprops=dict(arrowstyle="->",color=ORN,lw=1.2))
axB.set_xlabel("added blur $\\sigma$ (px)"); axB.set_ylabel("PSNR to clean (dB)")
axB.set_xticks(SIG); axB.set_ylim(33.0,46.5)
axB.legend(frameon=False,fontsize=9.5,loc="lower right")

fig.tight_layout(rect=[0,0.07,1,1])
for ax,txt in [(axA,"(a) Resolution recovery (measurement axis)"),
               (axB,"(b) Pixel fidelity (hallucination axis)")]:
    p=ax.get_position(); fig.text((p.x0+p.x1)/2,0.015,txt,ha="center",va="bottom",fontsize=11)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(DATA,"figs",f"F3_mtf_sweep.{ext}"),dpi=170 if ext=="png" else None,bbox_inches="tight")
with open(os.path.join(DATA,"sweep.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["blur","mtf_deg","mtf_wiener","mtf_merit","psnr_deg","psnr_wiener","psnr_merit"])
    for i,sb in enumerate(SIG): w.writerow([sb,md[i],mw[i],mm[i],pd[i],pw_[i],pm[i]])
    w.writerow(["clean_mtf",m_sharp,"","","","",""])
print(f"saved F3_mtf_sweep (2-panel) + sweep.csv; clean {m_sharp:.3f}")
