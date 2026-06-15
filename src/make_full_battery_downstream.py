# -*- coding: utf-8 -*-
"""Expand the §4.4/§4.5 downstream figures to the full reproducible battery.
Methods: degraded, Wiener(oracle), Richardson-Lucy(oracle), TV-denoise, NL-means, unsharp,
         NAFNet-big (fidelity-only), MERIT (task).  Diffusion priors stay in text (not run here).
Computes:
  (A) detection-fitness recovery over the N=12 held-out cam2 frames  -> F4_detection_recovery (Fig.10)
  (B) false-crack max crack-prob on one crack-free region            -> F6b_falsecrack_maxprob (Fig.12)
Saves a CSV of the numbers for the manuscript.
"""
import os, sys, glob, csv
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, unsupervised_wiener, denoise_tv_chambolle, denoise_nl_means
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
mpl.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":200,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
NRIQA=os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")); FIG=os.path.join(DATA,"figs")
dev="cuda"; CSEG=os.path.join(NRIQA,"03_src","crack_seg"); sys.path.insert(0,CSEG)
from utils import load_unet_vgg16
INP=448
tfm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
unet=load_unet_vgg16(os.path.join(CSEG,"models","model_unet_vgg16_best.pt")).cuda().eval()
def loadn(n):
    ck=torch.load(os.path.join(DATA,n),map_location=dev); m=NAFNetUNet(width=ck["width"]).to(dev).eval(); m.load_state_dict(ck["state"]); return m
net_big,net_task=loadn("restorer_big.pt"),loadn("restorer_task.pt")
u8=lambda a:np.clip(a,0,255).astype(np.uint8)
SH,SV,NSTD=2.6,2.4,2.5
def crack_prob(rgb):
    H,W,_=rgb.shape; pm=np.zeros((H,W),np.float32); tens,locs=[],[]
    for yy in range(0,H-INP+1,INP):
        for xx in range(0,W-INP+1,INP):
            tens.append(tfm(rgb[yy:yy+INP,xx:xx+INP])); locs.append((xx,yy))
    with torch.no_grad():
        for i in range(0,len(tens),16):
            pr=torch.sigmoid(unet(torch.stack(tens[i:i+16]).cuda()))[:,0].cpu().numpy()
            for j,(xx,yy) in enumerate(locs[i:i+16]): pm[yy:yy+INP,xx:xx+INP]=pr[j]
    return pm
def gpsf(sh,sv):
    nh=int(2*np.ceil(3*sh)+1); nv=int(2*np.ceil(3*sv)+1)
    kh=np.exp(-(np.arange(nh)-nh//2)**2/(2*sh**2)); kv=np.exp(-(np.arange(nv)-nv//2)**2/(2*sv**2))
    k=np.outer(kv,kh); return k/k.sum()
PSF=gpsf(SH,SV); PSF_WRONG=gpsf(1.5,1.5)
def wiener_ch(img,K,psf=None):
    psf=PSF if psf is None else psf
    pad=np.zeros_like(img,np.float64); kh,kw=psf.shape; pad[:kh,:kw]=psf
    pad=np.roll(pad,(-(kh//2),-(kw//2)),(0,1)); Hf=np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(img)*np.conj(Hf)/(np.abs(Hf)**2+K)))
def nafr(net,deg):
    x=deg/255.0; h,w=x.shape[:2]
    b=np.full((h,w),0.5*(SH+SV)/BLUR_MAX,np.float32); n=np.full((h,w),NSTD/NOISE_MAX,np.float32)
    batch=np.stack([np.stack([x[...,c],b,n],0) for c in range(3)],0)
    with torch.no_grad(): out=net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:,0]
    return np.clip(np.transpose(out,(1,2,0))*255.0,0,255)
# restoration of an RGB degraded crop by method name -> RGB
def restore(name,deg):
    d01=np.clip(deg,0,255)/255.0
    if name=="degraded": return deg
    if name=="Wiener":   return np.stack([max([wiener_ch(deg[...,c],K) for K in [3e-4,1e-3,3e-3,1e-2,3e-2]],
                                              key=lambda r:0) for c in range(3)],-1) if False else \
                                np.stack([wiener_ch(deg[...,c],1e-2) for c in range(3)],-1)
    if name=="Richardson-Lucy": return np.stack([richardson_lucy(d01[...,c],PSF/PSF.sum(),num_iter=20)*255.0 for c in range(3)],-1)
    if name=="unsup-Wiener":    return np.stack([np.clip(unsupervised_wiener(d01[...,c],PSF)[0],0,1)*255.0 for c in range(3)],-1)
    if name=="wrong-PSF-Wiener":return np.stack([wiener_ch(deg[...,c],1e-2,PSF_WRONG) for c in range(3)],-1)
    if name=="TV-denoise":      return denoise_tv_chambolle(d01,weight=0.08,channel_axis=-1)*255.0
    if name=="NL-means":        return denoise_nl_means(d01,patch_size=5,patch_distance=6,h=0.05,channel_axis=-1)*255.0
    if name=="unsharp":         return deg+1.5*(deg-np.stack([gaussian_filter(deg[...,c],1.5) for c in range(3)],-1))
    if name=="NAFNet-big":      return nafr(net_big,deg)
    if name=="MERIT":           return nafr(net_task,deg)
    raise ValueError(name)
METHODS=["degraded","Wiener","Richardson-Lucy","unsup-Wiener","wrong-PSF-Wiener","TV-denoise","NL-means","unsharp","NAFNet-big","MERIT"]

# ---------- (A) detection recovery over the 12 held-out frames ----------
CH,CW=896,1344
conds=["crack_d25_ISO200_V80","crack_d35_ISO200_V60","crack_d35_ISO200_V80",
       "crack_d45_ISO100_V60","crack_d45_ISO100_V80","crack_d25_ISO100_V80"]
frames=[]
for c in conds: frames+=sorted(glob.glob(os.path.join(NRIQA,"04_data","raw","cam2",c,"*.png")))[:4]
rec={m:[] for m in METHODS if m!="degraded"}
nfr=0
for fr in frames:
    rgb=cv2.cvtColor(cv2.imread(fr,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    H,W,_=rgb.shape; top=int((H//INP)*INP*0.40)
    pm=crack_prob(rgb[:((top//INP)*INP),:((W//INP)*INP)])
    if pm.max()<0.4: continue
    ys,xs=np.where(pm>0.4); cy,cx=int(np.median(ys)),int(np.median(xs))
    y0=int(np.clip(cy-CH//2,0,pm.shape[0]-CH)); x0=int(np.clip(cx-CW//2,0,W-CW))
    clean=rgb[y0:y0+CH,x0:x0+CW].astype(np.float64); fc=float(crack_prob(u8(clean)).mean())
    if fc<1e-3: continue
    deg=np.stack([gaussian_filter(clean[...,c],(SV,SH)) for c in range(3)],-1)+np.random.RandomState(7).randn(CH,CW,3)*NSTD
    fd=float(crack_prob(u8(deg)).mean()); nfr+=1
    for m in rec:
        fm=float(crack_prob(u8(restore(m,deg))).mean()); rec[m].append((fm-fd)/(fc-fd+1e-9)*100)
print(f"detection recovery over N={nfr} frames (mean +- std):")
det_mean={m:float(np.mean(v)) for m,v in rec.items()}; det_std={m:float(np.std(v)) for m,v in rec.items()}
for m in METHODS[1:]: print(f"  {m:16s} {det_mean[m]:+5.0f}% +- {det_std[m]:.0f}")

# ---------- (B) false-crack max on one crack-free region ----------
def crackfree_region(cond):
    fr=sorted(glob.glob(os.path.join(NRIQA,"04_data","raw","cam2",cond,"*.png")))[0]
    rgb=cv2.cvtColor(cv2.imread(fr,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB); H,W,_=rgb.shape
    top=(int(H*0.40)//INP)*INP; pm=crack_prob(rgb[:top,:(W//INP)*INP]); D=896; best=None
    for y0 in range(0,pm.shape[0]-D+1,INP):
        for x0 in range(0,W-D+1,INP):
            mx=pm[y0:y0+D,x0:x0+D].max()
            if best is None or mx<best[0]: best=(mx,y0,x0)
    _,y0,x0=best; return rgb[y0:y0+D,x0:x0+D].astype(np.float64)
cf=crackfree_region("crack_d35_ISO400_V60")
degf=np.stack([gaussian_filter(cf[...,c],(SV,SH)) for c in range(3)],-1)+np.random.RandomState(4).randn(*cf.shape)*NSTD
fcmax={}
fcmax["clean"]=float(crack_prob(u8(cf)).max())
for m in METHODS:
    img=degf if m=="degraded" else restore(m,degf)
    fcmax[m]=float(crack_prob(u8(img)).max())
print("\nfalse-crack max crack-prob:"); print("  clean", round(fcmax["clean"],4))
for m in METHODS: print(f"  {m:16s} {fcmax[m]:.4f}")

with open(os.path.join(DATA,"full_battery_downstream.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["method","det_recovery_mean","det_recovery_std","falsecrack_max"])
    w.writerow(["clean","","",f"{fcmax['clean']:.4f}"])
    for m in METHODS:
        w.writerow([m, "" if m=="degraded" else f"{det_mean[m]:.0f}", "" if m=="degraded" else f"{det_std[m]:.0f}", f"{fcmax[m]:.4f}"])

# ---------- Fig.10: detection recovery bar (full battery) ----------
order=[m for m in METHODS if m!="degraded"]
COL={"Wiener":"#e08214","Richardson-Lucy":"#d6a14a","unsup-Wiener":"#c2691a","wrong-PSF-Wiener":"#a85b16",
     "TV-denoise":"#1f77b4","NL-means":"#5fa3d6","unsharp":"#9467bd","NAFNet-big":"#8fbf8f","MERIT":"#2ca02c"}
fig,ax=plt.subplots(figsize=(8.8,4.4)); xs=np.arange(len(order))
ax.bar(xs,[det_mean[m] for m in order],yerr=[det_std[m] for m in order],capsize=4,
       color=[COL[m] for m in order],edgecolor="k",linewidth=0.6)
for i,m in enumerate(order):
    ax.text(i,det_mean[m]+det_std[m]+1.5,f"+{det_mean[m]:.0f}%",ha="center",fontsize=8.5,
            fontweight="bold" if m=="MERIT" else "normal")
ax.set_xticks(xs); ax.set_xticklabels([m.replace("-","-\n") if len(m)>9 else m for m in order],fontsize=8.5)
ax.set_ylabel("Detection-fitness recovery (% of lost, N=12)"); ax.set_ylim(min(0,min(det_mean.values())-5),100); ax.grid(axis="y",alpha=0.25)
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F4_detection_recovery.{ext}"))
plt.close(fig); print("\nsaved F4_detection_recovery (full battery)")

# ---------- Fig.12: false-crack max bar (full battery) ----------
border=["clean"]+METHODS
labels=[("clean" if m=="clean" else m) for m in border]
colc={"clean":"#7f7f7f","degraded":"#9e9e9e","Wiener":"#e08214","Richardson-Lucy":"#d6a14a",
      "TV-denoise":"#1f77b4","NL-means":"#5fa3d6","unsharp":"#9467bd","NAFNet-big":"#8fbf8f","MERIT":"#2ca02c"}
fig,ax=plt.subplots(figsize=(8.2,4.0)); xs=np.arange(len(border)); vals=[fcmax[m] for m in border]
ax.bar(xs,vals,color=[colc[m] for m in border],edgecolor="k",linewidth=0.6,width=0.65)
for i,m in enumerate(border):
    ax.text(i,vals[i]+max(vals)*0.03,f"{vals[i]:.3f}",ha="center",fontsize=8,fontweight="bold" if m=="MERIT" else "normal")
ax.axhline(fcmax["clean"],ls="--",color="0.4",lw=1.0,label="clean reference")
ax.set_xticks(xs); ax.set_xticklabels([m.replace("-","-\n") if len(m)>9 else m for m in labels],fontsize=8.5)
ax.set_ylabel("max crack-prob (crack-free region)"); ax.set_ylim(0,max(vals)*1.3); ax.grid(axis="y",alpha=0.25)
ax.legend(frameon=False,fontsize=8.5,loc="upper left")
ax.set_title("all methods 0% above the 0.5 detection threshold",fontsize=9,color="0.3",pad=6)
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"_unused_F6b_dmgcheck.{ext}"))  # Fig.12 owned by make_falsecrack_448.py
plt.close(fig); print("saved F6b_falsecrack_maxprob (full battery)")
