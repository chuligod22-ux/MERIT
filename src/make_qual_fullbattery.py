# -*- coding: utf-8 -*-
"""Full-battery qualitative figures (single regions, fast — no 12-frame loop):
  F5 (Fig.9): one crack region restored by the full reproducible battery, with detection-fitness %.
  F6 (Fig.11): one crack-free region segmenter-heat overlays for the full battery, with max crack-prob.
Same degradation/regions as the bars so numbers are consistent. Diffusion priors stay in text."""
import os, sys, glob
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, unsupervised_wiener, denoise_tv_chambolle, denoise_nl_means
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
NRIQA=os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")); DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")); FIG=os.path.join(DATA,"figs")
dev="cuda"; CSEG=os.path.join(NRIQA,"03_src","crack_seg"); sys.path.insert(0,CSEG)
from utils import load_unet_vgg16
INP=448; DISP=896; SH,SV,NSTD=2.6,2.4,2.5
tfm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
unet=load_unet_vgg16(os.path.join(CSEG,"models","model_unet_vgg16_best.pt")).cuda().eval()
def loadn(n):
    ck=torch.load(os.path.join(DATA,n),map_location=dev); m=NAFNetUNet(width=ck["width"]).to(dev).eval(); m.load_state_dict(ck["state"]); return m
net_big,net_task=loadn("restorer_big.pt"),loadn("restorer_task.pt")
u8=lambda a:np.clip(a,0,255).astype(np.uint8); gray=lambda a:cv2.cvtColor(u8(a),cv2.COLOR_RGB2GRAY)
psnr=lambda a,b:10*np.log10(255**2/max(np.mean((a-b)**2),1e-9))
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
    x=deg/255.0; h,w=x.shape[:2]; b=np.full((h,w),0.5*(SH+SV)/BLUR_MAX,np.float32); n=np.full((h,w),NSTD/NOISE_MAX,np.float32)
    batch=np.stack([np.stack([x[...,c],b,n],0) for c in range(3)],0)
    with torch.no_grad(): out=net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:,0]
    return np.clip(np.transpose(out,(1,2,0))*255.0,0,255)
def restore(name,deg):
    d01=np.clip(deg,0,255)/255.0
    if name=="Wiener": return np.stack([wiener_ch(deg[...,c],1e-2) for c in range(3)],-1)
    if name=="Richardson-Lucy": return np.stack([richardson_lucy(d01[...,c],PSF/PSF.sum(),num_iter=20)*255.0 for c in range(3)],-1)
    if name=="unsup-Wiener": return np.stack([np.clip(unsupervised_wiener(d01[...,c],PSF)[0],0,1)*255.0 for c in range(3)],-1)
    if name=="wrong-PSF-Wiener": return np.stack([wiener_ch(deg[...,c],1e-2,PSF_WRONG) for c in range(3)],-1)
    if name=="TV-denoise": return denoise_tv_chambolle(d01,weight=0.08,channel_axis=-1)*255.0
    if name=="NL-means": return denoise_nl_means(d01,patch_size=5,patch_distance=6,h=0.05,channel_axis=-1)*255.0
    if name=="unsharp": return deg+1.5*(deg-np.stack([gaussian_filter(deg[...,c],1.5) for c in range(3)],-1))
    if name=="NAFNet-big": return nafr(net_big,deg)
    if name=="MERIT": return nafr(net_task,deg)
def degrade(clean,seed):
    return np.stack([gaussian_filter(clean[...,c],(SV,SH)) for c in range(3)],-1)+np.random.RandomState(seed).randn(*clean.shape)*NSTD
def load_region(cond,center):
    fr=sorted(glob.glob(os.path.join(NRIQA,"04_data","raw","cam2",cond,"*.png")))[0]
    rgb=cv2.cvtColor(cv2.imread(fr,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB); H,W,_=rgb.shape
    top=(int(H*0.40)//INP)*INP; pm=crack_prob(rgb[:top,:(W//INP)*INP])
    if center:
        ys,xs=np.where(pm>0.4); cy,cx=(int(np.median(ys)),int(np.median(xs))) if len(ys)>20 else np.unravel_index(pm.argmax(),pm.shape)
    else:
        best=None
        for y0 in range(0,pm.shape[0]-DISP+1,INP):
            for x0 in range(0,W-DISP+1,INP):
                mx=pm[y0:y0+DISP,x0:x0+DISP].max()
                if best is None or mx<best[0]: best=(mx,y0,x0)
        _,cy,cx=best[0],best[1]+DISP//2,best[2]+DISP//2
    y0=int(np.clip(cy-DISP//2,0,pm.shape[0]-DISP)); x0=int(np.clip(cx-DISP//2,0,W-DISP))
    return rgb[y0:y0+DISP,x0:x0+DISP].astype(np.float64)
RESTOR=["Wiener","Richardson-Lucy","TV-denoise","NL-means","unsharp","NAFNet-big","MERIT"]

# ===== F5 (Fig.9): crack region, full battery incl diffusion, detection-fitness % =====
import cv2 as _cv2
from PIL import Image as _Image
clean=load_region("crack_d25_ISO100_V60",True); deg=degrade(clean,2)
fc=crack_prob(u8(clean)).mean()
# SD-x4 (448-tiled) on the crop
from diffusers import StableDiffusionUpscalePipeline
_pipe=StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler",torch_dtype=torch.float16).to(dev)
_pipe.set_progress_bar_config(disable=True)
def _sdx4(rgb):
    H,W,_=rgb.shape; out=np.zeros_like(rgb,np.float64)
    for yy in range(0,H,INP):
        for xx in range(0,W,INP):
            t=rgb[yy:yy+INP,xx:xx+INP]; th,tw=t.shape[:2]
            with torch.no_grad(): hr=_pipe(prompt="",image=_Image.fromarray(u8(t)).convert("RGB"),num_inference_steps=50,guidance_scale=0,generator=torch.Generator(dev).manual_seed(0)).images[0]
            out[yy:yy+th,xx:xx+tw]=_cv2.resize(np.array(hr.convert("RGB")).astype(np.float64),(tw,th),interpolation=_cv2.INTER_AREA)
    return out
sd=_sdx4(deg)
# DiffBIR (official inference output for this crop)
_dbp=os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_qual_out", "q0.png")
_db=_cv2.imread(_dbp,_cv2.IMREAD_COLOR)
db=None
if _db is not None:
    db=_cv2.cvtColor(_db,_cv2.COLOR_BGR2RGB).astype(np.float64)
    if db.shape[:2]!=(DISP,DISP): db=_cv2.resize(db,(DISP,DISP),interpolation=_cv2.INTER_AREA)
panels=[("Clean",clean),("Degraded",deg)]
for m in ["Wiener","Richardson-Lucy","unsup-Wiener","wrong-PSF-Wiener","TV-denoise","NL-means","unsharp"]: panels.append((m,restore(m,deg)))
panels.append(("SD-x4",sd))
if db is not None: panels.append(("DiffBIR",db))
for m in ["NAFNet-big","MERIT"]: panels.append((m,restore(m,deg)))
pmc=crack_prob(u8(clean)); zy,zx=np.unravel_index(gaussian_filter(pmc,8).argmax(),pmc.shape)
Z=300; zy=int(np.clip(zy-Z//2,0,DISP-Z)); zx=int(np.clip(zx-Z//2,0,DISP-Z))
gcl=gray(clean)[zy:zy+Z,zx:zx+Z]; VMIN,VMAX=np.percentile(gcl,2),np.percentile(gcl,98)
n=len(panels); ncol=(n+1)//2; nrow=2
fig,axes=plt.subplots(nrow,ncol,figsize=(1.85*ncol,2.3*nrow)); axes=axes.ravel()
for j,(name,im) in enumerate(panels):
    ax=axes[j]; g=gray(im)[zy:zy+Z,zx:zx+Z]; ax.imshow(g,cmap="gray",vmin=VMIN,vmax=VMAX); ax.set_xticks([]); ax.set_yticks([])
    det=crack_prob(u8(im)).mean()/fc*100
    sub="" if name=="Clean" else f"\ndet {det:.0f}%"
    col="#1a6e1a" if name=="MERIT" else ("#7d3fb0" if name in("SD-x4","DiffBIR") else "0.5")
    ax.set_title(name+sub,fontsize=9,color=col,fontweight="bold" if name=="MERIT" else "normal")
    for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(1.4 if name=="MERIT" else 0.6)
    print(f"  {name:16s} det {det:.0f}%")
for j in range(n,len(axes)): axes[j].axis("off")
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F5_qualitative.{ext}"))
plt.close(fig); print("saved F5_qualitative (full battery incl diffusion)")

# ===== F6 (Fig.11): crack-free region, full battery, segmenter heat + max =====
cf=load_region("crack_d35_ISO400_V60",False); degf=degrade(cf,4)
items=[("Clean",cf),("Degraded",degf)]+[(m,restore(m,degf)) for m in RESTOR]
n=len(items); fig,axes=plt.subplots(1,n,figsize=(1.5*n,2.0))
for j,(name,im) in enumerate(items):
    pm=crack_prob(u8(im)); axes[j].imshow(gray(im),cmap="gray",vmin=0,vmax=255)
    axes[j].imshow(np.ma.masked_less(pm,0.05),cmap="inferno",vmin=0,vmax=0.5,alpha=0.75); axes[j].set_xticks([]); axes[j].set_yticks([])
    axes[j].set_title(f"{name}\nmax {pm.max():.3f}",fontsize=8,fontweight="bold" if name=="MERIT" else "normal")
    for s in axes[j].spines.values(): s.set_edgecolor("#1a6e1a" if name=="MERIT" else "0.5"); s.set_linewidth(1.4 if name=="MERIT" else 0.6)
fig.tight_layout()
plt.close(fig); print("(F6/Fig.11 owned by make_falsecrack_448.py — not written here)")
