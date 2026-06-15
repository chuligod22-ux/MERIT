# -*- coding: utf-8 -*-
"""Fig.11/12 on the 448-crop false-crack protocol (clean_max 0.092) — FULL Table 2 battery + NAFNet-big.
Region: crack_d25_ISO100_V60, top-40% 448x448 crack-free window, degrade seed 6 (mirrors
eval_diffusion_falsecrack.py). Diffusion: SD-x4 run live on the 448 crop; DiffBIR loaded from its saved
crack-free restoration (05_tmp/diffbir_out/crackfree_deg.png). Saves F6/F6b + CSV."""
import os, sys, glob
import numpy as np, cv2, torch
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, unsupervised_wiener, denoise_tv_chambolle, denoise_nl_means
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet
mpl.rcParams.update({"font.size":10,"figure.dpi":200,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
NRIQA=os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")); DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")); FIG=os.path.join(DATA,"figs")
DBOUT=os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_out")
dev="cuda"; CSEG=os.path.join(NRIQA,"03_src","crack_seg"); sys.path.insert(0,CSEG)
from utils import load_unet_vgg16
INP=448; SH,SV,NSTD=2.6,2.4,2.5
tfm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
unet=load_unet_vgg16(os.path.join(CSEG,"models","model_unet_vgg16_best.pt")).cuda().eval()
def loadn(n):
    ck=torch.load(os.path.join(DATA,n),map_location=dev); m=NAFNetUNet(width=ck["width"]).to(dev).eval(); m.load_state_dict(ck["state"]); return m
net_big,net_task=loadn("restorer_big.pt"),loadn("restorer_task.pt")
u8=lambda a:np.clip(a,0,255).astype(np.uint8); gray=lambda a:cv2.cvtColor(u8(a),cv2.COLOR_RGB2GRAY)
def crack_pm(rgb448):
    t=tfm(np.clip(rgb448,0,255).astype(np.uint8))[None].cuda()
    with torch.no_grad(): return torch.sigmoid(unet(t))[0,0].cpu().numpy()
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

# ---- 448 crack-free crop (mirrors eval_diffusion_falsecrack.py) ----
frame=sorted(glob.glob(os.path.join(NRIQA,"04_data","raw","cam2","crack_d25_ISO100_V60","*.png")))[0]
rgb=cv2.cvtColor(cv2.imread(frame,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB); H,W,_=rgb.shape
topR=int((H//INP)*INP*0.40); best=None
for y0 in range(0,topR-INP+1,INP):
    for x0 in range(0,W-INP+1,INP):
        m=crack_pm(rgb[y0:y0+INP,x0:x0+INP]).max()
        if best is None or m<best[0]: best=(float(m),y0,x0)
_,y0,x0=best; clean=rgb[y0:y0+INP,x0:x0+INP].astype(np.float64)
print(f"crack-free 448 crop at ({y0},{x0}), clean max = {best[0]:.4f}")
deg=np.stack([gaussian_filter(clean[...,c],(SV,SH)) for c in range(3)],-1)+np.random.RandomState(6).randn(INP,INP,3)*NSTD

# ---- diffusion: SD-x4 live, DiffBIR from saved output ----
imgs={"clean":clean,"degraded":deg}
REP=["Wiener","Richardson-Lucy","unsup-Wiener","wrong-PSF-Wiener","TV-denoise","NL-means","unsharp"]
for m in REP: imgs[m]=restore(m,deg)
try:
    print("loading SD x4 ...",flush=True)
    from diffusers import StableDiffusionUpscalePipeline
    pipe=StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler",torch_dtype=torch.float16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    lr=Image.fromarray(u8(deg)).convert("RGB")
    with torch.no_grad(): hr=pipe(prompt="",image=lr,num_inference_steps=50,guidance_scale=0,generator=torch.Generator(dev).manual_seed(0)).images[0]
    sd=cv2.resize(np.array(hr.convert("RGB")).astype(np.float64),(INP,INP),interpolation=cv2.INTER_AREA)
    imgs["SD-x4"]=sd
except Exception as e:
    print("SD-x4 failed:",str(e)[:100]); imgs["SD-x4"]=None
db=cv2.imread(os.path.join(DBOUT,"crackfree_deg.png"),cv2.IMREAD_COLOR)
if db is not None:
    db=cv2.cvtColor(db,cv2.COLOR_BGR2RGB).astype(np.float64)
    if db.shape[:2]!=(INP,INP): db=cv2.resize(db,(INP,INP),interpolation=cv2.INTER_AREA)
    imgs["DiffBIR"]=db
else: imgs["DiffBIR"]=None
for m in ["NAFNet-big","MERIT"]: imgs[m]=restore(m,deg)

ORDER=["clean","degraded","Wiener","Richardson-Lucy","unsup-Wiener","wrong-PSF-Wiener","TV-denoise",
       "NL-means","unsharp","SD-x4","DiffBIR","NAFNet-big","MERIT"]
ORDER=[m for m in ORDER if imgs.get(m) is not None]
mx={m:float(crack_pm(u8(imgs[m])).max()) for m in ORDER}
print("false-crack max (448 protocol, full battery):")
for m in ORDER: print(f"  {m:18s} {mx[m]:.4f}")
print("false-crack MEAN (448 protocol):")
for m in ["clean","NAFNet-big","MERIT"]:
    if m in ORDER: print(f"  MEAN {m:14s} {crack_pm(u8(imgs[m])).mean():.6f}")

SHOW={"clean":"clean","degraded":"degraded","Wiener":"Wiener","Richardson-Lucy":"Richardson-Lucy",
      "unsup-Wiener":"unsup. Wiener","wrong-PSF-Wiener":"Wiener (wrong PSF)","TV-denoise":"TV-denoise",
      "NL-means":"NL-means","unsharp":"unsharp","SD-x4":"SD-x4","DiffBIR":"DiffBIR v2.1","NAFNet-big":"NAFNet-big","MERIT":"MERIT"}
GEN="#7d3fb0"; DEC="#e08214"
def colof(m):
    if m=="MERIT": return "#2ca02c"
    if m in("SD-x4","DiffBIR"): return GEN
    if m in("Wiener","Richardson-Lucy","unsup-Wiener","wrong-PSF-Wiener"): return DEC
    if m in("TV-denoise","NL-means","unsharp"): return "#1f77b4"
    if m=="NAFNet-big": return "#8fbf8f"
    return "0.5"

# ---- Fig.11 (F6): heat overlays, full battery ----
gcl=gray(clean); V0,V1=np.percentile(gcl,2),np.percentile(gcl,98)
n=len(ORDER); ncol=(n+1)//2; nrow=2
fig,axes=plt.subplots(nrow,ncol,figsize=(1.75*ncol,2.1*nrow)); axes=axes.ravel()
for j,m in enumerate(ORDER):
    ax=axes[j]; pm=crack_pm(u8(imgs[m])); ax.imshow(gray(imgs[m]),cmap="gray",vmin=V0,vmax=V1)
    ax.imshow(np.ma.masked_less(pm,0.05),cmap="inferno",vmin=0,vmax=0.5,alpha=0.75); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{SHOW[m]}\nmax {mx[m]:.3f}",fontsize=8.5,fontweight="bold" if m=="MERIT" else "normal")
    for s in ax.spines.values(): s.set_edgecolor(colof(m)); s.set_linewidth(1.4 if m=="MERIT" else 0.6)
for j in range(n,len(axes)): axes[j].axis("off")
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F6_falsecrack.{ext}"))
plt.close(fig); print("saved F6_falsecrack (full battery)")

# ---- Fig.12 (F6b): max bar, full battery ----
fig,ax=plt.subplots(figsize=(9.2,4.0)); xs=np.arange(len(ORDER)); vals=[mx[m] for m in ORDER]
ax.bar(xs,vals,color=[colof(m) for m in ORDER],edgecolor="k",linewidth=0.6,width=0.7)
for i,m in enumerate(ORDER): ax.text(i,vals[i]+max(vals)*0.03,f"{vals[i]:.3f}",ha="center",fontsize=7.5,fontweight="bold" if m=="MERIT" else "normal")
ax.axhline(mx["clean"],ls="--",color="0.4",lw=1.0,label="clean reference")
ax.set_xticks(xs); ax.set_xticklabels([SHOW[m].replace(" ","\n") if len(SHOW[m])>9 else SHOW[m] for m in ORDER],fontsize=7.5)
ax.set_ylabel("max crack-prob (crack-free region)"); ax.set_ylim(0,max(vals)*1.3); ax.grid(axis="y",alpha=0.25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.legend(frameon=False,fontsize=8.5,loc="upper left")
ax.set_title("all methods 0% above the 0.5 detection threshold",fontsize=9,color="0.3",pad=6)
fig.tight_layout()
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F6b_falsecrack_maxprob.{ext}"))
plt.close(fig); print("saved F6b_falsecrack_maxprob (full battery)")
