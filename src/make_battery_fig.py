# -*- coding: utf-8 -*-
"""Fig.5: the held-out chart ROI at the reference degradation (the Table 2 input) restored by the
FULL Table 2 battery — all nine baselines + MERIT. Same degradation as eval_mpd.py (sh2.6/sv2.4/n2.5,
seed 5) so MTF50/PSNR match Table 2. Diffusion priors: SD-x4 run live (diffusers); DiffBIR loaded from
05_tmp/diffbir_out/chart_deg.png (its official inference output). Saves F10_battery_chart.{png,pdf}."""
import os, glob
import numpy as np, torch, cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from skimage.restoration import richardson_lucy, unsupervised_wiener, denoise_tv_chambolle, denoise_nl_means
from mtf_util import imread_u, esf_mtf
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA  = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
DBOUT = os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_out")
dev = "cuda"
ck = torch.load(os.path.join(WEIGHTS, "restorer_task.pt"),map_location=dev)
net = NAFNetUNet(width=ck["width"]).to(dev).eval(); net.load_state_dict(ck["state"])

CAM1 = glob.glob(os.path.join(NRIQA,"04_data","raw","cam1","60km_2.5m_ISO100","MTF*","frame_000102.png"))[0]
img = imread_u(CAM1).astype(np.float64); Mg=48
big = img[158-Mg:554+Mg, 937-Mg:1133+Mg]
inner = lambda a: a[Mg:Mg+396, Mg:Mg+196]
clean = inner(big); m_clean = esf_mtf(clean)[2]
psnr = lambda a,b: 10*np.log10(255.0**2/max(np.mean((np.clip(a,0,255)-b)**2),1e-9))

def gpsf(sh,sv):
    nh=int(2*np.ceil(3*sh)+1); nv=int(2*np.ceil(3*sv)+1)
    kh=np.exp(-(np.arange(nh)-nh//2)**2/(2*sh**2)); kv=np.exp(-(np.arange(nv)-nv//2)**2/(2*sv**2))
    k=np.outer(kv,kh); return k/k.sum()
def wiener_f(im,psf,K):
    pad=np.zeros_like(im); kh,kw=psf.shape; pad[:kh,:kw]=psf
    pad=np.roll(pad,(-(kh//2),-(kw//2)),(0,1)); H=np.fft.fft2(pad)
    return np.real(np.fft.ifft2(np.fft.fft2(im)*np.conj(H)/(np.abs(H)**2+K)))
def wiener_best(im,psf):
    best=None
    for K in [3e-4,1e-3,3e-3,1e-2,3e-2]:
        r=wiener_f(im,psf,K); m=esf_mtf(inner(r))[2]
        if best is None or m>best[1]: best=(r,m)
    return best[0]
def naf(im):
    xi=np.clip(im,0,255)/255.0; b=np.full_like(xi,0.5*(sh+sv)/BLUR_MAX); nn=np.full_like(xi,nstd/NOISE_MAX)
    x=torch.from_numpy(np.stack([xi,b,nn],0)[None]).float().to(dev)
    H,W=x.shape[-2:]; ph,pw=(8-H%8)%8,(8-W%8)%8
    xp=torch.nn.functional.pad(x,(0,pw,0,ph),mode="reflect")
    with torch.no_grad(): return net(xp)[0,0,:H,:W].cpu().numpy()*255.0

sh,sv,nstd = 2.6,2.4,2.5
deg = gaussian_filter(big,sigma=(sv,sh)) + np.random.RandomState(5).randn(*big.shape)*nstd
psf = gpsf(sh,sv); psf_wrong = gpsf(1.5,1.5); d01 = np.clip(deg,0,255)/255.0

# --- SD-x4 (live diffusers; optional — skipped if the diffusers env is unavailable) ---
sd_chart = None
try:
    print("loading SD x4 ...", flush=True)
    from diffusers import StableDiffusionUpscalePipeline
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    def sdx4(im):
        lr = Image.fromarray(np.clip(im,0,255).astype(np.uint8)).convert("RGB")
        with torch.no_grad():
            hr = pipe(prompt="", image=lr, num_inference_steps=50, guidance_scale=0).images[0]
        return cv2.resize(np.array(hr.convert("L")).astype(np.float64),(im.shape[1],im.shape[0]),interpolation=cv2.INTER_AREA)
    sd_chart = sdx4(big)
except Exception as e:
    print("SD-x4 unavailable, skipping panel:", str(e)[:120])

# --- DiffBIR (official inference output) ---
db = cv2.cvtColor(cv2.imread(os.path.join(DBOUT,"chart_deg.png"),cv2.IMREAD_COLOR),cv2.COLOR_BGR2GRAY).astype(np.float64)
if db.shape != big.shape: db = cv2.resize(db,(big.shape[1],big.shape[0]),interpolation=cv2.INTER_AREA)

GEN="#7d3fb0"
panels = [
    ("clean (reference)", clean, "k"),
    ("degraded input", inner(deg), "#a33"),
    ("Wiener (oracle PSF)", inner(wiener_best(deg,psf)), "0.3"),
    ("Richardson–Lucy", inner(richardson_lucy(d01,psf/psf.sum(),num_iter=20)*255.0), "0.3"),
    ("unsup. Wiener (oracle)", inner(unsupervised_wiener(d01,psf)[0]*255.0), "0.3"),
    ("Wiener (wrong PSF)", inner(wiener_best(deg,psf_wrong)), "0.3"),
    ("TV denoise", inner(denoise_tv_chambolle(d01,weight=0.08)*255.0), "0.3"),
    ("NL-means", inner(denoise_nl_means(d01,patch_size=5,patch_distance=6,h=0.05)*255.0), "0.3"),
    ("unsharp mask", inner(deg+1.5*(deg-gaussian_filter(deg,1.5))), "0.3"),
]
if sd_chart is not None:
    panels.append(("SD-x4 diffusion", inner(sd_chart), GEN))
panels.append(("DiffBIR v2.1", inner(db), GEN))
panels.append(("MERIT (ours)", inner(naf(deg)), "#1a6e1a"))

VMIN,VMAX = np.percentile(clean,2), np.percentile(clean,98)
crop = lambda a: a[60:360]
ncol = 4; nrow = int(np.ceil(len(panels)/ncol))
fig = plt.figure(figsize=(1.7*ncol, 3.0*nrow))
gs = GridSpec(nrow,ncol,hspace=0.22,wspace=0.05,left=0.01,right=0.99,top=0.955,bottom=0.01)
print(f"clean MTF50={m_clean:.3f}")
for k,(name,im,col) in enumerate(panels):
    r,c = divmod(k,ncol); ax=fig.add_subplot(gs[r,c])
    ax.imshow(crop(im),cmap="gray",vmin=VMIN,vmax=VMAX,aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    m=esf_mtf(im)[2]; P=psnr(im,clean)
    pct = "" if name.startswith("clean") else f" ({m/m_clean*100:.0f}%)"
    sub = f"MTF {m:.3f}{pct}\nPSNR {P:.1f} dB" if not name.startswith("clean") else f"MTF {m:.3f}"
    ax.set_title(name,fontsize=9.5,color=col,fontweight=("bold" if "MERIT" in name else "normal"),pad=3)
    bc = "#1a6e1a" if "MERIT" in name else ("#a33" if "degraded" in name else (GEN if col==GEN else "0.25"))
    ax.text(0.5,0.03,sub,transform=ax.transAxes,ha="center",va="bottom",fontsize=8.2,color="white",
            bbox=dict(boxstyle="round,pad=0.18",fc=bc,ec="none",alpha=0.85))
    for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(1.6 if ("MERIT" in name or "degraded" in name) else 0.8)
    print(f"  {name.encode('ascii','replace').decode():22s} MTF {m:.3f}  PSNR {P:.1f}")
for ext in ("png","pdf"):
    fig.savefig(os.path.join(DATA,"figs",f"F10_battery_chart.{ext}"),dpi=170 if ext=="png" else None,bbox_inches="tight")
print("saved F10_battery_chart (full Table 2 battery)")
