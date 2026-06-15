# -*- coding: utf-8 -*-
"""Fig.8 = the N=12 held-out cam2 crack-frame SAMPLE: each frame as a degraded | MERIT-restored pair
with its per-frame detection recovery. Shows MERIT recovers the crack across the whole sample and the
spread (+17% .. +126%). Method-vs-method comparison lives in Fig.9/Fig.10, so this figure stays 2-method
to avoid redundancy. Replicates eval_stats.py section (1) frame selection. Saves F11_det_frames.{png,pdf}."""
import os, sys, glob
import numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
from dataset import BLUR_MAX, NOISE_MAX
from model import NAFNetUNet

NRIQA = os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw"))
DATA  = os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
WEIGHTS = os.environ.get("MERIT_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights"))
dev = "cuda"
CSEG = os.path.join(NRIQA, "03_src", "crack_seg"); sys.path.insert(0, CSEG)
from utils import load_unet_vgg16
INP = 448
tfm = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
unet = load_unet_vgg16(os.path.join(CSEG,"models","model_unet_vgg16_best.pt")).cuda().eval()
ck = torch.load(os.path.join(WEIGHTS, "restorer_task.pt"),map_location=dev)
net = NAFNetUNet(width=ck["width"]).to(dev).eval(); net.load_state_dict(ck["state"])
u8 = lambda a: np.clip(a,0,255).astype(np.uint8)

def crack_prob(rgb):
    H,W,_=rgb.shape; pm=np.zeros((H,W),np.float32); tens,locs=[],[]
    for yy in range(0,H-INP+1,INP):
        for xx in range(0,W-INP+1,INP):
            tens.append(tfm(rgb[yy:yy+INP,xx:xx+INP])); locs.append((xx,yy))
    with torch.no_grad():
        for i in range(0,len(tens),32):
            pr=torch.sigmoid(unet(torch.stack(tens[i:i+32]).cuda()))[:,0].cpu().numpy()
            for j,(xx,yy) in enumerate(locs[i:i+32]): pm[yy:yy+INP,xx:xx+INP]=pr[j]
    return pm
def naf_rgb(deg,sh,sv,nstd,CH,CW):
    x=deg/255.0; b=np.full((CH,CW),0.5*(sh+sv)/BLUR_MAX,np.float32); n=np.full((CH,CW),nstd/NOISE_MAX,np.float32)
    batch=np.stack([np.stack([x[...,c],b,n],0) for c in range(3)],0)
    with torch.no_grad(): out=net(torch.from_numpy(batch).float().to(dev)).cpu().numpy()[:,0]
    return np.clip(np.transpose(out,(1,2,0))*255.0,0,255)

CH,CW=896,1344
conds=["crack_d25_ISO200_V80","crack_d35_ISO200_V60","crack_d35_ISO200_V80",
       "crack_d45_ISO100_V60","crack_d45_ISO100_V80","crack_d25_ISO100_V80"]
frames=[]
for c in conds: frames+=sorted(glob.glob(os.path.join(NRIQA,"04_data","raw","cam2",c,"*.png")))[:4]
sh,sv,nstd=2.6,2.4,2.5
items=[]
for fr in frames:
    rgb=cv2.cvtColor(cv2.imread(fr,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
    H,W,_=rgb.shape; top=int((H//INP)*INP*0.40)
    pm=crack_prob(rgb[:((top//INP)*INP),:((W//INP)*INP)])
    if pm.max()<0.4: continue
    ys,xs=np.where(pm>0.4); cy,cx=int(np.median(ys)),int(np.median(xs))
    y0=int(np.clip(cy-CH//2,0,pm.shape[0]-CH)); x0=int(np.clip(cx-CW//2,0,W-CW))
    clean=rgb[y0:y0+CH,x0:x0+CW].astype(np.float64); fc=float(crack_prob(u8(clean)).mean())
    if fc<1e-3: continue
    deg=np.stack([gaussian_filter(clean[...,c],(sv,sh)) for c in range(3)],-1)+np.random.RandomState(7).randn(CH,CW,3)*nstd
    res=naf_rgb(deg,sh,sv,nstd,CH,CW); fd=float(crack_prob(u8(deg)).mean())
    rec=(float(crack_prob(u8(res)).mean())-fd)/(fc-fd+1e-9)*100
    cond=os.path.basename(os.path.dirname(fr))[6:]
    lcy,lcx=cy-y0,cx-x0; ph,pw=170,250
    ty0=int(np.clip(lcy-ph,0,CH-2*ph)); tx0=int(np.clip(lcx-pw,0,CW-2*pw))
    cl=clean[ty0:ty0+2*ph,tx0:tx0+2*pw]; lo,hi=np.percentile(cl,2),np.percentile(cl,98)
    st=lambda a: np.clip((a[ty0:ty0+2*ph,tx0:tx0+2*pw]-lo)/(hi-lo+1e-9),0,1)
    items.append((st(deg),st(res),cond,rec))
    print(f"{cond:18s} MERIT recovery {rec:+.0f}%")
print(f"\nN={len(items)} frames; mean MERIT recovery {np.mean([r for _,_,_,r in items]):+.0f}%")

FR_PER_ROW=3; nrow=int(np.ceil(len(items)/FR_PER_ROW)); ncol=FR_PER_ROW*2
fig=plt.figure(figsize=(1.55*ncol,1.55*nrow+0.3))
gs=GridSpec(nrow,ncol,hspace=0.26,wspace=0.04,left=0.005,right=0.995,top=0.93,bottom=0.01)
for k,(dimg,rimg,cond,rec) in enumerate(items):
    fr,fc2=divmod(k,FR_PER_ROW); cd=fc2*2
    axd=fig.add_subplot(gs[fr,cd]); axr=fig.add_subplot(gs[fr,cd+1])
    axd.imshow(dimg,vmin=0,vmax=1); axr.imshow(rimg,vmin=0,vmax=1)
    for a in (axd,axr): a.set_xticks([]); a.set_yticks([])
    for s in axd.spines.values(): s.set_edgecolor("#a33"); s.set_linewidth(1.3)
    for s in axr.spines.values(): s.set_edgecolor("#1a6e1a"); s.set_linewidth(1.3)
    p=axd.get_position(); q=axr.get_position()
    fig.text((p.x0+q.x1)/2,q.y1+(0.030 if fr==0 else 0.004),cond,ha="center",va="bottom",fontsize=8)
    if fr==0:
        axd.set_title("degraded",fontsize=8.5,color="#a33",pad=2)
        axr.set_title("MERIT",fontsize=8.5,color="#1a6e1a",pad=2)
    axr.text(0.5,0.04,f"{rec:+.0f}%",transform=axr.transAxes,ha="center",va="bottom",fontsize=8.5,
             color="white",fontweight="bold",bbox=dict(boxstyle="round,pad=0.18",fc="#1a6e1a",ec="none",alpha=0.85))
for ext in ("png","pdf"): fig.savefig(os.path.join(DATA,"figs",f"F11_det_frames.{ext}"),dpi=150 if ext=="png" else None,bbox_inches="tight")
print("saved F11_det_frames (degraded|MERIT sample)")
