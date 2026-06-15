# -*- coding: utf-8 -*-
"""Fig: DiffBIR fabricates cracks on a crack-region detection crop (the +1126% detection 'recovery' is
hallucination, not recovery). Shows degraded input vs DiffBIR output with segmenter heat overlay."""
import os,sys
import numpy as np,cv2,torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torchvision.transforms as T
NRIQA=os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")); DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")); FIG=os.path.join(DATA,"figs")
DBIN=os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_det_in")
DBOUT=os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_det_out")
CSEG=os.path.join(NRIQA,"03_src","crack_seg"); sys.path.insert(0,CSEG); from utils import load_unet_vgg16
INP=448
tfm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
unet=load_unet_vgg16(os.path.join(CSEG,"models","model_unet_vgg16_best.pt")).cuda().eval()
def crack_prob(rgb):
    H,W,_=rgb.shape; pm=np.zeros((H,W),np.float32); tens,locs=[],[]
    for yy in range(0,H-INP+1,INP):
        for xx in range(0,W-INP+1,INP):
            tens.append(tfm(rgb[yy:yy+INP,xx:xx+INP])); locs.append((xx,yy))
    with torch.no_grad():
        for i in range(0,len(tens),8):
            pr=torch.sigmoid(unet(torch.stack(tens[i:i+8]).cuda()))[:,0].cpu().numpy()
            for j,(xx,yy) in enumerate(locs[i:i+8]): pm[yy:yy+INP,xx:xx+INP]=pr[j]
    return pm
fid="f05"
deg=cv2.cvtColor(cv2.imread(os.path.join(DBIN,fid+".png"),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
db=cv2.cvtColor(cv2.imread(os.path.join(DBOUT,fid+".png"),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
if db.shape[:2]!=deg.shape[:2]: db=cv2.resize(db,(deg.shape[1],deg.shape[0]),interpolation=cv2.INTER_AREA)
gray=lambda a:cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
pm_d=crack_prob(deg); pm_b=crack_prob(db)
items=[("Degraded input\n(faint real crack)",deg,pm_d,"#a33"),
       ("DiffBIR output\n(fabricated crack network)",db,pm_b,"#7d3fb0")]
fig,axes=plt.subplots(2,2,figsize=(8.4,5.8),gridspec_kw={"hspace":0.10,"wspace":0.04})
for j,(name,im,pm,col) in enumerate(items):
    axes[0,j].imshow(gray(im),cmap="gray"); axes[0,j].axis("off")
    axes[0,j].set_title(name,fontsize=10,color=col,fontweight="bold")
    axes[1,j].imshow(gray(im),cmap="gray")
    axes[1,j].imshow(np.ma.masked_less(pm,0.4),cmap="autumn",vmin=0.4,vmax=1.0,alpha=0.8); axes[1,j].axis("off")
    axes[1,j].set_title(f"segmenter detections (>0.5): mean {pm.mean():.5f}",fontsize=9,color=col)
for ext in ("png","pdf"): fig.savefig(os.path.join(FIG,f"F12b_diffbir_fabrication.{ext}"),dpi=160 if ext=="png" else None,bbox_inches="tight")
print(f"deg mean {pm_d.mean():.4f}  DiffBIR mean {pm_b.mean():.4f}  ratio {pm_b.mean()/max(pm_d.mean(),1e-9):.0f}x")
print("saved F12b_diffbir_fabrication")
