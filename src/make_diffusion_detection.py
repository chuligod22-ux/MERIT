# -*- coding: utf-8 -*-
"""Diffusion detection-recovery over the N=12 held-out cam2 crack frames (same selection as
make_full_battery_downstream.py). Computes SD-x4 (live, 448-tiled) detection recovery, and SAVES each
degraded crop + per-frame clean/degraded fitness so DiffBIR can be run via its CLI and read back.
Outputs: 05_tmp/diffbir_det_in/*.png (degraded crops), 04_data/diffusion_det.csv (fc/fd + SD-x4)."""
import os, sys, glob, csv
import numpy as np, cv2, torch
from PIL import Image
from scipy.ndimage import gaussian_filter
import torchvision.transforms as T
NRIQA=os.environ.get("MERIT_NRIQA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")); DATA=os.environ.get("MERIT_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
DBIN=os.path.join(os.environ.get("MERIT_TMP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")), "diffbir_det_in"); os.makedirs(DBIN,exist_ok=True)
dev="cuda"; CSEG=os.path.join(NRIQA,"03_src","crack_seg"); sys.path.insert(0,CSEG)
from utils import load_unet_vgg16
INP=448; SH,SV,NSTD=2.6,2.4,2.5
tfm=T.Compose([T.ToTensor(),T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
unet=load_unet_vgg16(os.path.join(CSEG,"models","model_unet_vgg16_best.pt")).cuda().eval()
u8=lambda a:np.clip(a,0,255).astype(np.uint8)
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
# SD-x4 (448-tiled to avoid OOM on the large crack crops)
from diffusers import StableDiffusionUpscalePipeline
pipe=StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler",torch_dtype=torch.float16).to(dev)
pipe.set_progress_bar_config(disable=True)
def sdx4(rgb):
    H,W,_=rgb.shape; out=np.zeros_like(rgb,np.float64)
    for yy in range(0,H,INP):
        for xx in range(0,W,INP):
            tile=rgb[yy:yy+INP,xx:xx+INP]; th,tw=tile.shape[:2]
            lr=Image.fromarray(u8(tile)).convert("RGB")
            with torch.no_grad(): hr=pipe(prompt="",image=lr,num_inference_steps=50,guidance_scale=0).images[0]
            out[yy:yy+th,xx:xx+tw]=cv2.resize(np.array(hr.convert("RGB")).astype(np.float64),(tw,th),interpolation=cv2.INTER_AREA)
    return out

CH,CW=896,1344
conds=["crack_d25_ISO200_V80","crack_d35_ISO200_V60","crack_d35_ISO200_V80",
       "crack_d45_ISO100_V60","crack_d45_ISO100_V80","crack_d25_ISO100_V80"]
frames=[]
for c in conds: frames+=sorted(glob.glob(os.path.join(NRIQA,"04_data","raw","cam2",c,"*.png")))[:4]
rows=[]; rec_sd=[]; k=0
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
    fd=float(crack_prob(u8(deg)).mean())
    cv2.imwrite(os.path.join(DBIN,f"f{k:02d}.png"),cv2.cvtColor(u8(deg),cv2.COLOR_RGB2BGR))   # for DiffBIR CLI
    fsd=float(crack_prob(u8(sdx4(deg))).mean()); r=(fsd-fd)/(fc-fd+1e-9)*100; rec_sd.append(r)
    rows.append([f"f{k:02d}",fc,fd]); k+=1
    print(f"f{k-1:02d} fc {fc:.4f} fd {fd:.4f}  SD-x4 {r:+.0f}%")
print(f"\nN={len(rec_sd)}  SD-x4 detection recovery {np.mean(rec_sd):+.0f}% +- {np.std(rec_sd):.0f}")
with open(os.path.join(DATA,"diffusion_det.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["frame","fc","fd","sdx4_recovery"]);
    for (fid,fc,fd),r in zip(rows,rec_sd): w.writerow([fid,f"{fc:.5f}",f"{fd:.5f}",f"{r:.1f}"])
print("saved diffbir_det_in/*.png + diffusion_det.csv")
