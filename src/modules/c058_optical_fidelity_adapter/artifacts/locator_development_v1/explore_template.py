from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(r'<LOCAL_PATH_REDACTED>')
COHORT=Path(r'<LOCAL_PATH_REDACTED>')
OUT=Path(__file__).parent

def rd(p): return cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)

def template(r, angle):
    h=w=round(2.2*r)
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    x=xx-(w-1)/2; y=yy-(h-1)/2
    c=np.cos(np.deg2rad(angle)); s=np.sin(np.deg2rad(angle))
    xr=c*x+s*y; yr=-s*x+c*y
    rr=np.sqrt(x*x+y*y)
    # Smooth-edged diagonal checker; outer pixels are neutral.
    t=np.zeros((h,w),np.float32)
    inside=rr < .88*r
    t[inside]=np.where(xr[inside]*yr[inside]>=0,-1,1)
    # exclude the immediate axes, which may be broad/blurred
    t[inside & ((np.abs(xr)<.08*r)|(np.abs(yr)<.08*r))]=0
    return t

TM=[(r,a,template(r,a)) for r in (25,32,40,50,62,76,92,110) for a in (-15,-10,-5,0,5,10,15)]

def detect(g):
    a=cv2.resize(g,(1024,683),interpolation=cv2.INTER_AREA).astype(np.float32)
    # top padding permits circles cut by the sensor boundary
    a=np.pad(a,((120,0),(0,0)),mode='edge')
    search=a[:120+280,:420]
    best=(-2,None)
    for r,angle,t in TM:
        if t.shape[0]>search.shape[0] or t.shape[1]>search.shape[1]: continue
        q=cv2.matchTemplate(search,t,cv2.TM_CCOEFF_NORMED)
        _,mx,_,loc=cv2.minMaxLoc(np.abs(q))
        if mx>best[0]:
            best=(mx,(loc[0]+(t.shape[1]-1)/2,loc[1]+(t.shape[0]-1)/2-120,r,angle))
    return a[120:],best

df=pd.read_csv(COHORT).drop_duplicates('frame_id').sort_values(['iso','speed_kmh','distance_m','frame']).groupby('condition',sort=True).head(1)
rec=[]; tiles=[]
for row in df.itertuples():
    a,(score,b)=detect(rd(ROOT/row.condition/row.frame)); x,y,r,angle=b
    im=cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_GRAY2BGR); cv2.circle(im,(round(x),round(y)),r,(0,255,0),2)
    cv2.putText(im,f'{row.condition} s={score:.2f}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
    rec.append(dict(condition=row.condition,score=score,x=x,y=y,r=r,angle=angle)); tiles.append(cv2.resize(im,(512,342)))
pd.DataFrame(rec).to_csv(OUT/'explore_template.csv',index=False)
for i in range(0,len(tiles),10):
    batch=tiles[i:i+10]
    if len(batch)<10: batch += [np.zeros_like(batch[0])]*(10-len(batch))
    sheet=np.vstack((np.hstack(batch[:5]),np.hstack(batch[5:])))
    cv2.imencode('.jpg',sheet)[1].tofile(OUT/f'template_{i:02d}.jpg')
