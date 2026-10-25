from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(r"<LOCAL_PATH_REDACTED>")
COHORT = Path(r"<LOCAL_PATH_REDACTED>")
OUT = Path(__file__).parent


def read_gray(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def detect(gray):
    small = cv2.resize(gray, (1024, 683), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=80, param2=45, minRadius=25, maxRadius=170,
    )
    return small, [] if circles is None else circles[0]


df = pd.read_csv(COHORT)
frames = df.drop_duplicates("frame_id").sort_values(["iso", "speed_kmh", "distance_m", "frame"])
frames = frames.groupby("condition", sort=True).head(1)
records=[]
tiles=[]
for row in frames.itertuples():
    p=ROOT/row.condition/row.frame
    a=read_gray(p)
    small, circles=detect(a)
    b=cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    # Score the two-light/two-dark diagonal quadrant pattern.  The chart is
    # frequently clipped by the top image edge, so valid pixels are counted
    # per quadrant rather than rejecting a partial circle.
    ranked=[]
    for x,y,r in circles:
        x,y,r=map(float,(x,y,r))
        if x > 0.38*b.shape[1] or y > 0.38*b.shape[0] or r > 125: continue
        yy,xx=np.ogrid[:b.shape[0],:b.shape[1]]
        rr=np.sqrt((xx-x)**2+(yy-y)**2)
        tl=(xx<x-0.10*r)&(yy<y-0.10*r)
        tr=(xx>x+0.10*r)&(yy<y-0.10*r)
        bl=(xx<x-0.10*r)&(yy>y+0.10*r)
        br=(xx>x+0.10*r)&(yy>y+0.10*r)
        inner=(rr<0.72*r)&(rr>0.18*r)
        vals=small.astype(float)
        masks=[inner&q for q in (tl,tr,bl,br)]
        if min(map(np.count_nonzero,masks)) < 20: continue
        means=[vals[m].mean() for m in masks]
        sep=abs((means[0]+means[3]-means[1]-means[2])/2)
        consistency=(abs(means[0]-means[3])+abs(means[1]-means[2]))/2
        score=sep-0.6*consistency
        ranked.append((score,x,y,r))
    ranked.sort(reverse=True)
    best=ranked[0] if ranked else (np.nan,)*4
    records.append(dict(condition=row.condition,n=len(circles),score=best[0],x=best[1],y=best[2],r=best[3]))
    if ranked:
        _,x,y,r=best
        cv2.circle(b,(round(x),round(y)),round(r),(0,255,0),2)
    cv2.putText(b,row.condition,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    tiles.append(cv2.resize(b,(512,342)))
pd.DataFrame(records).to_csv(OUT/'explore_hough.csv',index=False)
for i in range(0,len(tiles),10):
    sheet=np.vstack([np.hstack(tiles[j:j+5]) for j in range(i,min(i+10,len(tiles)),5)])
    cv2.imencode('.jpg',sheet)[1].tofile(OUT/f'explore_{i:02d}.jpg')
print(pd.DataFrame(records).to_string(index=False))
