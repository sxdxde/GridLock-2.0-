import pandas as pd, numpy as np, warnings, time
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
tr=pd.read_csv('dataset/train.csv'); te=pd.read_csv('dataset/test.csv')
def ts2min(s): h,m=s.split(':'); return int(h)*60+int(m)
for df in (tr,te): df['tmin']=df['timestamp'].map(ts2min)
_b32='0123456789bcdefghjkmnpqrstuvwxyz'
def decode(gh):
    a=[-90.,90.];o=[-180.,180.];e=True
    for c in gh:
        cd=_b32.index(c)
        for m in (16,8,4,2,1):
            if e: mid=(o[0]+o[1])/2; o[0 if cd&m else 1]=mid
            else: mid=(a[0]+a[1])/2; a[0 if cd&m else 1]=mid
            e=not e
    return (a[0]+a[1])/2,(o[0]+o[1])/2
dec={g:decode(g) for g in pd.concat([tr.geohash,te.geohash]).unique()}
def base(df):
    d=df.copy()
    d['lat']=d.geohash.map(lambda g:dec[g][0]);d['lon']=d.geohash.map(lambda g:dec[g][1])
    d['hour']=d.tmin//60
    for k in (1,2,3): d[f'sin{k}']=np.sin(2*np.pi*k*d.tmin/1440);d[f'cos{k}']=np.cos(2*np.pi*k*d.tmin/1440)
    return d
trf=base(tr);cat=['RoadType','LargeVehicles','Landmarks','Weather']
for c in cat: trf[c]=trf[c].astype('category')
y=trf['demand'].values.astype('float32'); trf['__y']=y
harm=[f'{p}{k}' for k in (1,2,3) for p in ('sin','cos')]
basef=['lat','lon','tmin','hour','NumberofLanes','Temperature']+harm+cat
def mk(): return lgb.LGBMRegressor(n_estimators=700,learning_rate=0.05,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,min_child_samples=20,n_jobs=-1,verbose=-1)
kf=KFold(5,shuffle=True,random_state=1)
def sm(d,k,s=10): g=d.groupby(k)['__y']; gm=g.mean();gc=g.count();gl=d['__y'].mean(); return (gm*gc+gl*s)/(gc+s),gl
def run(ght,knn,lbl):
    oof=np.zeros(len(trf)); t=time.time()
    for tri,vai in kf.split(trf):
        A=trf.iloc[tri].copy();B=trf.iloc[vai].copy();F=list(basef)
        enc,gl=sm(A,'geohash'); A['gh_te']=A.geohash.map(enc).fillna(gl);B['gh_te']=B.geohash.map(enc).fillna(gl)
        st=A.groupby('geohash')['__y'].std();A['gh_std']=A.geohash.map(st).fillna(0);B['gh_std']=B.geohash.map(st).fillna(0)
        F+=['gh_te','gh_std']
        if ght:
            e2,g2=sm(A,['geohash','hour'],5)
            A['ghr']=pd.Series(A.set_index(['geohash','hour']).index.map(e2),index=A.index).fillna(A['gh_te'])
            B['ghr']=pd.Series(B.set_index(['geohash','hour']).index.map(e2),index=B.index).fillna(B['gh_te']); F+=['ghr']
        if knn:
            kn=KNeighborsRegressor(n_neighbors=20,weights='distance').fit(A[['lat','lon']],A['__y'])
            A['knn']=kn.predict(A[['lat','lon']]);B['knn']=kn.predict(B[['lat','lon']]);F+=['knn']
        m=mk();m.fit(A[F],y[tri],categorical_feature=cat);oof[vai]=m.predict(B[F])
    print(lbl, round(r2_score(y,oof),4), f"({time.time()-t:.0f}s)",flush=True)
run(False,False,"gh_te+std:")
run(True,False,"+gh*hour TE:")
run(True,True,"+gh*hour TE +KNN:")
