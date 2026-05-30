import pandas as pd, numpy as np, warnings
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
    lat=(-90.,90.);lon=(-180.,180.);even=True
    for c in gh:
        cd=_b32.index(c)
        for mask in (16,8,4,2,1):
            if even: mid=(lon[0]+lon[1])/2;lon=(mid,lon[1]) if cd&mask else (lon[0],mid)
            else: mid=(lat[0]+lat[1])/2;lat=(mid,lat[1]) if cd&mask else (lat[0],mid)
            even=not even
    return (lat[0]+lat[1])/2,(lon[0]+lon[1])/2
dec={g:decode(g) for g in pd.concat([tr.geohash,te.geohash]).unique()}
def base(df):
    d=df.copy()
    d['lat']=d.geohash.map(lambda g:dec[g][0]);d['lon']=d.geohash.map(lambda g:dec[g][1])
    d['hour']=d.tmin//60
    for k in (1,2,3):
        d[f'sin{k}']=np.sin(2*np.pi*k*d.tmin/1440);d[f'cos{k}']=np.cos(2*np.pi*k*d.tmin/1440)
    return d
trf=base(tr);tef=base(te)
cat=['RoadType','LargeVehicles','Landmarks','Weather']
for c in cat: trf[c]=trf[c].astype('category')
y=trf['demand'].values; trf['__y']=y
def mk(n=1200):
    return lgb.LGBMRegressor(n_estimators=n,learning_rate=0.03,num_leaves=127,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.7,min_child_samples=20,
        reg_lambda=1.0,n_jobs=-1,verbose=-1)
kf=KFold(5,shuffle=True,random_state=1)
harm=[f'{p}{k}' for k in (1,2,3) for p in ('sin','cos')]
base_f=['lat','lon','tmin','hour','NumberofLanes','Temperature']+harm+cat

def smooth_te(tr_df,key,sm=10):
    g=tr_df.groupby(key)['__y']; gm=g.mean(); gc=g.count(); glob=tr_df['__y'].mean()
    return ((gm*gc+glob*sm)/(gc+sm)), glob

def run(use_ght=False, use_knn=False, label=''):
    oof=np.zeros(len(trf))
    for tri,vai in kf.split(trf):
        A=trf.iloc[tri].copy(); B=trf.iloc[vai].copy(); feats=list(base_f)
        # geohash TE
        enc,glob=smooth_te(A,'geohash'); A['gh_te']=A.geohash.map(enc).fillna(glob); B['gh_te']=B.geohash.map(enc).fillna(glob)
        gstd=A.groupby('geohash')['__y'].std(); A['gh_std']=A.geohash.map(gstd).fillna(0); B['gh_std']=B.geohash.map(gstd).fillna(0)
        feats+=['gh_te','gh_std']
        if use_ght:
            for col,k in [('gh_hr',['geohash','hour'])]:
                enc2,gl2=smooth_te(A,k,sm=5)
                A[col]=A.set_index(k).index.map(enc2).astype(float); B[col]=B.set_index(k).index.map(enc2).astype(float)
                A[col]=A[col].fillna(A['gh_te']); B[col]=B[col].fillna(B['gh_te']); feats.append(col)
        if use_knn:
            kn=KNeighborsRegressor(n_neighbors=15,weights='distance')
            kn.fit(A[['lat','lon']],A['__y'])
            A['knn']=kn.predict(A[['lat','lon']]); B['knn']=kn.predict(B[['lat','lon']]); feats.append('knn')
        m=mk(); m.fit(A[feats],y[tri],categorical_feature=cat); oof[vai]=m.predict(B[feats])
    print(label, round(r2_score(y,oof),4))
    return oof

run(False,False,"f1 deeper (gh_te+std):")
run(True,False,"+ geohash*hour TE:")
run(True,True,"+ geohash*hour TE + KNN spatial:")
