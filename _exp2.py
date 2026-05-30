import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
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
    d['sin_t']=np.sin(2*np.pi*d.tmin/1440);d['cos_t']=np.cos(2*np.pi*d.tmin/1440)
    d['sin_2t']=np.sin(4*np.pi*d.tmin/1440);d['cos_2t']=np.cos(4*np.pi*d.tmin/1440)
    return d
trf=base(tr);tef=base(te)
cat=['RoadType','LargeVehicles','Landmarks','Weather']
for c in cat: trf[c]=trf[c].astype('category');tef[c]=tef[c].astype('category')
y=trf['demand'].values
def mk(n=900):
    return lgb.LGBMRegressor(n_estimators=n,learning_rate=0.03,num_leaves=127,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,min_child_samples=20,
        reg_lambda=1.0,n_jobs=-1,verbose=-1)
kf=KFold(5,shuffle=True,random_state=1)

def cv(feats,extra=None,n=900):
    oof=np.zeros(len(trf))
    for tri,vai in kf.split(trf):
        Xtr=trf.iloc[tri].copy();Xv=trf.iloc[vai].copy()
        if extra: extra(Xtr,Xv,tri,vai)
        m=mk(n);m.fit(Xtr[feats],y[tri],categorical_feature=[c for c in cat if c in feats])
        oof[vai]=m.predict(Xv[feats])
    return r2_score(y,oof)

f0=['lat','lon','tmin','hour','sin_t','cos_t','sin_2t','cos_2t','NumberofLanes','Temperature']+cat
print("f0 (base+harmonics, deeper):", round(cv(f0),4))

# add geohash target encoding (out of fold within train fold via nested smoothing)
def add_te(Xtr,Xv,tri,vai):
    g=Xtr.groupby('geohash')['__y']
    gm=g.mean();gc=g.count();glob=Xtr['__y'].mean();sm=10
    enc=(gm*gc+glob*sm)/(gc+sm)
    Xtr['gh_te']=Xtr['geohash'].map(enc).fillna(glob)
    Xv['gh_te']=Xv['geohash'].map(enc).fillna(glob)
    # geohash demand std and count
    Xtr['gh_std']=Xtr['geohash'].map(g.std()).fillna(0); Xv['gh_std']=Xv['geohash'].map(g.std()).fillna(0)
trf['__y']=y
f1=f0+['gh_te','gh_std']
print("f1 (+geohash target enc + std):", round(cv(f1,add_te),4))
