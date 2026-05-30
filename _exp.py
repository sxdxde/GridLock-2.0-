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
    lat=(-90.,90.); lon=(-180.,180.); even=True
    for c in gh:
        cd=_b32.index(c)
        for mask in (16,8,4,2,1):
            if even: mid=(lon[0]+lon[1])/2; lon=(mid,lon[1]) if cd&mask else (lon[0],mid)
            else: mid=(lat[0]+lat[1])/2; lat=(mid,lat[1]) if cd&mask else (lat[0],mid)
            even=not even
    return (lat[0]+lat[1])/2,(lon[0]+lon[1])/2
dec={g:decode(g) for g in pd.concat([tr.geohash,te.geohash]).unique()}

def base_feats(df):
    d=df.copy()
    d['lat']=d.geohash.map(lambda g:dec[g][0]); d['lon']=d.geohash.map(lambda g:dec[g][1])
    d['hour']=d.tmin//60
    d['sin_t']=np.sin(2*np.pi*d.tmin/1440); d['cos_t']=np.cos(2*np.pi*d.tmin/1440)
    return d
trf=base_feats(tr); tef=base_feats(te)
cat=['RoadType','LargeVehicles','Landmarks','Weather']
for c in cat:
    trf[c]=trf[c].astype('category'); tef[c]=tef[c].astype('category')
feats=['lat','lon','tmin','hour','sin_t','cos_t','NumberofLanes','Temperature']+cat

def mk(n=700):
    return lgb.LGBMRegressor(n_estimators=n,learning_rate=0.05,num_leaves=63,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.8,min_child_samples=30,n_jobs=-1,verbose=-1)

# A) honest day-holdout: train day48 -> day49-night
d48=trf[trf.day==48]; d49=trf[trf.day==49]
m=mk(); m.fit(d48[feats],d48['demand'],categorical_feature=cat)
print("A) Honest day48->day49night R2:", round(r2_score(d49['demand'],m.predict(d49[feats])),4))

# B) random KFold on day48 (proxy for leaderboard)
kf=KFold(5,shuffle=True,random_state=1); oof=np.zeros(len(d48)); y=d48['demand'].values; X=d48[feats]
for tri,vai in kf.split(X):
    m=mk(); m.fit(X.iloc[tri],y[tri],categorical_feature=cat); oof[vai]=m.predict(X.iloc[vai])
print("B) day48 random 5-fold R2:", round(r2_score(y,oof),4))

# C) all train random KFold
y=trf['demand'].values; X=trf[feats]; oof=np.zeros(len(trf))
for tri,vai in kf.split(X):
    m=mk(); m.fit(X.iloc[tri],y[tri],categorical_feature=cat); oof[vai]=m.predict(X.iloc[vai])
print("C) full-train random 5-fold R2:", round(r2_score(y,oof),4))
