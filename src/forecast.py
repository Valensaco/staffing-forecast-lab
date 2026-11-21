"""Step 2: walk-forward backtest.

Planning horizon: a schedule for day t is set 7 days earlier, using census known up to day t-7.
  Method A  : average census of the same weekday over the previous 8 weeks (>=4 weeks needed)
  Naive     : census of the same weekday last week (t-7)
  Method B  : XGBoost forecast of census(t), trained only on targets BEFORE the test month
Scheduled hours = forecast census x TARGET hours per resident day. Needed hours = actual census x TARGET.
"""
import pandas as pd, numpy as np, json, itertools, time, warnings
import xgboost as xgb
warnings.filterwarnings('ignore')
B='/home/claude/staffing-lab'
TARGET=3.48; H=7
df=pd.read_parquet(f'{B}/data/clean.parquet')
dates=pd.date_range('2023-07-01','2026-06-30')
Cdf=df.pivot(index='date',columns='PROVNUM',values='census').reindex(dates)
facs=Cdf.columns.values
V=Cdf.values  # dates x facilities
nD,nF=V.shape
def shift(a,k):
    out=np.full_like(a,np.nan); out[k:]=a[:-k]; return out
lags=np.stack([shift(V,H*k) for k in range(1,9)])      # lag k weeks: census at t-7k  (known at origin t-7)
cnt=(~np.isnan(lags)).sum(0)
with np.errstate(all='ignore'):
    A=np.where(cnt>=4,np.nanmean(lags,axis=0),np.nan)
    A4=np.where((~np.isnan(lags[:4])).sum(0)>=2,np.nanmean(lags[:4],axis=0),np.nan)
N=lags[0]                                                # census at t-7
roll=lambda w,mp,f: getattr(Cdf.rolling(w,min_periods=mp),f)().shift(H).values
r7=roll(7,4,'mean'); r28=roll(28,14,'mean'); r90=roll(90,30,'mean'); sd7=roll(7,4,'std')
dt=pd.DatetimeIndex(dates)
wd=np.repeat(dt.dayofweek.values[:,None],nF,1); mo=np.repeat(dt.month.values[:,None],nF,1)
dom=np.repeat(dt.day.values[:,None],nF,1)
with np.errstate(all='ignore'):
    feats={'c_o':N,'lag2':lags[1],'lag3':lags[2],'lag4':lags[3],'A8':A,'A4':A4,'r7':r7,'r28':r28,'r90':r90,'sd7':sd7,
           'trend7_28':r7/r28,'o_over_A':N/A,'wd':wd,'month':mo,'dom':dom,'first':(dom==1).astype(float),'last3':(dom>=29).astype(float)}
names=list(feats)
X=np.stack([feats[k].ravel() for k in names],1).astype(np.float32)
y=V.ravel(); base=N.ravel(); fA=A.ravel(); fN=N.ravel()
date_idx=np.repeat(np.arange(nD),nF); fac_idx=np.tile(np.arange(nF),nD)
tdate=dates[date_idx]
valid=(~np.isnan(y))&(~np.isnan(fN))&(~np.isnan(fA))      # scoreable / trainable rows
print('rows valid',valid.sum(),'of',len(y))

def fit(train_end, params, n_est=300):
    m=valid&(tdate<train_end)
    mdl=xgb.XGBRegressor(n_estimators=n_est,tree_method='hist',objective='reg:absoluteerror',subsample=0.8,colsample_bytree=0.8,
                         min_child_weight=50,n_jobs=2,random_state=7,**params)
    mdl.fit(X[m],(y-base)[m]); return mdl
def wape(f,yy): return np.abs(f-yy).sum()/yy.sum()

# ---- tuning: uses ONLY data before the first test month (Jul 2025) ----
t0=time.time()
val=valid&(tdate>=pd.Timestamp('2025-03-01'))&(tdate<pd.Timestamp('2025-07-01'))
grid=[dict(max_depth=d,learning_rate=lr) for d,lr in itertools.product([4,6,8],[0.05,0.1])]
tune=[]
for p in grid:
    mdl=fit(pd.Timestamp('2025-03-01'),p)
    pred=base[val]+mdl.predict(X[val])
    tune.append({**p,'val_wape':wape(pred,y[val])}); print(tune[-1],round(time.time()-t0))
best=min(tune,key=lambda r:r['val_wape']); bp={k:best[k] for k in ['max_depth','learning_rate']}
print('locked params',bp, 'A on val',wape(fA[val],y[val]),'naive on val',wape(fN[val],y[val]))
json.dump({'grid':tune,'locked':bp,'tuning_window':'train t<2025-03-01, validate 2025-03-01..2025-06-30'},open(f'{B}/results/tuning.json','w'),indent=1)

# ---- walk-forward test months ----
tests={'2025-07':'Summer','2025-10':'Fall','2026-01':'Winter','2026-04':'Spring',
       '2025-08':'Summer (adjacent)','2025-11':'Fall (adjacent)','2026-02':'Winter (adjacent)','2026-05':'Spring (adjacent)'}
rows=[]; imp=[]
for mth,season in tests.items():
    s=pd.Timestamp(mth+'-01'); e=s+pd.offsets.MonthBegin(1)
    mdl=fit(s,bp); m=valid&(tdate>=s)&(tdate<e)
    fB=base[m]+mdl.predict(X[m])
    part=pd.DataFrame({'month':mth,'season':season,'PROVNUM':facs[fac_idx[m]],'date':tdate[m],'actual':y[m],
                       'A':fA[m],'Naive':fN[m],'B':fB})
    rows.append(part); imp.append(pd.Series(mdl.feature_importances_,index=names,name=mth))
    print(mth,season,len(part),'WAPE A %.4f Naive %.4f B %.4f'%(wape(part.A,part.actual),wape(part.Naive,part.actual),wape(part.B,part.actual)),round(time.time()-t0))
R=pd.concat(rows,ignore_index=True)
R['B']=R.B.clip(lower=0)
R.to_parquet(f'{B}/results/test_forecasts.parquet')
pd.concat(imp,axis=1).to_csv(f'{B}/results/feature_importance.csv')
print('done',round(time.time()-t0))
