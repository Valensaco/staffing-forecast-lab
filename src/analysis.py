"""Step 3: KPIs, confidence intervals, segments, real-world coverage, ROI + sensitivity."""
import pandas as pd, numpy as np, json
B='/home/claude/staffing-lab'; R_=f'{B}/results'
TARGET=3.48
R=pd.read_parquet(f'{R_}/test_forecasts.parquet')
R['wd']=R.date.dt.dayofweek
main=['2025-07','2025-10','2026-01','2026-04']
M=['A','Naive','B']
LABEL={'A':'Method A (weekday average)','Naive':'Baseline (same day last week)','B':'Method B (XGBoost forecast)'}
rng=np.random.default_rng(7)

def kpis(d,m):
    f=d[m].values; y=d.actual.values
    e=f-y
    ratio=y/np.where(f>0,f,np.nan)
    mult=np.nanquantile(ratio,0.95)                      # buffer multiplier needed for 95% of facility-days covered
    waste95=np.maximum(f*mult-y,0).sum()*TARGET/len(d)   # wasted hours per facility-day at 95% coverage
    short=np.maximum(-e,0).sum()*TARGET/len(d)
    waste=np.maximum(e,0).sum()*TARGET/len(d)
    return dict(n=len(d),mae_residents=np.abs(e).mean(),wape_pct=100*np.abs(e).sum()/y.sum(),bias_pct=100*e.sum()/y.sum(),
                risk_pct_short_gt5=100*np.mean(f<0.95*y),risk_pct_short_gt10=100*np.mean(f<0.90*y),
                short_hrs_per_facday=short,waste_hrs_per_facday=waste,buffer_pct_for_95cov=100*(mult-1),
                waste_hrs_per_facday_at95=waste95)
rows=[]
for (mth,season),d in R.groupby(['month','season']):
    for m in M: rows.append(dict(month=mth,season=season,method=LABEL[m],**kpis(d,m)))
bym=pd.DataFrame(rows); bym.to_csv(f'{R_}/metrics_by_month.csv',index=False)
dm=R[R.month.isin(main)]
pool=pd.DataFrame([dict(method=LABEL[m],**kpis(dm,m)) for m in M]); pool.to_csv(f'{R_}/metrics_main4_pooled.csv',index=False)
seas=pd.DataFrame([dict(season=s,method=LABEL[m],**kpis(d,m)) for s,d in dm.groupby('season') for m in M]); seas.to_csv(f'{R_}/metrics_by_season.csv',index=False)
print(pool.round(3).T.to_string())

# ---- bootstrap CI (resample facilities) ----
def agg(d):
    g=d.groupby('PROVNUM')
    out=pd.DataFrame({'y':g.actual.sum()})
    for m in M: out['e_'+m]=(d[m]-d.actual).abs().groupby(d.PROVNUM).sum()
    return out
def boot(d,B_=1000):
    a=agg(d); n=len(a); res={k:[] for k in ['A-B','Naive-B','A-Naive']}
    Y=a.y.values; E={m:a['e_'+m].values for m in M}
    idx=rng.integers(0,n,(B_,n))
    w=lambda m: 100*E[m][idx].sum(1)/Y[idx].sum(1)
    wA,wN,wB=w('A'),w('Naive'),w('B')
    out={}
    for nm,(x,z) in {'A-B':(wA,wB),'Naive-B':(wN,wB),'A-Naive':(wA,wN)}.items():
        dlt=x-z; out[nm]=(dlt.mean(),*np.quantile(dlt,[.025,.975]))
    return out
ci=[]
for s,d in list(dm.groupby('season'))+[('All 4 seasons',dm)]:
    for k,(mu,lo,hi) in boot(d).items(): ci.append(dict(season=s,diff_in_wape_pct_points=k,mean=mu,ci_low=lo,ci_high=hi))
ci=pd.DataFrame(ci); ci.to_csv(f'{R_}/ci_wape_difference.csv',index=False); print(ci.round(3).to_string())

# ---- segments ----
size=R.groupby('PROVNUM').actual.median(); R['size']=pd.qcut(R.PROVNUM.map(size),3,labels=['Small','Medium','Large'])
dm=R[R.month.isin(main)]
seg=[]
for col in ['wd','size']:
    for v,d in dm.groupby(col,observed=True):
        for m in M: seg.append(dict(segment=col,value=str(v),method=LABEL[m],**kpis(d,m)))
pd.DataFrame(seg).to_csv(f'{R_}/metrics_segments.csv',index=False)

# ---- real-world coverage: what facilities actually staffed vs need (all 36 months, clean data) ----
c=pd.read_parquet(f'{B}/data/clean.parquet'); c=c[c.hours_ok&c.census.notna()].copy()
c['need']=c.census*TARGET; c['wd']=c.date.dt.dayofweek; c['yr_q']=c.date.dt.to_period('Q').astype(str)
c['below']=c.total<0.95*c.need
rw=c.groupby('wd').agg(mean_hprd=('total',lambda s:0),share_below_target=('below','mean'),ctr_share=('total_ctr','sum'),tot=('total','sum')).reset_index()
rw['mean_hprd']=c.groupby('wd').apply(lambda d:d.total.sum()/d.census.sum()).values
rw['ctr_share']=100*rw.ctr_share/rw.tot; rw['share_below_target']*=100
rw['weekday']=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']; rw=rw[['weekday','mean_hprd','share_below_target','ctr_share']]
rw.to_csv(f'{R_}/real_staffing_by_weekday.csv',index=False); print(rw.round(2).to_string())
rq=c.groupby('yr_q').apply(lambda d:pd.Series(dict(mean_hprd=d.total.sum()/d.census.sum(),share_below_target=100*d.below.mean(),ctr_share=100*d.total_ctr.sum()/d.total.sum()))).reset_index()
rq.to_csv(f'{R_}/real_staffing_by_quarter.csv',index=False)
overall=dict(mean_hprd=c.total.sum()/c.census.sum(),share_below_target=100*c.below.mean(),ctr_share=100*c.total_ctr.sum()/c.total.sum(),
             weekend_hprd=c[c.wd>=5].total.sum()/c[c.wd>=5].census.sum(),weekday_hprd=c[c.wd<5].total.sum()/c[c.wd<5].census.sum(),
             mean_census=c.census.mean(),facility_days=len(c))
json.dump(overall,open(f'{R_}/real_staffing_overall.json','w'),indent=1); print(overall)

# ---- ROI (all money numbers use LABELED ASSUMPTIONS) ----
mm=pool.set_index('method')
wA=mm.loc[LABEL['A'],'waste_hrs_per_facday_at95']; wN=mm.loc[LABEL['Naive'],'waste_hrs_per_facday_at95']; wBb=mm.loc[LABEL['B'],'waste_hrs_per_facday_at95']
facs=R.PROVNUM.nunique()
roi=[]
for wage in [25,32,40]:
    for sw in [25000,75000,250000]:
        for tgt in [3.2,3.48,3.8]:
            s=tgt/TARGET
            for nfac,lab in [(20,'20-facility operator'),(facs,f'All {facs} Texas facilities')]:
                for frm,wf in [('Method A',wA),('Naive',wN)]:
                  for real in [0.25,0.5,1.0]:   # share of the wasted-buffer hours a real operator could actually flex away
                    saved_h=(wf-wBb)*s*nfac*365*real
                    saved=saved_h*wage
                    roi.append(dict(wage=wage,switch_cost=sw,target_hprd=tgt,realization=real,scope=lab,vs=frm,hours_saved_per_year=saved_h,
                                    dollars_saved_per_year=saved,roi_pct=100*(saved-sw)/sw))
roi=pd.DataFrame(roi); roi.to_csv(f'{R_}/roi_sensitivity.csv',index=False)
base_roi=roi[(roi.wage==32)&(roi.switch_cost==75000)&(roi.target_hprd==3.48)&(roi.scope=='20-facility operator')]
print(base_roi.round(1).to_string())

# ---- agency break-even: lean schedule + agency fills shortfall vs own-staff buffer ----
be=[]
for m in M:
    f=dm[m].values; y=dm.actual.values
    mult=np.nanquantile(y/np.where(f>0,f,np.nan),0.95)
    buffer_h=(f*(mult-1)).sum()*TARGET; short_h=np.maximum(y-f,0).sum()*TARGET
    be.append(dict(method=LABEL[m],buffer_hours_per_facday=buffer_h/len(dm),shortfall_hours_per_facday_lean=short_h/len(dm),
                   breakeven_agency_premium_x=buffer_h/short_h))
pd.DataFrame(be).to_csv(f'{R_}/agency_breakeven.csv',index=False); print(pd.DataFrame(be).round(3).to_string())
json.dump(dict(facilities_scored=int(facs),scored_rows_main4=int(len(dm)),target=TARGET,horizon_days=7),open(f'{R_}/run_meta.json','w'))
