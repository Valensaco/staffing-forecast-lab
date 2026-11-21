"""Step 1b: cleaning rules. Every rule logs how many rows it touches."""
import pandas as pd, numpy as np, json
B='/home/claude/staffing-lab'
df=pd.read_parquet(f'{B}/data/combined_raw.parquet')
log=json.load(open(f'{B}/results/cleaning_log_step1.json'))
c=log.copy(); c['rules']={}
n0=len(df)
# R1 invalid census (<=0): cannot compute hours per resident, cannot be forecast
m=df.census<=0
c['rules']['R1_census_le_0_set_missing']={'rows':int(m.sum()),'facilities':int(df[m].PROVNUM.nunique())}
df.loc[m,'census']=np.nan
df['hprd']=df.total/df.census
# R2 chronic implausible hours -> drop facility entirely (data not trustworthy)
bad_share=(df.hprd>10).groupby(df.PROVNUM).mean()
bad_fac=bad_share[bad_share>0.10].index.tolist()
c['rules']['R2_facilities_dropped_chronic_hprd_gt_10']={'facilities':bad_fac,'rows':int(df.PROVNUM.isin(bad_fac).sum())}
df=df[~df.PROVNUM.isin(bad_fac)].copy()
# R3 implausible daily hours (hprd>10 or <0.5): hours set missing, census kept
m=(df.hprd>10)|(df.hprd<0.5)
c['rules']['R3_hours_implausible_set_missing']={'rows':int(m.sum()),'of_which_gt10':int((df.hprd>10).sum()),'of_which_lt0.5':int((df.hprd<0.5).sum())}
for col in ['total','total_emp','total_ctr']: df.loc[m,col]=np.nan
df['hours_ok']=~m
# R4 isolated census spikes: differs >50% from neighbours that agree with each other
df=df.sort_values(['PROVNUM','date'])
g=df.groupby('PROVNUM')
prev=g.census.shift(1); nxt=g.census.shift(-1)
nbr=(prev+nxt)/2
ok_nbr=((prev-nxt).abs()/nbr<0.2)
spike=ok_nbr&((df.census-nbr).abs()/nbr>0.5)
c['rules']['R4_isolated_census_spikes_set_missing']={'rows':int(spike.sum())}
df.loc[spike,'census']=np.nan
c['rows_after_cleaning']=len(df); c['facilities_after_cleaning']=int(df.PROVNUM.nunique())
c['note_kept']='Month-start census step changes (27 rows >50% jump) kept: they look like real level shifts (admissions batches / census recalculation), which any forecast must live with.'
keep=['PROVNUM','PROVNAME','CITY','COUNTY_NAME','date','census','total','total_emp','total_ctr','hours_ok']
df[keep].to_parquet(f'{B}/data/clean.parquet')
json.dump(c,open(f'{B}/results/cleaning_log.json','w'),indent=1,default=str)
print(json.dumps(c['rules'],indent=1,default=str)); print(c['rows_after_cleaning'],c['facilities_after_cleaning'])
