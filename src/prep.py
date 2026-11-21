"""Step 1: combine the 12 CMS PBJ Texas quarters and clean them."""
import pandas as pd, numpy as np, glob, os, json
BASE='/home/claude'
Q=['q3_2023','q4_2023','q1_2024','q2_2024','q3_2024','q4_2024','q1_2025','q2_2025','q3_2025','q4_2025','q1_2026']
paths=[glob.glob(f'{BASE}/pbj_{q}/*.csv')[0] for q in Q]+[f'{BASE}/pbj_raw/PBJ_Daily_Nurse_Staffing_Q2_2026.csv']
ROLES=['RNDON','RNadmin','RN','LPNadmin','LPN','CNA','NAtrn','MedAide']
def rd(p):
    try: return pd.read_csv(p,dtype={'PROVNUM':str})
    except UnicodeDecodeError: return pd.read_csv(p,dtype={'PROVNUM':str},encoding='latin-1')
df=pd.concat([rd(p) for p in paths],ignore_index=True)
log={}
log['rows_raw']=len(df)
df['PROVNAME']=df['PROVNAME'].astype(str).str.replace('\xa0',' ').str.strip()
df['date']=pd.to_datetime(df['WorkDate'].astype(str),format='%Y%m%d')
log['dup_rows']=int(df.duplicated(['PROVNUM','date']).sum())
df=df.drop_duplicates(['PROVNUM','date'])
log['state_values']=df.STATE.unique().tolist()
df['total']=df[[f'Hrs_{r}' for r in ROLES]].sum(axis=1)
df['total_emp']=df[[f'Hrs_{r}_emp' for r in ROLES]].sum(axis=1)
df['total_ctr']=df[[f'Hrs_{r}_ctr' for r in ROLES]].sum(axis=1)
log['emp_plus_ctr_mismatch_rows']=int((abs(df.total-(df.total_emp+df.total_ctr))>0.05).sum())
df['census']=df['MDScensus']
df['hprd']=df.total/df.census
log['census_le0']=int((df.census<=0).sum())
log['hprd_gt10']=int((df.hprd>10).sum()); log['hprd_gt20']=int((df.hprd>20).sum())
log['hprd_lt0.5']=int((df.hprd<0.5).sum()); log['hprd_eq0']=int((df.total==0).sum())
log['hprd_max']=float(df.hprd.max())
# calendar coverage
full=pd.date_range(df.date.min(),df.date.max())
log['calendar_days']=len(full)
cov=df.groupby('PROVNUM').date.nunique()
log['facilities']=int(cov.size)
log['facilities_full_coverage']=int((cov==len(full)).sum())
log['facility_days_present']=len(df); log['facility_days_possible']=int(cov.size*len(full))
# census jumps
df=df.sort_values(['PROVNUM','date'])
df['census_prev']=df.groupby('PROVNUM').census.shift(1)
df['gap']=(df.date-df.groupby('PROVNUM').date.shift(1)).dt.days
jump=(df.census/df.census_prev)
log['census_jump_gt50pct']=int(((jump>1.5)|(jump<0.5)).sum())
print(json.dumps(log,indent=1))
print(df.hprd.describe(percentiles=[.001,.01,.5,.99,.999]))
print(df.groupby(df.date.dt.dayofweek).hprd.mean().round(3))
df.to_parquet(f'{BASE}/staffing-lab/data/combined_raw.parquet')
json.dump(log,open(f'{BASE}/staffing-lab/results/cleaning_log_step1.json','w'),indent=1)
