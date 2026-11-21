import pandas as pd, json
B='/home/claude/staffing-lab'; R=f'{B}/results'
p=pd.read_csv(f'{R}/metrics_main4_pooled.csv').set_index('method')
s=pd.read_csv(f'{R}/metrics_by_season.csv'); ci=pd.read_csv(f'{R}/ci_wape_difference.csv')
rw=pd.read_csv(f'{R}/real_staffing_by_weekday.csv'); ov=json.load(open(f'{R}/real_staffing_overall.json'))
roi=pd.read_csv(f'{R}/roi_sensitivity.csv'); be=pd.read_csv(f'{R}/agency_breakeven.csv').set_index('method')
cl=json.load(open(f'{R}/cleaning_log.json')); tu=json.load(open(f'{R}/tuning.json')); meta=json.load(open(f'{R}/run_meta.json'))
A,N,Bm=p.index[0],p.index[1],p.index[2]
def row(col,fmt): return ' | '.join(fmt%p.loc[m,col] for m in [A,N,Bm])
def cirow(season,key):
    r=ci[(ci.season==season)&(ci.diff_in_wape_pct_points==key)].iloc[0]; return r['mean'],r.ci_low,r.ci_high
a_b=cirow('All 4 seasons','A-B'); n_b=cirow('All 4 seasons','Naive-B')
seas_tab='\n'.join(f"| {se} | "+' | '.join('%.2f%%'%s[(s.season==se)&(s.method==m)].wape_pct.iloc[0] for m in [A,N,Bm])+' |' for se in ['Summer','Fall','Winter','Spring'])
wk_tab='\n'.join(f"| {r.weekday} | {r.mean_hprd:.2f} | {r.share_below_target:.0f}% | {r.ctr_share:.1f}% |" for r in rw.itertuples())
base=roi[(roi.wage==32)&(roi.switch_cost==75000)&(roi.target_hprd==3.48)&(roi.scope=='20-facility operator')]
def rr(vs,real): r=base[(base.vs==vs)&(base.realization==real)].iloc[0]; return r.dollars_saved_per_year,r.roi_pct
lo=roi[(roi.scope=='20-facility operator')&(roi.vs=='Naive')&(roi.realization==0.25)]
txt=f"""# Nursing home staffing: does a forecast beat the weekday average?

A backtest on **real data**: three years of daily nurse staffing from Texas nursing homes (CMS Payroll Based Journal, July 2023 to June 2026, about 1.27 million facility-days, {cl['facilities_after_cleaning']:,} facilities).

I built this to learn how staffing demand, schedule efficiency and workload distribution are analyzed, using a field where real data is public. The questions are the same ones an airport operations team asks: *how many people do we need, when, and is a new planning tool worth it?*

## Result in one paragraph

Scheduling from a **forecast model (XGBoost)** was more accurate than the usual **weekday-average** rule in every season, every weekday and every facility size. Pooled over four test months (one per season), planning error fell from {p.loc[A,'wape_pct']:.2f}% to {p.loc[Bm,'wape_pct']:.2f}% of residents, and the share of facility-days scheduled more than 5% short fell from {p.loc[A,'risk_pct_short_gt5']:.1f}% to {p.loc[Bm,'risk_pct_short_gt5']:.1f}%. **The honest catch:** most of that gain comes from simply using *last week's census* instead of a weekday average. The XGBoost model adds a small extra improvement ({n_b[0]:.2f} percentage points, 95% interval {n_b[1]:.2f} to {n_b[2]:.2f}), which is real but small. Whether it is worth building depends on the size of the operator (see ROI).

## Methods compared

Planning horizon: schedules are set **7 days ahead**, using only census known up to that day.

| | Method | How it forecasts census for day t |
|---|---|---|
| **A** | Weekday average (the usual way) | Average census of the same weekday over the previous 8 weeks |
| **Baseline** | Same day last week | Census on day t-7 (a free, no-model benchmark) |
| **B** | XGBoost forecast (the new tool) | Predicts census change from recent levels, trend, weekday, month, facility size |

Scheduled hours = forecast census x **3.48** hours per resident per day. Needed hours = actual census x 3.48. The 3.48 benchmark is an **assumption** (a figure used in CMS's 2024 minimum-staffing rule, not a requirement I am claiming applies). It scales hours and dollars but **does not change which method wins**, because both methods use the same target.

## How it was tested (walk-forward, no peeking)

For each test month the models are retrained using **only data from before that month**:

| Test month | Season | Trained on |
|---|---|---|
| Jul 2025 | Summer | Jul 2023 to Jun 2025 |
| Oct 2025 | Fall | up to Sep 2025 |
| Jan 2026 | Winter | up to Dec 2025 |
| Apr 2026 | Spring | up to Mar 2026 |

An adjacent month for each season (Aug, Nov, Feb, May) was also run as a check, so one odd month cannot decide a season. Model settings were tuned **once**, on Mar to Jun 2025 (before any test month), then locked (max depth {tu['locked']['max_depth']}, learning rate {tu['locked']['learning_rate']}). All three methods are scored on the **same** {meta['scored_rows_main4']:,} facility-days.

## Results (4 test months, pooled)

| KPI | A: weekday average | Baseline: last week | B: XGBoost |
|---|---|---|---|
| Planning error (% of residents) | {row('wape_pct','%.2f%%')} |
| Mean miss (residents) | {row('mae_residents','%.2f')} |
| **Risk**: facility-days scheduled >5% below need | {row('risk_pct_short_gt5','%.1f%%')} |
| Risk: facility-days scheduled >10% below need | {row('risk_pct_short_gt10','%.2f%%')} |
| Buffer needed to cover 95% of days | {row('buffer_pct_for_95cov','%.1f%%')} |
| **Wasted hours per facility-day at 95% coverage** | {row('waste_hrs_per_facday_at95','%.1f')} |

Difference in planning error, with a 95% interval from resampling facilities: **A minus B = {a_b[0]:.2f} points ({a_b[1]:.2f} to {a_b[2]:.2f})**; **Baseline minus B = {n_b[0]:.2f} points ({n_b[1]:.2f} to {n_b[2]:.2f})**.

Planning error by season:

| Season | A | Baseline | B |
|---|---|---|---|
{seas_tab}

![Planning error by season](figures/fig1_planning_error_by_season.png)

## What actually happens in the real data (workload distribution)

Residents do not change by weekday, but staffing does. Using reported hours (days with implausible hours excluded):

| Weekday | Hours per resident | Days >5% below 3.48 | Agency (contract) share of hours |
|---|---|---|---|
{wk_tab}

Weekends run about **{100*(1-ov['weekend_hprd']/ov['weekday_hprd']):.0f}% lower** ({ov['weekend_hprd']:.2f} vs {ov['weekday_hprd']:.2f} hours per resident). Across all days, {ov['share_below_target']:.0f}% of facility-days were staffed more than 5% below the 3.48 benchmark, and agency staff made up only {ov['ctr_share']:.1f}% of hours. This is a schedule-efficiency finding that a forecast alone does not fix: demand is flat across the week while staffing is not.

![Actual staffing by weekday](figures/fig4_actual_staffing_by_weekday.png)

## ROI (assumptions are labeled and flexed)

The file has **no wages and no cost to switch**, so these are placeholders, not facts. Scenario: an operator with **20 facilities**, a blended cost of **$32 per nursing hour**, a one-time cost of **$75,000** to build and maintain the tool, and 3.48 hours per resident. Savings come from fewer wasted buffer hours at the same 95% coverage. Because not every hour can really be flexed day to day, I show what happens if only 25%, 50% or all of those hours can be cut.

| Switching from | 25% of hours flexible | 50% | 100% (upper bound) |
|---|---|---|---|
| **A (weekday average)** to B | ${rr('Method A',0.25)[0]:,.0f}/yr (ROI {rr('Method A',0.25)[1]:,.0f}%) | ${rr('Method A',0.5)[0]:,.0f}/yr (ROI {rr('Method A',0.5)[1]:,.0f}%) | ${rr('Method A',1.0)[0]:,.0f}/yr (ROI {rr('Method A',1.0)[1]:,.0f}%) |
| **Baseline (last week)** to B | ${rr('Naive',0.25)[0]:,.0f}/yr (ROI {rr('Naive',0.25)[1]:,.0f}%) | ${rr('Naive',0.5)[0]:,.0f}/yr (ROI {rr('Naive',0.5)[1]:,.0f}%) | ${rr('Naive',1.0)[0]:,.0f}/yr (ROI {rr('Naive',1.0)[1]:,.0f}%) |

ROI = (savings minus switching cost) divided by switching cost. Full grid (wages $25 to $40, switching cost $25K to $250K, targets 3.2 to 3.8, 20 facilities vs all Texas) is in `results/roi_sensitivity.csv`.

**Reading it:** moving off the weekday average pays off under almost any assumption. Moving from the free "last week" rule to XGBoost is marginal for a small operator (ROI near break-even at 25% flexibility, and negative when the switching cost is high), and only clearly pays at larger scale. My recommendation would be: **stop using the weekday average first; adopt the model only if the operator is large enough to absorb the build cost.**

**Agency vs buffer:** holding a buffer in own staff to cover 95% of days costs about {be.loc[B and Bm.replace('x','x'),'buffer_hours_per_facday'] if False else be.loc[Bm,'buffer_hours_per_facday']:.1f} extra hours per facility-day, while filling the shortfalls of a lean schedule with agency staff takes about {be.loc[Bm,'shortfall_hours_per_facday_lean']:.1f}. Agency is cheaper unless it costs more than about **{be.loc[Bm,'breakeven_agency_premium_x']:.1f}x** the normal rate (this assumes agency staff can be found on short notice).

## Data cleaning (every rule logged)

| Rule | Rows touched |
|---|---|
| Raw rows loaded (12 quarterly files, Texas only, no duplicates) | {cl['rows_raw']:,} |
| Census of 0 or less set to missing (new facilities ramping up) | {cl['rules']['R1_census_le_0_set_missing']['rows']:,} |
| Facilities dropped for chronic implausible hours (>10 hours per resident on over 10% of days) | {len(cl['rules']['R2_facilities_dropped_chronic_hprd_gt_10']['facilities'])} facilities, {cl['rules']['R2_facilities_dropped_chronic_hprd_gt_10']['rows']:,} rows |
| Daily hours set to missing (over 10 or under 0.5 hours per resident, e.g. a 35-hour-per-resident entry) | {cl['rules']['R3_hours_implausible_set_missing']['rows']:,} |
| Isolated census spikes | {cl['rules']['R4_isolated_census_spikes_set_missing']['rows']} found |
| Rows after cleaning | {cl['rows_after_cleaning']:,} ({cl['facilities_after_cleaning']:,} facilities) |

Census step changes at the start of a month were kept (27 rows jumped by more than 50%): they look like real level shifts that any forecast has to live with.

## Limits (please read)

- **No wages and no quality outcomes** (falls, complaints) in the data, so "better" means coverage and cost, not patient results. All dollar figures rest on labeled assumptions.
- **The 3.48 hours benchmark is an assumption**, used consistently for all methods.
- This is a **backtest**: it shows what would have happened on past data, not a guarantee.
- The models forecast **census (number of residents)**, not workload by shift or by role. Real planning also depends on case mix, shifts and labor rules.
- **One month per season** is tested; the adjacent-month check supports but does not replace more years.
- The weekday-average method is weak mainly because census does not follow a weekday pattern. A smarter baseline (last week) closes most of the gap, which is why that comparison is included.

## Reproduce

Data: CMS Payroll Based Journal Daily Nurse Staffing (data.cms.gov), filtered to STATE = TX, quarters 2023Q3 to 2026Q2. Raw files are not included here (they are public and large); place them under `data/raw/` and run in order: `src/prep.py`, `src/clean.py`, `src/forecast.py`, `src/analysis.py`, `src/figures.py`. Python 3.13, pandas, numpy, xgboost, scikit-learn, matplotlib.

## Files

`src/` code, `results/` all result tables, `figures/` charts, `powerbi/` data and layout for the dashboard.
"""
open(f'{B}/README.md','w').write(txt); print(len(txt))
