import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
B='/home/claude/staffing-lab'; R=f'{B}/results'; F=f'{B}/figures'
COL={'Method A (weekday average)':'#D55E00','Baseline (same day last week)':'#0072B2','Method B (XGBoost forecast)':'#009E73'}
SHORT={'Method A (weekday average)':'A: weekday average','Baseline (same day last week)':'Baseline: same day last week','Method B (XGBoost forecast)':'B: XGBoost forecast'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9a9a96','axes.labelcolor':'#3a3a38','xtick.color':'#3a3a38','ytick.color':'#3a3a38','axes.facecolor':'#fcfcfb','figure.facecolor':'#fcfcfb'})
def style(ax):
    ax.grid(axis='y',color='#e6e6e2',lw=0.8); ax.set_axisbelow(True)
def bars(ax,vals,fmt):
    for b in ax.patches:
        ax.annotate(fmt%b.get_height(),(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='bottom',fontsize=9,color='#3a3a38',xytext=(0,2),textcoords='offset points')
s=pd.read_csv(f'{R}/metrics_by_season.csv'); order=['Summer','Fall','Winter','Spring']
fig,ax=plt.subplots(figsize=(9,4.8)); w=0.26
for i,m in enumerate(COL):
    v=[s[(s.season==se)&(s.method==m)].wape_pct.iloc[0] for se in order]
    ax.bar(np.arange(4)+(i-1)*(w+0.02),v,w,color=COL[m],label=SHORT[m],edgecolor='#fcfcfb',linewidth=2)
bars(ax,None,'%.2f'); ax.set_xticks(range(4)); ax.set_xticklabels(order); ax.set_ylabel('Planning error (% of residents)'); style(ax)
ax.set_title('Planning error, 7 days ahead, by season (lower is better)',loc='left',fontsize=12); ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(0.5,-0.22),fontsize=9,ncol=3); ax.set_ylim(0,4.4)
fig.tight_layout(); fig.savefig(f'{F}/fig1_planning_error_by_season.png',dpi=160); plt.close()
p=pd.read_csv(f'{R}/metrics_main4_pooled.csv')
for col,fn,title,ylab,fmt in [('risk_pct_short_gt5','fig2_risk_days_short.png','Risk: facility-days scheduled more than 5% below need (lower is better)','% of facility-days','%.1f'),
                              ('waste_hrs_per_facday_at95','fig3_wasted_hours_at_same_coverage.png','Wasted hours when each method must cover 95% of days (lower is better)','Hours per facility-day','%.1f')]:
    fig,ax=plt.subplots(figsize=(8,4.6))
    ax.bar([SHORT[m] for m in p.method],p[col],color=[COL[m] for m in p.method],width=0.55)
    bars(ax,None,fmt); ax.set_ylabel(ylab); ax.set_title(title,loc='left',fontsize=11.5); style(ax); ax.set_ylim(0,p[col].max()*1.18)
    plt.setp(ax.get_xticklabels(),fontsize=9.5); fig.tight_layout(); fig.savefig(f'{F}/{fn}',dpi=160); plt.close()
w_=pd.read_csv(f'{R}/real_staffing_by_weekday.csv')
fig,ax=plt.subplots(figsize=(8.6,4.6)); ax.bar(w_.weekday,w_.mean_hprd,color='#0072B2',width=0.6)
ax.axhline(3.48,color='#3a3a38',lw=1.2,ls='--'); ax.text(6.45,3.52,'3.48 benchmark',ha='right',fontsize=9,color='#3a3a38')
for b in ax.patches: ax.annotate('%.2f'%b.get_height(),(b.get_x()+b.get_width()/2,b.get_height()),ha='center',va='top',fontsize=10,color='white',xytext=(0,-6),textcoords='offset points')
ax.set_ylim(0,4.1); ax.set_ylabel('Nursing hours per resident per day'); style(ax)
ax.set_title('Actual staffing by weekday: weekends run about 16% lower',loc='left',fontsize=12)
fig.tight_layout(); fig.savefig(f'{F}/fig4_actual_staffing_by_weekday.png',dpi=160); plt.close()
fig,ax=plt.subplots(figsize=(8.6,4.6)); ax.bar(w_.weekday,w_.share_below_target,color='#0072B2',width=0.6)
bars(ax,None,'%.0f%%'); ax.set_ylim(0,100); ax.set_ylabel('% of facility-days below target'); style(ax)
ax.set_title('Facility-days staffed more than 5% below the 3.48 benchmark',loc='left',fontsize=12)
fig.tight_layout(); fig.savefig(f'{F}/fig5_days_below_benchmark_by_weekday.png',dpi=160); plt.close()
print('ok')
