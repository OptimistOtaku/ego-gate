"""Derive manuscript tables/figure from committed observations, never invented values."""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'paper'/'ieee'
DEST.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT/'results/digits/per_seed.csv').open()))
names = {'random':'Random', 'stratified_random':'Balanced random', 'doubt':'Doubt',
         'curiosity':'Curiosity', 'egogate':'EgoGate', 'embedding_kcenter':'Embedding k-center',
         'egogate_diverse':'EgoGate + diversity', 'full':'Full memory', 'none':'No replay'}
groups = {k:sorted([r for r in rows if r['condition']==k],key=lambda r:int(r['seed'])) for k in names}
table=[]
for k in ['none','random','stratified_random','embedding_kcenter','curiosity','egogate','egogate_diverse','doubt','full']:
    data=groups[k]
    vals=[]
    for metric in ['task_a_post','forgetting','task_b_post']:
        v=np.array([float(r[metric])*100 for r in data])
        vals.append(f'{v.mean():.2f} $\\pm$ {v.std(ddof=1):.2f}')
    table.append(names[k]+' & '+('720' if k=='full' else '0' if k=='none' else '40')+' & '+' & '.join(vals)+r' \\')
(DEST/'results_table.tex').write_text('\n'.join(table),encoding='utf-8')
main=DEST/'main.tex'
if main.exists():
    text=main.read_text()
    text=re.sub(r'% BEGIN GENERATED TABLE.*?% END GENERATED TABLE',lambda m: '% BEGIN GENERATED TABLE\n'+'\n'.join(table)+'\n% END GENERATED TABLE',text,flags=re.S)
    main.write_text(text)

r=np.array([float(x['forgetting']) for x in groups['random']])
e=np.array([float(x['forgetting']) for x in groups['egogate']])
d=r-e
ci=t.interval(.95,df=len(d)-1,loc=d.mean(),scale=d.std(ddof=1)/np.sqrt(len(d)))
report={'n':len(d),'mean_difference':float(d.mean()),'relative_reduction':float(d.mean()/r.mean()),
        'paired_t':float(ttest_rel(r,e).statistic),'p':float(ttest_rel(r,e).pvalue),
        'dz':float(d.mean()/d.std(ddof=1)),'ci95':list(ci),
        'input_sha256':hashlib.sha256((ROOT/'results/digits/per_seed.csv').read_bytes()).hexdigest(),
        'evidence':'Existing repository results; this analysis does not rerun training.'}
(DEST/'statistics_audit.json').write_text(json.dumps(report,indent=2))
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.05,2.55),layout='constrained')
keys=['random','stratified_random','doubt','curiosity','egogate','egogate_diverse','full']
for i,k in enumerate(keys):
    v=np.array([float(x['forgetting'])*100 for x in groups[k]])
    axs[0].barh(i,v.mean(),xerr=t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)),color='#315b79',capsize=2)
axs[0].set_yticks(range(len(keys)),[names[k] for k in keys])
axs[0].invert_yaxis(); axs[0].set_xlabel('Forgetting (percentage points)')
axs[0].set_title('(a) Mean and 95% confidence interval')
axs[1].scatter(r*100,e*100,s=18,color='#315b79')
axs[1].plot([0,30],[0,30],'--',color='gray',linewidth=.7)
axs[1].set(xlim=(0,30),ylim=(0,30),xlabel='Random forgetting (points)',ylabel='EgoGate forgetting (points)')
axs[1].set_title('(b) Seed-paired forgetting')
fig.savefig(DEST/'retention.pdf'); fig.savefig(DEST/'retention.png',dpi=180)
print(json.dumps(report,indent=2))
