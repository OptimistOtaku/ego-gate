"""Validate completed stream outputs and emit summary plus declared paired tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t, ttest_rel

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
a=p.parse_args()
rows=json.loads((a.directory/'raw.json').read_text())
manifest=json.loads((a.directory/'manifest.json').read_text())
args=manifest['arguments']
expected=len(args['datasets'])*len(args['noise'])*args['seeds']*len(manifest['conditions'])
keys=[(r['dataset'],r['noise'],r['seed'],r['policy'],r['schedule'],r['admission']) for r in rows]
assert len(keys)==len(set(keys))==expected, 'Incomplete or duplicate results; do not report as complete.'
for name in args['datasets']:
    for noise in args['noise']:
        for seed in range(args['seeds']):
            group=[r for r in rows if (r['dataset'],r['noise'],r['seed'])==(name,noise,seed)]
            assert len({r['updates'] for r in group})==1
            assert len({r['training_exposures'] for r in group})==1
            assert all(r['buffer_examples']==r['memory'] for r in group)
            assert all(len(r['trace'])==5 for r in group)

def write(name, data):
    with (a.directory/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]))
        writer.writeheader();writer.writerows(data)

summary=[]
comparisons=[]
for name in args['datasets']:
    for noise in args['noise']:
        for policy,schedule,admission in manifest['conditions']:
            g=[r for r in rows if (r['dataset'],r['noise'],r['policy'],r['schedule'],r['admission'])==(name,noise,policy,schedule,admission)]
            d={'dataset':name,'noise':noise,'policy':policy,'schedule':schedule,'admission':admission,'n':len(g)}
            for metric in ['final_accuracy','forgetting','mean_seen_accuracy','buffer_corruption','admission_fraction','wall_seconds','score_seconds','peak_pending_examples','updates','training_exposures']:
                v=np.array([r[metric] for r in g]);d[metric+'_mean']=float(v.mean());d[metric+'_sd']=float(v.std(ddof=1))
            summary.append(d)
        for label,b,c in [
            ('admission',('egogate_audit','immediate',False),('egogate_audit','immediate',True)),
            ('schedule',('egogate_audit','immediate',True),('egogate_audit','periodic',True))]:
            def values(condition):
                g=sorted([r for r in rows if (r['dataset'],r['noise'])==(name,noise) and (r['policy'],r['schedule'],r['admission'])==condition],key=lambda r:r['seed'])
                return np.array([r['final_accuracy'] for r in g])
            bv,cv=values(b),values(c);diff=cv-bv
            sd=diff.std(ddof=1);se=sd/np.sqrt(len(diff));mean=diff.mean()
            if sd==0:
                stat,pv=(0.,1.) if mean==0 else (float(np.sign(mean)*np.inf),0.)
            else:
                test=ttest_rel(cv,bv);stat,pv=float(test.statistic),float(test.pvalue)
            ci=(mean-t.ppf(.975,len(diff)-1)*se,mean+t.ppf(.975,len(diff)-1)*se)
            comparisons.append({'dataset':name,'noise':noise,'comparison':label,'n':len(diff),'accuracy_improvement':float(mean),'ci_low':float(ci[0]),'ci_high':float(ci[1]),'paired_t':stat,'p':pv})
order=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['p'])
last=0.
for rank,i in enumerate(order):
    last=max(last,(len(order)-rank)*comparisons[i]['p']);comparisons[i]['holm_p']=min(last,1.)
write('summary.csv',summary);write('primary_comparisons.csv',comparisons)
print('Validated',len(rows),'runs; wrote descriptive and primary paired analyses.')
