"""Validate the entire fixed benchmark grid before producing inferential output."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

METHODS=('random','stratified_random','doubt','curiosity','egogate','embedding_kcenter','egogate_diverse','loss','herding')
BUDGETS={'digits':[20,40,100],'fashion':[50,200,500],'cifar10':[50,200,500]}


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output):
    rows=[]
    manifests=list(root.glob('**/manifest.json'))
    if not manifests:
        raise ValueError('No execution manifests')
    hashes=set()
    for file in manifests:
        m=json.loads(file.read_text())
        assert m['gpu'] and m['device'].startswith('cuda') and not m['smoke']
        assert (file.parent/'COMPLETE.json').exists(), f'Incomplete shard: {file}'
        hashes.add(m['source_sha256'])
    assert len(hashes)==1, 'Mixed source versions'
    for dataset in BUDGETS:
        for file in root.glob(f'**/{dataset}_*.json'):
            if '.partial.' in file.name:
                raise ValueError(f'Incomplete evidence: {file}')
            doc=json.loads(file.read_text())
            assert doc['complete'] and doc['dataset']==dataset
            rows.extend(doc['rows'])
    keys=[(r['dataset'],r['seed'],r['budget'],r['method']) for r in rows]
    assert len(keys)==len(set(keys)), 'Duplicate evidence'
    expected={(d,s,b,m) for d,bs in BUDGETS.items() for s in range(100,110) for b in bs for m in METHODS}
    bounded=[r for r in rows if r['method'] in METHODS]
    assert {k for k in keys if k[3] in METHODS}==expected, 'Incomplete bounded grid'
    anchors=[r for r in rows if r['method'] not in METHODS]
    assert len(anchors)==60 and {(r['dataset'],r['seed'],r['method']) for r in anchors}=={
        (d,s,m) for d in BUDGETS for s in range(100,110) for m in ('none','full')}
    for d in BUDGETS:
        for s in range(100,110):
            group=[r for r in bounded if r['dataset']==d and r['seed']==s]
            assert len({(r['updates'],r['exposures'],r['replay_exposures']) for r in group})==1
            assert len({r['task_a_pre'] for r in group})==1
    for r in rows:
        ix=r['selected_indices']
        assert len(ix)==r['budget']==len(set(ix))
        assert sum(r['class_counts'])==r['budget']
        assert hashlib.sha256(np.asarray(ix,dtype='<i8').tobytes()).hexdigest()==r['selection_sha256']
        assert all(np.isfinite(r[k]) for k in ('final_accuracy','forgetting','selection_seconds','training_seconds'))
        assert abs(r['final_accuracy']-(r['task_a_post']+r['task_b_post'])/2)<1e-10
    output.mkdir(parents=True,exist_ok=True)
    flat=[{k:v for k,v in r.items() if k not in ('selected_indices','class_counts')} for r in rows]
    write_csv(output/'per_run.csv',flat)
    summary=[]
    for d,bs in BUDGETS.items():
        for b in bs:
            for m in METHODS:
                group=[r for r in bounded if (r['dataset'],r['budget'],r['method'])==(d,b,m)]
                row=dict(dataset=d,budget=b,method=m,n=len(group))
                for metric in ('final_accuracy','forgetting','task_a_pre','task_a_post','task_b_post','selection_seconds','training_seconds','coverage','memory_bytes','cuda_peak_bytes'):
                    values=np.asarray([r[metric] for r in group])
                    row[metric+'_mean']=float(values.mean())
                    row[metric+'_std']=float(values.std(ddof=1))
                summary.append(row)
    write_csv(output/'summary.csv',summary)
    contrasts=[]
    for d,bs in BUDGETS.items():
        for b in bs:
            for comparator in ('random','doubt','herding'):
                def values(m):
                    return np.asarray([next(r['final_accuracy'] for r in bounded if (r['dataset'],r['budget'],r['method'],r['seed'])==(d,b,m,s)) for s in range(100,110)])
                diff=100*(values('egogate')-values(comparator))
                se=stats.sem(diff)
                ci=stats.t.ppf(.975,9)*se
                p=float(stats.ttest_1samp(diff,0).pvalue) if se else (1. if diff.mean()==0 else 0.)
                contrasts.append(dict(dataset=d,budget=b,comparator=comparator,n=10,
                                      accuracy_delta_pp=float(diff.mean()),ci_low=float(diff.mean()-ci),ci_high=float(diff.mean()+ci),p_value=p))
    running=0.
    for rank,i in enumerate(sorted(range(len(contrasts)),key=lambda i:contrasts[i]['p_value'])):
        running=max(running,(len(contrasts)-rank)*contrasts[i]['p_value'])
        contrasts[i]['holm_p_value']=min(running,1.)
    write_csv(output/'primary_comparisons.csv',contrasts)
    audit=dict(complete=True,rows=len(rows),bounded_rows=len(bounded),anchor_rows=len(anchors),
               primary_tests=len(contrasts),source_sha256=list(hashes)[0],gpu_manifests=len(manifests))
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    analyze(a.root,a.output)
