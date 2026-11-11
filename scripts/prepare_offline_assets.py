"""Produce manuscript assets only from a completed, audited benchmark."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS={'random':'Random','stratified_random':'Balanced random','doubt':'Doubt',
        'curiosity':'Curiosity','egogate':'EgoGate','embedding_kcenter':'Embedding k-center',
        'egogate_diverse':'EgoGate-diverse','loss':'High loss','herding':'Herding adaptation',
        'none':'No replay','full':'Full-pool replay'}
DATASETS={'digits':'Digits','fashion':'Fashion-MNIST','cifar10':'CIFAR-10'}


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def make_assets(source, dest):
    audit=json.loads((source/'audit.json').read_text())
    assert audit['complete'] and audit['rows']==870 and audit['primary_tests']==27
    rows=read_csv(source/'per_run.csv')
    summary=read_csv(source/'summary.csv')
    comparisons=read_csv(source/'primary_comparisons.csv')
    dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(7.0,2.65),layout='constrained')
    colors={'random':'#777777','doubt':'#228833','herding':'#4477AA','egogate':'#CC3311'}
    styles={'random':('o','--'),'doubt':('s','-.'),'herding':('^',':'),'egogate':('D','-')}
    for ax,(dataset,name) in zip(axes,DATASETS.items()):
        for method in colors:
            group=sorted([r for r in summary if r['dataset']==dataset and r['method']==method],key=lambda r:int(r['budget']))
            x=[int(r['budget']) for r in group]
            mean=np.array([float(r['final_accuracy_mean'])*100 for r in group])
            std=np.array([float(r['final_accuracy_std'])*100 for r in group])
            marker,style=styles[method]
            ax.plot(x,mean,marker=marker,linestyle=style,label=LABELS[method],color=colors[method],markersize=3)
            ax.fill_between(x,mean-std,mean+std,color=colors[method],alpha=.1)
        ax.set(title=name,xlabel='Stored examples',xticks=x)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Final mean accuracy (%)')
    axes[1].legend(fontsize=7,loc='best')
    fig.savefig(dest/'offline_benchmark.pdf')
    fig.savefig(dest/'offline_benchmark.png',dpi=180)
    plt.close(fig)
    table=[r'\begin{table*}[t]',r'\centering\small',
           r'\caption{Expanded offline benchmark: final accuracy (\%) $\pm$ sample standard deviation over ten paired seeds. Central budgets are 40 for Digits and 200 for Fashion-MNIST and CIFAR-10. Anchors have different storage budgets.}',
           r'\label{tab:offlineexpanded}',r'\begin{tabular}{lrrr}\toprule',
           r'Selector & Digits & Fashion-MNIST & CIFAR-10\\\midrule']
    for method,label in LABELS.items():
        cells=[]
        for dataset in DATASETS:
            budget=40 if dataset=='digits' else 200
            group=[float(r['final_accuracy'])*100 for r in rows if r['dataset']==dataset and r['method']==method and (method in ('none','full') or int(r['budget'])==budget)]
            assert len(group)==10
            cells.append(f'{np.mean(group):.2f} $\\pm$ {np.std(group,ddof=1):.2f}')
        table.append(label+' & '+' & '.join(cells)+r' \\')
    table.extend([r'\bottomrule\end{tabular}',r'\end{table*}'])
    (dest/'offline_table.tex').write_text('\n'.join(table)+'\n')
    lines=[r'\begin{table}[t]\centering\small',
           r'\caption{EgoGate minus random storage in final accuracy (percentage points), with pointwise seed-paired 95\% intervals and Holm-adjusted $p$ values from the 27-test primary family.}',
           r'\label{tab:offlinecontrasts}',r'\begin{tabular}{lrrr}\toprule',
           r'Dataset / $K$ & Difference & 95\% CI & $p_{\mathrm{Holm}}$\\\midrule']
    for r in comparisons:
        if r['comparator']!='random':continue
        name={'digits':'Digits','fashion':'Fashion','cifar10':'CIFAR-10'}[r['dataset']]
        delta,lo,hi,p=[float(r[k]) for k in ('accuracy_delta_pp','ci_low','ci_high','holm_p_value')]
        ptext=f'{p:.3f}' if p>=.001 else '$<0.001$'
        lines.append(f'{name} / {r["budget"]} & {delta:+.2f} & [{lo:+.2f}, {hi:+.2f}] & {ptext}'+r' \\')
    lines.extend([r'\bottomrule\end{tabular}',r'\end{table}'])
    (dest/'offline_contrasts.tex').write_text('\n'.join(lines)+'\n')
    report=[]
    for comparator in ('random','doubt','herding'):
        subset=[r for r in comparisons if r['comparator']==comparator]
        wins=[r for r in subset if float(r['holm_p_value'])<.05 and float(r['accuracy_delta_pp'])>0]
        losses=[r for r in subset if float(r['holm_p_value'])<.05 and float(r['accuracy_delta_pp'])<0]
        report.append(dict(comparator=comparator,significant_wins=len(wins),significant_losses=len(losses),not_rejected=len(subset)-len(wins)-len(losses)))
    (dest/'offline_findings.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--dest',type=Path,default=Path('paper/ieee'))
    a=parser.parse_args()
    make_assets(a.source,a.dest)
