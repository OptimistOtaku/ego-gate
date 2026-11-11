from pathlib import Path
import csv,json,hashlib,re
root=Path.cwd();p=root/'paper/ieee/main.tex';s=p.read_text()
if 'Independent Reproduction}' in s:
 raise SystemExit('Kaggle evidence already integrated; refusing to duplicate sections.')
m=json.loads((root/'results/kaggle_ieee/ego_gate/COMPLETE.json').read_text())
for f,h in m['source_hashes'].items():
 assert hashlib.sha256((root/f).read_text().encode()).hexdigest()==h,f
s=s.replace('Ego Gate: Selective Memory Admission and Controlled Consolidation for Continual Learning','Ego Gate: Evaluating Selective Replay, Online Admission, and Periodic Consolidation')
a=s.index('\\begin{abstract}');b=s.index('\\end{abstract}',a)
s=s[:a]+r'''\begin{abstract}
Continual learners must acquire new knowledge while retaining earlier competence under limited memory and computation. We present Ego Gate, a framework separating observation valuation, memory admission, and consolidation, and evaluate two distinct implementations. An archived Split-Digits study finds that entropy-plus-gradient selection reduces forgetting by 39.4\% relative to random storage across 20 paired seeds. An independent Kaggle reproduction confirms the direction, with a 40.1\% reduction. A subsequent 200-run study evaluates single-pass streams on Digits and Fashion-MNIST, with five seeds, clean and corrupted labels, and ten storage/admission/scheduling conditions. The tested streaming extension does not inherit the offline benefit: on clean Fashion-MNIST, reservoir replay achieves 72.44\% final accuracy versus 29.66\% for the combined score. None of eight specified admission or scheduling contrasts is significant after Holm correction. We distinguish Bayesian information gain from supervised gradient magnitude, identify an intrinsic retention limit of unconditional exploratory replacement, and report resource-accounting and evaluation limitations. The evidence supports a small offline storage result while exposing failures of the particular online extension. These findings motivate coverage-aware control and stronger validation, rather than claims that the architecture already enables general intelligence.
'''+s[b:]
s=s.replace('but the present experiment establishes a narrower fact about replay-buffer construction.', 'and our experiments test both replay-buffer construction and a specific streaming extension.')
s=s.replace('(4) an executable extension protocol that makes the architectural claims falsifiable.', '(4) a completed streaming evaluation that reveals where a particular extension fails.')
s=s.replace('A separate private Kaggle notebook packages an independent reproduction and the streaming extension.', 'A private Kaggle notebook additionally completed an independent reproduction and the streaming extension, reported below.')
s=s.replace('\\section{Executable Evaluation Protocol for the Gap}', '\\section{Kaggle Streaming Evaluation}')
s=s.replace('The accompanying notebook defines an additional study before execution. This section describes that protocol; it contains no claimed outcomes.', 'The accompanying notebook fixed the following protocol before execution. All 200 condition-seed runs completed on Kaggle, and downloaded raw trajectories passed checks for unique conditions, matched update/exposure budgets, buffer size, and five evaluation checkpoints.')
s=s.replace('The default protocol uses five seeds', 'The executed protocol uses five seeds')
s=s.replace('Paired inference should correct across this eight-comparison family and report effect intervals.', 'Paired inference corrects across this eight-comparison family and reports effect intervals.')
s=s.replace('A private Kaggle run must finish and its downloaded artifacts must be audited before any new result is inserted into the manuscript.', 'The completion marker and normalized source hashes match the supplied code. The run used CPU execution, Python 3.13.18, PyTorch 2.11.0+cpu, NumPy 2.1.3, SciPy 1.16.3, and scikit-learn 1.6.1. CPU model and torchvision version were not recorded, limiting exact environmental reconstruction.')
pos=s.index('\\section{From Learning Control to Agent Memory}')
rows=list(csv.DictReader(open(root/'results/kaggle_ieee/ego_gate/stream_results/summary.csv')))
conditions=[('reservoir','immediate',False,'Reservoir'),('balanced','immediate',False,'Balanced random'),('doubt','immediate',False,'Doubt'),('curiosity','immediate',False,'Curiosity'),('egogate','immediate',False,'EgoGate'),('egogate_audit','immediate',False,'EgoGate + bypass'),('reservoir','periodic',False,'Reservoir periodic'),('egogate_audit','periodic',False,'Bypass periodic'),('egogate_audit','immediate',True,'Bypass + admission'),('egogate_audit','periodic',True,'Admission periodic')]
table=[]
for policy,schedule,admission,label in conditions:
 vals=[]
 for dataset,noise in [('digits',0),('digits',.2),('fashion',0),('fashion',.2)]:
  r=next(r for r in rows if r['dataset']==dataset and float(r['noise'])==noise and r['policy']==policy and r['schedule']==schedule and r['admission']==str(admission))
  vals.append(f"{100*float(r['final_accuracy_mean']):.2f} $\\pm$ {100*float(r['final_accuracy_sd']):.2f}")
 table.append(label+' & '+' & '.join(vals)+r' \\')
addition=r'''\subsection{Independent Reproduction}
The Kaggle reproduction contains 180 condition-seed runs. Mean forgetting is 19.28 points for random storage and 11.55 for EgoGate: a 40.1\% relative reduction, paired $t(19)=6.78$, $d_z=1.52$, and Holm-adjusted $p=1.07\times10^{-5}$. Doubt yields 10.39 points; its comparison with EgoGate is not significant (Holm $p=0.365$). Small numerical differences from the archive are retained rather than overwritten. This confirms the qualitative offline finding, not bitwise reproducibility or cross-dataset generality.

\begin{table*}[t]
\caption{Completed streaming study: final mean accuracy (\%) $\pm$ sample standard deviation across five seeds. Noise is the training-label corruption rate. All ten conditions are reported; selectors use the streaming rules, not offline top-$K$ selection.}
\label{tab:stream}\centering\small
\begin{tabular}{lrrrr}\toprule
Condition & Digits, 0\% & Digits, 20\% & Fashion, 0\% & Fashion, 20\%\\\midrule
'''+ '\n'.join(table)+r'''
\bottomrule\end{tabular}
\end{table*}

\subsection{Streaming Results and Inference}
Table~\ref{tab:stream} shows that the offline advantage does not transfer to this implementation. On clean Fashion-MNIST, reservoir replay reaches 72.44\% versus 29.66\% for EgoGate. With 20\% label corruption, the corresponding means are 69.09\% and 31.57\%. These are descriptive secondary comparisons. On clean Digits, balanced random storage has the highest immediate-policy mean (56.12\%), while on corrupted Digits Doubt reaches 39.87\% versus reservoir's 32.29\%. No selector dominates across these settings.

For the specified primary comparisons, admission changes final accuracy by -4.09 and -0.62 points on clean/noisy Digits, and +0.47 and -0.40 on clean/noisy Fashion-MNIST. Periodic rather than immediate admission changes accuracy by -4.85, -10.01, -0.72, and +0.21 points, respectively. All eight 95\% paired intervals include zero; Holm-adjusted $p$ values are at least 0.912. Five seeds provide limited precision, and these results establish neither equivalence nor a general impossibility of useful admission or scheduling.

Scoring imposes substantial overhead. Clean Fashion-MNIST mean timed execution is 4.65 s for reservoir, 22.61 s for Doubt, and 37.97 s for EgoGate. These timings include stream processing and evaluation but exclude dataset loading and initialization; they are not deployment latency or universal hardware benchmarks. The gradient score provides no measured efficiency advantage here.

The signed forgetting diagnostic subtracts final accuracy from the maximum earlier checkpoint for each of the first four phases. It can be negative when later replay improves an initially undertrained task. We emphasize final average accuracy rather than treating negative forgetting as exceptional retention. The final checkpoint is measured after flushing the queue; intermediate checkpoints can contain pending observations and therefore reflect adaptation delay.

\subsection{Interpretation and Identified Failure Mechanism}
The streaming study changes several features of the archived setting: five phases instead of two, single-pass learning instead of repeated epochs, a different architecture and learning rate, bounded score transforms, and stochastic replacement instead of global top-$K$ selection. Consequently, its failures cannot be attributed to the abstract Ego Gate concept or to the original offline scoring rule alone.

The exploratory bypass has a specific retention weakness. Once the buffer is full, an existing item is removed by an unconditional bypass with probability $\epsilon/M$ per arrival. Its probability of surviving $n$ arrivals is therefore at most $(1-\epsilon/M)^n$, since score-based replacement may remove it sooner. With $M=200$, $\epsilon=0.1$, and 12,000 subsequent arrivals (one Fashion-MNIST phase), this bound is about 0.0025. Thus a constant bypass of this form promotes access for new evidence at the expense of old-memory survival. It is not equivalent to a protected reservoir for exploration. This observation explains a structural risk, but the study does not record item lifetimes or class occupancy sufficiently to establish the complete causal account of the accuracy losses.

Other explanations, including concentration on difficult/noisy examples and changed gradient scale, remain hypotheses. Periodic scheduling also changes within-block update order and the memory available at replay time; it is not a pure delay-only intervention. New observations can appear both in the current minibatch and replay pool. These mechanisms must be disentangled before claiming an intrinsic benefit or harm from offline consolidation. A subsequent study should protect historical coverage, use a separate reservoir for exploratory admission, and compare timing with matched order and available evidence. Those modifications are not evaluated here.

'''
s=s[:pos]+addition+s[pos:]
a=s.index('\\section{Conclusion}');b=s.index('\\section*{Reproducibility}',a)
s=s[:a]+r'''\section{Conclusion}
Ego Gate frames continual learning as decisions about which experiences to retain, verify, learn, and consolidate. Its small offline replay result reproduces independently on Kaggle. The completed streaming study, however, shows that a direct heuristic extension can substantially underperform reservoir replay, and finds no statistically supported gain from the specified admission and periodic-consolidation changes. An unconditional exploratory replacement rule also imposes a severe retention bound.

The contribution is therefore a framework with both positive component evidence and a documented failure to generalize to the tested online design. Coverage preservation, reliable utility estimation, and controlled scheduling experiments are prerequisites for stronger claims. The present evidence does not validate an adaptive meta-controller, a commercial memory system, or a route to AGI.

'''+s[b:]
s=s.replace('The proposed streaming extension deliberately sacrifices some sophistication for transparent implementation;', 'The evaluated streaming extension deliberately sacrifices some sophistication for transparent implementation;')
p.write_text(s)
print('Integrated completed evidence and checked all source hashes.')
