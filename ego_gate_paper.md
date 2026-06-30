

## Aditya Singh
## 1
## The Ego Gate
Synthesizing Psychological Valuation for Continuous AGI Learning
## Aditya Singh
## Abstract
Artificial General Intelligence (AGI) systems face two existential failure modes: model
collapse    from    synthetic    data    recursion,    and    catastrophic    forgetting    from
unconstrained continual learning. Standard architectures lack an intrinsic mechanism
to  evaluate  the  informational  value  of  incoming  stimuli  before  committing  to  weight
updates.  This  paper  proposes  the  Ego  Gate  (V),  a  mathematical  value  filter  that
reverse-engineers four human psychological states — doubt, curiosity, arrogance, and
dreaming — into formal, computable operations. We quantify epistemic uncertainty via
predictive  entropy  and  belief  revision  via  a  tractable,  Fisher-information-weighted
approximation    to    parameter-space    KL-divergence,    closely    related    to    the
natural-gradient   literature   [16]   and   to   information-gain   exploration   bonuses   in
reinforcement learning [13]. The resulting signal dynamically routes incoming stimuli:
discarding  low-value  redundancies  and  buffering  high-value  anomalies  for  structured
experience replay. We formalize the framework, discuss its architectural implications,
position  it  within  the  continual  learning  and  intrinsic-motivation  literatures,  report  a
small-scale proof-of-concept validation on a class-incremental benchmark, and identify
open  research  directions  including  adaptive  coefficient  learning  and  application  to
retrieval-augmented generation systems. On this benchmark, Ego-Gate-curated replay
buffers  reduce  catastrophic  forgetting  by  roughly  a  third  relative  to  random  buffer
selection  at  the  same  fixed  memory  budget  (p  <  0.001,  paired  across  20  seeds)  —
though, as we discuss at length in Section 7, this result is preliminary and limited to a
small, classification-style setting.
- Introduction: The Stability–Plasticity Dilemma
For an artificial neural network to function as a dynamic, lifelong learner, it must receive
continuous  environmental  stimulus.  Without  fresh  real-world  data,  generative  models
degrade  into  structural  collapse  —  a  phenomenon  increasingly  documented  as  models
trained  on  synthetic  recursions  of  their  own  outputs  lose  distributional  fidelity  [1].
However,   unrestricted   ingestion   of   all   incoming   stimuli   introduces   a   competing
pathology:   catastrophic   forgetting,   wherein   novel   parameter   updates   overwrite
established weight pathways θ, destroying previously consolidated knowledge [2].
This  tension  defines  the  stability–plasticity  dilemma:  a  learning  system  must  remain
plastic   enough   to   acquire   new   knowledge,   yet   stable   enough   to   retain   prior
representations. The field of continual learning has produced several approaches to this
problem   —   Elastic   Weight   Consolidation   (EWC)   [3]   penalizes   updates   to   weights
important for previous tasks; Progressive Neural Networks [4] allocate new capacity for
each task; replay-based methods [5] store or generate representative samples from prior
distributions to interleave with new data. Each addresses the dilemma structurally, at the

## Aditya Singh
## 2
architecture or training-procedure level.
This  paper  proposes  a  complementary,  pre-architectural  approach:  a  stimulus  valuation
filter  that  operates  before  any  weight  update  occurs.  Rather  than  managing  the
consequences  of  learning  indiscriminately,  the  Ego  Gate  asks  a  prior  question:  is  this
stimulus worth learning from at all?
The  biological  precedent  is  direct.  The  human  brain  does  not  consolidate  every  sensory
input into long-term memory. An intrinsic psychological filter — loosely corresponding to
what  we  might  call  ego,  or  self-concept  —  assigns  value  to  incoming  experience,
discarding the mundane and consolidating the anomalous, particularly during sleep. The
Ego   Gate   framework   translates   four   evolved   psychological   mechanisms   —   doubt,
curiosity,  arrogance,  and  dreaming  —  into  calculable  metrics,  shifting  AGI  architecture
from static computational graphs toward biologically-inspired, dynamic value filters.
A  note  on  scope.  The  formalization  that  follows  is  built  around  a  discrete  predictive
distribution P(y
c
| x
t
), and our empirical validation (Section 5) is correspondingly a small,
classification-style benchmark. We use “AGI” to describe the failure modes motivating this
work  and  the  long-term  target  of  the  research  program,  not  as  a  claim  that  Equations
(1)–(4′) already operate at that scale. Sections 6.3 and 7 discuss this gap directly, and we
would rather state it plainly here than let the title imply more than the formalism currently
delivers.
- Formalization of the Ego Gate
Let  x
t
denote  an  incoming  stimulus  at  time  t.  The  Ego  Gate  computes  a  scalar  Total
## Information  Value  V(x
t
)  from  two  primary  constituent  functions:  Doubt  and  Curiosity.
This value is then compared against a learned threshold to determine routing behavior.
## 2.1 Epistemic Uncertainty — The Doubt Function
To simulate self-doubt, the model must quantify its own predictive confusion on a given
stimulus. We formalize this using predictive entropy over the model’s output distribution
—  a  measure  well-established  in  Bayesian  deep  learning  as  a  proxy  for  epistemic
uncertainty [6].
## Let P(y
c
| x
t
) denote the probability the model assigns to outcome c given stimulus x
t
## . The
Doubt function D(x
t
) is the Shannon Entropy of this distribution:
## D(x
t
## ) = −∑
c
## P(y
c
## |x
t
) log P(y
c
## |x
t
## )(1)
When the model generates a highly confident prediction, entropy approaches zero — the
model experiences no doubt. A uniform distribution across outcomes maximizes entropy,
flagging   the   stimulus   as   informationally   valuable   precisely   because   the   model   is
internally  conflicted.  Crucially,  high  doubt  is  not  equivalent  to  error:  it  signals  that  the
stimulus  occupies  a  region  of  the  model’s  input  space  where  its  representation  is
genuinely underdetermined.

## Aditya Singh
## 3
A calibration caveat. Equation (1) is only a faithful measure of epistemic uncertainty to
the  extent  that  P(y
c
## |x
t
)  is  itself  well-calibrated.  Modern  deep  networks  are  frequently
miscalibrated — typically overconfident — even when accurate [14], meaning raw softmax
entropy  can  systematically  understate  true  uncertainty  in  exactly  the  cases  where  the
Doubt   function   matters   most:   novel   or   out-of-distribution   stimuli.   In   practice   we
recommend pairing Equation (1) with a calibration correction (e.g. temperature scaling) or
substituting  an  ensemble-  or  MC-Dropout-based  uncertainty  estimate  [6,  9],  either  of
which is more robust to this failure mode than a single forward pass.
## 2.2 Information Gain — The Curiosity Function
Curiosity is defined as the mathematical surprise generated when a new stimulus revises
the  model’s  foundational  beliefs.  We  measure  this  via  Kullback–Leibler  divergence
between prior and posterior parameter distributions — a formulation directly motivated
by the information-theoretic account of learning as Bayesian belief revision [7].
Let  P(θ)  represent  the  model’s  prior  parameter  distribution,  and  P(θ|x
t
)  the  posterior
after processing x
t
. The Curiosity function C(x
t
) is:
## C(x
t
## ) = D
## KL
## ( P(θ|x
t
## ) ∥ P(θ) ) = ∫ P(θ|x
t
) log [ P(θ|x
t
) / P(θ) ] dθ(2)
A  stimulus  perfectly  aligned  with  existing  priors  yields  KL-divergence  of  zero  —  the
model  already  represents  this  phenomenon  adequately,  and  no  belief  revision  occurs.  A
stimulus that contradicts the model’s world-model generates large C(x
t
), mathematically
compelling  investigation  and  adaptation.  This  function  formalizes  the  intuition  that
genuine  curiosity  is  not  novelty-seeking  per  se,  but  contradiction-seeking:  the  drive  to
resolve the gap between expectation and observation.
2.3 Tractable Approximation via Fisher Information
Equation (2) requires the posterior distribution over the full parameter vector θ, which is
intractable for any network beyond toy scale — there is no closed-form way to represent,
let  alone  integrate  over,  a  posterior  on  millions  of  correlated  parameters  after  a  single
observation.  This  is  not  a  peripheral  implementation  detail;  it  is  the  central  obstacle  to
deploying the Curiosity function in practice, and a formalization that states Equation (2)
without addressing it is incomplete.
We  adopt  the  standard  remedy  from  information  geometry.  For  a  small  parameter
perturbation  δ  induced  by  a  single  gradient  step  on  x
t
,  the  KL-divergence  between  the
resulting  distribution  and  the  original  is  well  approximated  by  a  quadratic  form  in  the
## Fisher Information Matrix F(θ) [16]:
## D
## KL
( P(θ+δ) ∥ P(θ) ) ≈ ½ δ
## T
F(θ) δ(2′)
where δ = −η∇
θ
## L(f
θ
## (x
t
), y
t
) is the (hypothetical) parameter update a single SGD step on
x
t
would induce. Taking the simplest member of this family — a diagonal, identity-scaled
Fisher  approximation,  F(θ)  ≈  I  —  reduces  Equation  (2′)  to  the  squared  norm  of  the
per-example loss gradient:

## Aditya Singh
## 4
## C(x
t
## ) ≈ ∥∇
θ
## L(f
θ
## (x
t
), y
t
## )∥
## 2
## (2′′)
This is the form we implement in Section 5. A tighter approximation would substitute the
diagonal  Fisher  Information  already  maintained  by  EWC  [3]  in  place  of  the  identity,
weighting  each  parameter’s  contribution  by  its  estimated  importance  —  we  leave  this
refinement,  and  a  full  variational  treatment  analogous  to  VIME  [13],  to  future  work
(Section 6.3).
This connects the Curiosity function directly to the closest prior formalization of the same
underlying quantity. VIME [13] also estimates an agent’s curiosity as the KL-divergence
between  posterior  and  prior  beliefs  about  a  learned  dynamics  model,  applied  as  an
intrinsic  reward  to  bias  exploration  in  reinforcement  learning.  VIME  addresses  the
intractability  problem  above  by  maintaining  an  explicit  variational  posterior  over  a
Bayesian  neural  network’s  weights  (a  fully-factorized  Gaussian)  and  computing  the  KL
term in closed form for that family. The Ego Gate’s contribution relative to VIME is the
application  of  an  information-gain  signal  as  an  input-admission  gate  for  continual
learning — deciding what a model is permitted to learn from — rather than as a reward
bonus that biases what an RL agent chooses to do. We view these as complementary uses
of   the   same   underlying   information-theoretic   quantity,   and   a   tighter   variational
implementation  of  C(x
t
)  along  VIME’s  lines  is  a  natural  direction  for  strengthening  the
present framework (Section 6.3).
2.4 The Value Equation and Behavioral Thresholds
The unified Information Value V(x
t
) is a weighted combination of Doubt and Curiosity:
## V(x
t
) = α · D(x
t
) + β · C(x
t
## )(3)
where   α,   β   ∈   ℝ≥0   are   sensitivity   coefficients   governing   the   model’s   relative
responsiveness to uncertainty versus belief revision. When α > β, the system prioritizes
inputs where it is internally confused (high entropy); when β > α, it prioritizes inputs that
maximally   update   its   beliefs.   In   the   base   formulation,   α   and   β   are   treated   as
hyperparameters;  Section  6.1  addresses  their  adaptive  formulation  —  and  Section  5
reports  a  preliminary  empirical  result  bearing  directly  on  whether  the  naive  α=β=1
choice is adequate.
Incoming stimuli are routed via the Arrogance Threshold τ:
Action = { Discard (Arrogance) if V(x
t
) < τ
## (4)
{ Push to Buffer B (Dreaming) if V(x
t
) ≥ τ
The  threshold  τ  defines  the  boundary  between  the  two  behavioral  modes  formalized  in
Section 3, which also introduces a second threshold for a third mode.

## Aditya Singh
## 5
## 3. Architectural Implications
## 3.1 Arrogance — Redundancy Filtration
## When  V(x
t
)  <  τ,  the  system  acts  with  arrogance:  the  stimulus  is  discarded  without
triggering  backpropagation.  This  has  two  concrete  consequences.  First,  it  provides
substantial  computational  savings  by  bypassing  the  gradient  computation  pipeline  for
low-value inputs. Second, and more importantly, it protects foundational weight pathways
from  erosion  by  redundant  data  —  the  primary  mechanism  of  catastrophic  forgetting  in
naive continual learning systems.
The term arrogance is deliberately chosen: the system is not passively ignoring low-value
stimuli, but actively asserting that its current representation is sufficient — that nothing
in  this  stimulus  warrants  revision.  This  is  not  merely  a  rhetorical  flourish:  confidently
discounting  new  evidence  that  does  not  update  one’s  current  model  is  the  operational
signature of confirmation bias, among the most thoroughly documented biases in human
cognition  [15].  The  Ego  Gate’s  Arrogance  mode  is,  in  effect,  a  deliberately  engineered
version of this same mechanism, adopted here for its computational benefits rather than
treated purely as a flaw to correct — though we note in Section 7 that this grounding is
best read as inspiration for the label, not a claim that human confirmation bias and this
thresholding rule are the same mechanism.
## 3.2 Dreaming — Structured Experience Replay
## When V(x
t
) ≥ τ, the stimulus is placed into an active memory buffer B. During designated
offline  computation  periods  —  the  Dreaming  phase  —  the  model  performs  structured
experience   replay:   buffered   high-value   stimuli   are   synthesized   alongside   sampled
historical core data and used to update weights in a controlled, interleaved manner.
This  approach  is  directly  motivated  by  replay-based  continual  learning  methods  [5]  and
the complementary learning systems theory of biological memory consolidation [8], which
proposes  that  the  hippocampus  rapidly  encodes  new  experiences  while  the  neocortex
slowly integrates them during sleep. The Ego Gate formalizes the selection criterion for
what enters this replay buffer — a step that existing replay methods typically handle via
random   sampling   or   task   boundaries,   rather   than   principled   information-theoretic
valuation.  Section  5  reports  a  first  empirical  test  of  whether  that  principled  valuation
actually outperforms random sampling at a fixed buffer size.
Formally, the replay objective during the Dreaming phase minimizes:
## L
dream
## = E
x∼B
## [ L(f
θ
, x) ] + λ · E
x∼P
core
## [ L(f
θ
, x) ](5)
where  P
core
represents  the  distribution  of  historical  core  training  data,  λ  is  a  stability
coefficient,  and  the  joint  optimization  prevents  both  forgetting  and  overfitting  to  the
buffered anomalies.

## Aditya Singh
## 6
## 3.3 Active Learning — Human Oracle Intervention
A  third  behavioral  mode  emerges  when  the  Doubt  function  alone  approaches  its
theoretical  maximum  —  near-uniform  predictive  entropy  indicating  complete  model
confusion.  In  this  regime,  the  system  enters  a  quarantine  state:  parameter  updates  are
paused   and   human   oracle   intervention   is   requested.   This   provides   a   principled
mechanism  for  detecting  potential  data  poisoning  or  out-of-distribution  adversarial
inputs,  and  creates  a  human-in-the-loop  checkpoint  at  the  boundary  of  the  model’s
competence.
We formalize this with a second, higher threshold τ′ applied to D(x
t
) alone, extending the
routing rule of Equation (4):
Action = { Discard (Arrogance) if V < τ
## (4′)
{ Push to Buffer B (Dreaming) if V ≥ τ and D < τ′
{ Quarantine (Active Learning) if D ≥ τ′
τ′ is set independently of τ because quarantine is triggered by confusion in isolation — a
stimulus can produce high Curiosity (large belief revision) without approaching maximal
entropy,  and  such  stimuli  are  precisely  the  ones  we  want  the  system  to  learn  from
autonomously  (Dreaming),  not  refer  to  a  human.  Quarantine  is  reserved  for  the  regime
where the model’s predictive distribution is so close to uniform that no class is preferred,
which is the operating definition we use for “complete model confusion” and a plausible
signature of data poisoning or severe distribution shift.
## 4. Positioning Within Continual Learning Literature
The  Ego  Gate  framework  synthesizes  ideas  from  several  established  research  threads
while proposing a novel unification.
Continual learning and catastrophic forgetting.  The  core  challenge  this  framework
addresses  is  well-documented  [2,  3,  4,  5].  EWC  [3]  protects  important  weights  via  a
quadratic  penalty;  the  Ego  Gate  protects  them  via  input  filtration  before  gradient
computation. These approaches are orthogonal and potentially composable: the Ego Gate
determines  which  inputs  are  processed,  while  EWC  determines  which  weights  are
updated.
Bayesian deep learning and uncertainty quantification. The Doubt function (Eq. 1)
directly  employs  predictive  entropy,  a  standard  uncertainty  measure  in  Bayesian  neural
networks  and  Monte  Carlo  Dropout  approaches  [6,  9].  The  Curiosity  function  (Eq.  2)
draws on variational inference and information-theoretic accounts of learning [7].
Intrinsic  motivation  in  reinforcement  learning.  The  Curiosity  function  (Eq.  2)  is
closely related to — and was independently motivated by the same information-theoretic
intuition  as  —  Variational  Information  Maximizing  Exploration  (VIME)  [13],  which  uses
the KL-divergence between posterior and prior beliefs over a learned dynamics model as
an  intrinsic  reward  for  exploration.  This  is  a  closer  formal  match  than  the  count-based
[10]  or  prediction-error-based  [11]  approaches,  which  use  different  (non-KL)  novelty

## Aditya Singh
## 7
signals.  The  distinction  from  VIME  is  one  of  application,  not  formulation:  VIME  biases
which  actions  an  RL  agent  takes,  by  adding  the  information-gain  term  to  the  external
reward; the Ego Gate instead uses the same kind of term to decide whether a supervised
or self-supervised model is permitted to update on a given stimulus at all, independent of
any reward signal. We view the Ego Gate’s Curiosity function as VIME’s information-gain
criterion  repurposed  from  an  exploration  bonus  into  an  input-admission  gate  —  see
Section  2.3  for  the  tractable  approximation  we  use  in  place  of  VIME’s  full  variational
treatment.
Replay-based  continual  learning.  Experience  replay  [5]  and  generative  replay  [12]
methods are the closest architectural antecedents to the Dreaming phase. The Ego Gate’s
contribution   is   providing   a   principled,   information-theoretic   criterion   for   buffer
population — replacing random or task-boundary-based selection with stimulus valuation.
Section 5 tests this claim directly.
- Empirical Validation: A Class-Incremental Proof of Concept
The  framework  as  originally  presented  was  purely  theoretical.  This  section  reports  a
small-scale proof-of-concept experiment testing its central, falsifiable claim: that curating
a bounded replay buffer by Information Value V(x
t
) reduces catastrophic forgetting more
than curating it by uniform random sampling, at the same fixed memory budget.
## 5.1 Setup
Data  and  task.  We  use  scikit-learn’s  digits  dataset  (1,797  8×8  grayscale  images  of
handwritten  digits  0–9,  64  pixel  features  normalized  to  [0,1]).  We  construct  a  standard
class-incremental  split:  Task  A  =  digits  {0,1,2,3,4}  (901  images),  Task  B  =  digits
{5,6,7,8,9} (896 images), each split 80/20 into train/test.
Model.  A  single-hidden-layer  MLP  (64  →  64  ReLU  →  10  softmax,  ≈4,800  parameters),
trained  with  plain  mini-batch  SGD  (no  momentum  or  Adam,  so  per-example  gradients
used for scoring are transparent and easy to verify).
Procedure.  (1)  Train  to  convergence  on  Task  A  (60  epochs,  lr=0.5,  batch=32);  this
checkpoint is the shared starting point for every condition below. (2) Score every Task A
training  example  using  the  frozen  checkpoint:  Doubt  via  Eq.  (1);  Curiosity  via  the
practical approximation of Eq. (2′′) (F ≈ I, the simplest member of the family in Section
2.3).  (3)  Combine  via  V  =  z-score(Doubt)  +  z-score(Curiosity),  i.e.  α=β=1,  the  naive
unweighted baseline. (4) Build six replay buffers of at most K=40 examples (≈5.6% of the
720  Task-A  training  examples):  None  (empty),  Random  (uniform  sample),  EgoGate
(top-K  by  V),  DoubtOnly  (top-K  by  Doubt  alone),  CuriosityOnly  (top-K  by  Curiosity
alone),  and  Full  (all  720  examples,  an  unbounded  oracle  upper  bound).  (5)  From  the
same  checkpoint,  continue  training  for  40  epochs  on  Task-B  data  interleaved  with  the
buffer (each condition starts from a fresh copy of the checkpoint). (6) Evaluate accuracy
on the held-out Task A test set (retention) and Task B test set (plasticity). (7) Repeat the
full pipeline — fresh initialization, fresh random buffer draw — for 20 seeds; report mean
± standard deviation, with paired t-tests across seeds.

## Aditya Singh
## 8
## 5.2 Results
## Condition
## Task A
acc.
(pre-Task
## B)
Task A acc.
(post-Task
## B)
## Forgetting
## Task B
acc.
## None0.994
## 0.004 ±
## 0.012
## 0.991 ±
## 0.012
## 0.998 ±
## 0.003
## Random0.994
## 0.820 ±
## 0.043
## 0.174 ±
## 0.043
## 0.994 ±
## 0.004
EgoGate0.994
## 0.877 ±
## 0.033
## 0.118 ±
## 0.032
## 0.992 ±
## 0.007
DoubtOnly0.994
## 0.883 ±
## 0.036
## 0.111 ±
## 0.035
## 0.992 ±
## 0.005
CuriosityOnly0.994
## 0.854 ±
## 0.036
## 0.140 ±
## 0.035
## 0.995 ±
## 0.003
## Full (oracle)0.994
## 0.981 ±
## 0.006
## 0.013 ±
## 0.006
## 0.981 ±
## 0.005
Table  1.  Task  A  retention  and  Task  B  plasticity  after  continual  training,  by  buffer-selection
condition (K=40, except None and Full). Mean ± std over 20 seeds.
Significance.  Paired  t-tests  on  forgetting  across  the  same  20  seeds:  Random  vs.
EgoGate,  t=4.22,  p=0.0005;  Random  vs.  DoubtOnly,  t=5.42,  p<0.0001;  Random  vs.
CuriosityOnly,  t=3.17,  p=0.0051;  Random  vs.  Full,  t=15.11,  p<0.0001.  EgoGate  vs.
DoubtOnly: t=0.74, p=0.47 (not significant).
Without  replay,  the  network  exhibits  near-total  catastrophic  forgetting  (99.1  points  of
Task A accuracy lost), confirming the benchmark induces the failure mode the framework
targets.  All  three  information-theoretic  selection  criteria  —  EgoGate,  DoubtOnly,  and
CuriosityOnly — significantly reduce forgetting relative to random selection at the same
fixed  buffer  size  (p  ≤  0.005  for  all  three),  supporting  the  central  hypothesis  of  this
section:  informative  buffer  curation  outperforms  random  retention  under  a  constrained
memory budget.
A  more  interesting,  and  humbler,  finding:  Doubt  alone  was  the  strongest  individual
predictor   of   buffer   value   in   this   setting,   and   the   unweighted   Doubt+Curiosity
combination (EgoGate) was statistically indistinguishable from Doubt alone (p=0.47) — it
did  not  significantly  improve  on  Doubt,  and  numerically  trailed  it  slightly.  This  is  an
informative negative result rather than a failure: it suggests the naive α=β=1 weighting
is     not     optimal     here,     and     lends     direct     empirical     motivation     to     the
adaptive-coefficient-learning  direction  in  Section  6.1,  rather  than  supporting  an  implicit
assumption  that  equal  weighting  is  sufficient.  The  Full-replay  oracle  nearly  eliminates
forgetting  (1.3%  residual),  confirming  the  upper  bound  and  indicating  that  the  gap
between EgoGate and Full is attributable to the bounded buffer size K, not a flaw in the
underlying task.

## Aditya Singh
## 9
5.3 Limitations of This Evaluation
This experiment uses one small dataset, one architecture, one task boundary, and one fixed
combination rule; Section 7 discusses how far these results can and cannot be generalized.
We report it as a first, falsifiable test of the framework’s central claim — not as evidence
that the Ego Gate is validated at the scale the Introduction motivates it for.
## 6. Open Research Directions
## 6.1 Adaptive Coefficient Learning
The sensitivity coefficients α and β in Equation (3) govern the system’s relative weighting
of   uncertainty   versus   information   gain.   In   the   base   formulation,   these   are   fixed
hyperparameters.  A  natural  extension  is  to  learn  them  adaptively:  treating  α  and  β  as
outputs  of  a  meta-controller  trained  via  reinforcement  learning,  where  rewards  reflect
downstream task performance and memory stability. The threshold τ itself admits similar
treatment.
We    note    that    meta-RL    controllers    of    this    kind    are    themselves    notoriously
sample-inefficient  and  can  introduce  their  own  training  instability  —  using  a  hard
problem (gating credit assignment) to solve another hard problem (online policy learning)
is not obviously a net simplification. A more immediately tractable interim approach, and
the one we would prioritize first, is periodic grid search or Bayesian optimization over (α,
β,  τ)  against  a  small  held-out  validation  stream,  re-fit  on  a  fixed  schedule  (e.g.  at  each
task boundary) rather than online. Our own result in Section 5 — that the equal-weighted
combination  underperformed  Doubt  alone  —  suggests  that  even  this  simpler,  periodic
re-weighting would likely yield a meaningful improvement before resorting to a learned
meta-controller.
6.2 Application to Retrieval-Augmented Generation
A  promising  near-term  application  is  adapting  the  Ego  Gate  as  a  response-depth
calibration  layer  in  retrieval-augmented  generation  (RAG)  systems.  Current  answer
engines  treat  each  query  as  an  isolated  retrieval  task,  with  no  model  of  the  user’s
knowledge  state  or  the  conceptual  dependencies  between  topics.  The  Doubt  function
could  measure  the  system’s  uncertainty  about  user  comprehension  rather  than  output
class  probability;  the  Curiosity  function  could  measure  how  far  the  current  answer
deviates from what a user at this knowledge level actually requires. The Ego Gate would
then  determine  whether  to  retrieve  deeper,  restructure  the  explanation,  or  surface  a
missing conceptual prerequisite.

## Aditya Singh
## 10
## 6.3 Scaling Beyond Toy Benchmarks
Section  5  provides  a  first  proof  of  concept  on  a  small,  two-task,  classification-style
benchmark.  The  natural  next  steps  are:  (1)  repeating  the  evaluation  on  standard
continual-learning    benchmarks    with    more    tasks    and    higher-dimensional    inputs
(Split-CIFAR,   Permuted-MNIST,   CORe50);   (2)   replacing   the   identity-scaled   Fisher
approximation in Section 2.3 with the diagonal Fisher already computed by EWC, or with
a  full  variational  treatment  along  the  lines  of  VIME  [13];  (3)  testing  whether  the
qualitative  finding  that  Doubt  alone  matches  or  exceeds  the  naive  Doubt+Curiosity
combination holds at larger scale, or whether Curiosity’s relative contribution grows with
task  diversity;  and  (4)  extending  the  formalism  beyond  fixed-class  classification  to
token-level  or  embedding-level  stimuli,  which  is  the  necessary  bridge  between  the
present   framework   and   the   generative,   foundation-model   settings   invoked   in   the
## Introduction.
- Limitations and Scope
We  are  explicit  here  about  what  this  paper  does  and  does  not  establish,  consolidating
points raised throughout.
Scope  of  the  formalism.   Equations   (1)–(4′)   are   defined   over   a   discrete   output
distribution  P(y
c
## |x
t
),  the  natural  setting  for  classification-style  continual  learners.  The
framework  as  written  does  not  directly  specify  how  Doubt  and  Curiosity  would  be
computed for token-level generative models, where “the prediction” is a distribution over
a  large  vocabulary  at  every  position  rather  than  a  single  distribution  over  a  small  fixed
label set. Bridging this gap — most plausibly via the RAG application sketched in Section
6.2,  or  via  a  sequence-level  reformulation  of  Equations  (1)–(2)  —  is  the  most  important
piece of unfinished work in this line of research.
Scope  of  the  empirical  validation.  The  result  in  Section  5  uses  one  small  dataset
(1,797  8×8  images),  one  architecture  (a  single-hidden-layer  MLP),  one  task  boundary,
and   one   non-adaptive,   equally-weighted   combination   rule.   It   is   evidence   that
information-theoretic buffer curation can outperform random curation at a fixed memory
budget  in  at  least  one  small  setting  —  consistent  with,  but  far  short  of  confirming,  the
broader  claims  in  the  Introduction.  We  would  caution  against  citing  this  result  as
evidence that the Ego Gate “works” for AGI-scale continual learning; it is evidence that
the core mechanism is not obviously wrong, which is a substantially weaker claim.
Scope  of  the  Curiosity  approximation.  The  C(x
t
)  used  in  Section  5  sets  F(θ)  ≈  I  in
Equation  (2′),  discarding  the  per-parameter  importance  weighting  that  makes  the
Fisher-information  approximation  more  than  a  generic  gradient-norm  heuristic.  The
reported results should be read as a lower bound on what a properly Fisher-weighted or
fully variational (VIME-style) implementation of Curiosity could achieve.
Psychological  grounding.  We  use  doubt,  curiosity,  arrogance,  and  dreaming  as
organizing  labels  for  formal  operations  that  have  genuine,  independently-motivated
grounding  in  Bayesian  uncertainty  estimation,  information-theoretic  belief  revision,

## Aditya Singh
## 11
confirmation bias, and complementary learning systems theory, respectively [6, 7, 8, 15].
We do not claim that these four labels constitute an exhaustive or validated taxonomy of
the  psychological  mechanisms  underlying  human  memory  consolidation;  the  mapping  is
best read as inspiration and exposition rather than a tested theory of human cognition.
## 8. Conclusion
The transition from narrow AI to general intelligence requires systems that do not merely
process  data,  but  actively  evaluate  its  worth.  The  Ego  Gate  provides  a  mathematical
framework  for  continuous  learning  models  to  self-regulate  cognitive  load  —  discarding
the  redundant,  buffering  the  anomalous,  and  protecting  foundational  knowledge  from
both noise and forgetting.
The framework’s four psychological primitives — doubt as predictive entropy, curiosity as
KL-divergence  (here  approximated  via  Fisher  information,  Section  2.3),  arrogance  as
principled  discarding  grounded  in  the  mechanics  of  confirmation  bias,  and  dreaming  as
structured  replay  —  are  not  merely  evocative  labels.  Each  is  tied  to  a  formal  operation
and,  in  the  case  of  doubt  and  curiosity  combined,  to  a  measurable  reduction  in
catastrophic   forgetting   in   a   small   proof-of-concept   benchmark   (Section   5).   Taken
together, they constitute a pre-architectural selectivity layer largely absent from current
continual  learning  approaches  —  though  substantial  work  remains:  scaling  beyond  toy
benchmarks,  tightening  the  Curiosity  approximation,  and  learning  the  α/β/τ  coefficients
adaptively,  before  this  claim  can  be  made  at  the  scale  the  title  invokes.  Whether
implemented as a standalone filter or integrated into existing EWC or replay frameworks,
the Ego Gate proposes that the path to stable, lifelong learning runs through a question
every intelligent system must eventually answer: what is this worth knowing?
## References
[1]  Shumailov,  I.  et  al.  (2023).  The  Curse  of  Recursion:  Training  on  Generated  Data  Makes
Models Forget. arXiv:2305.17493.
[2] McCloskey, M., & Cohen, N.J. (1989). Catastrophic interference in connectionist networks.
Psychology of Learning and Motivation, 24, 109–165.
[3] Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS,
## 114(13), 3521–3526.
[4] Rusu, A.A. et al. (2016). Progressive Neural Networks. arXiv:1606.04671.
[5]  Robins,  A.  (1995).  Catastrophic  forgetting,  rehearsal  and  pseudorehearsal.  Connection
## Science, 7(2), 123–146.
[6] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML 2016.
[7] Blundell, C. et al. (2015). Weight Uncertainty in Neural Networks. ICML 2015.
[8] McClelland, J.L., McNaughton, B.L., & O’Reilly, R.C. (1995). Why there are complementary
learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419–457.
[9] Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
NeurIPS 2017.

## Aditya Singh
## 12
[10] Bellemare, M.G. et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation.
NeurIPS 2016.
[11] Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction. ICML
## 2017.
[12] Shin, H. et al. (2017). Continual Learning with Deep Generative Replay. NeurIPS 2017.
[13] Houthooft, R., Chen, X., Duan, Y., Schulman, J., De Turck, F., & Abbeel, P. (2016). VIME:
Variational Information Maximizing Exploration. NeurIPS 2016. arXiv:1605.09674.
[14]  Guo,  C.,  Pleiss,  G.,  Sun,  Y.,  &  Weinberger,  K.Q.  (2017).  On  Calibration  of  Modern  Neural
Networks. Proceedings of the 34th ICML, PMLR 70, 1321–1330.
[15]  Nickerson,  R.S.  (1998).  Confirmation  Bias:  A  Ubiquitous  Phenomenon  in  Many  Guises.
Review of General Psychology, 2(2), 175–220.
[16]  Amari,  S.  (1998).  Natural  Gradient  Works  Efficiently  in  Learning.  Neural  Computation,
## 10(2), 251–276.