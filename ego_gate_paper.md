

Aditya Singh  |  Amity Centre for Artificial Intelligence  |
## The Ego Gate
Synthesizing Psychological Valuation for Continuous AGI Learning
## Aditya Singh
Amity Centre for Artificial Intelligence

## Abstract
Artificial General Intelligence (AGI) systems face two existential failure modes: model collapse from
synthetic data recursion, and catastrophic forgetting from unconstrained continual learning. Standard
architectures lack an intrinsic mechanism to evaluate the informational value of incoming stimuli
before committing to weight updates. This paper proposes the Ego Gate (V), a unified
mathematical value filter that reverse-engineers four human psychological states — doubt, curiosity,
arrogance, and dreaming — into formal algorithms. By quantifying epistemic uncertainty via
predictive entropy and belief revision via KL-divergence, the Ego Gate dynamically routes incoming
stimuli: discarding low-value redundancies and buffering high-value anomalies for structured
experience replay. We formalize the framework, discuss its architectural implications, position it
within the continual learning literature, and identify open research directions including adaptive
threshold learning and application to retrieval-augmented generation systems.

-  Introduction: The Stability-Plasticity Dilemma
For an artificial neural network to function as a dynamic, lifelong learner, it must receive continuous
environmental stimulus. Without fresh real-world data, generative models degrade into structural
collapse — a phenomenon increasingly documented as models trained on synthetic recursions of
their own outputs lose distributional fidelity [1]. However, unrestricted ingestion of all incoming
stimuli introduces a competing pathology: catastrophic forgetting, wherein novel parameter updates
overwrite established weight pathways θ, destroying previously consolidated knowledge [2].
This tension defines the stability-plasticity dilemma: a learning system must remain plastic
enough to acquire new knowledge, yet stable enough to retain prior representations. The field of
continual learning has produced several approaches to this problem — Elastic Weight Consolidation
(EWC) [3] penalizes updates to weights important for previous tasks; Progressive Neural Networks
[4] allocate new capacity for each task; replay-based methods [5] store or generate representative
samples from prior distributions to interleave with new data. Each addresses the dilemma
structurally, at the architecture or training-procedure level.
This paper proposes a complementary, pre-architectural approach: a stimulus valuation filter that
operates before any weight update occurs. Rather than managing the consequences of learning
indiscriminately, the Ego Gate asks a prior question: is this stimulus worth learning from at all?

Aditya Singh  |  Amity Centre for Artificial Intelligence  |
The biological precedent is direct. The human brain does not consolidate every sensory input into
long-term memory. An intrinsic psychological filter — loosely corresponding to what we might call
ego, or self-concept — assigns value to incoming experience, discarding the mundane and
consolidating the anomalous, particularly during sleep. The Ego Gate framework translates four
evolved psychological mechanisms — doubt, curiosity, arrogance, and dreaming — into calculable
metrics, shifting AGI architecture from static computational graphs toward biologically-inspired,
dynamic value filters.
-  Formalization of the Ego Gate
Let x_t denote an incoming stimulus at time t. The Ego Gate computes a scalar Total Information
Value V(x_t) from two primary constituent functions: Doubt and Curiosity. This value is then
compared against a learned threshold to determine routing behavior.
## 2.1  Epistemic Uncertainty — The Doubt Function
To simulate self-doubt, the model must quantify its own predictive confusion on a given stimulus.
We formalize this using predictive entropy over the model's output distribution — a measure well-
established in Bayesian deep learning as a proxy for epistemic uncertainty [6].
Let P(yc | xt) denote the probability the model assigns to outcome c given stimulus x_t. The Doubt
function D(x_t) is the Shannon Entropy of this distribution:
D(x_t)  =  -∑_c  P(yc | xt)  log  P(yc | xt)     (1)
When the model generates a highly confident prediction, entropy approaches zero — the model
experiences no doubt. A uniform distribution across outcomes maximizes entropy, flagging the
stimulus as informationally valuable precisely because the model is internally conflicted. Crucially,
high doubt is not equivalent to error: it signals that the stimulus occupies a region of the model's
input space where its representation is genuinely underdetermined.
## 2.2  Information Gain — The Curiosity Function
Curiosity is defined as the mathematical surprise generated when a new stimulus revises the model's
foundational beliefs. We measure this via Kullback-Leibler divergence between prior and
posterior parameter distributions — a formulation directly motivated by the information-theoretic
account of learning as Bayesian belief revision [7].
Let P(θ) represent the model's prior parameter distribution, and P(θ | x_t) the posterior after
processing x_t. The Curiosity function C(x_t) is:
C(x_t)  =  D_KL( P(θ | x_t) ‖ P(θ) )  =  ∫ P(θ | x_t) log [ P(θ | x_t) / P(θ) ] dθ     (2)
A stimulus perfectly aligned with existing priors yields KL-divergence of zero — the model already
represents this phenomenon adequately, and no belief revision occurs. A stimulus that contradicts
the model's world-model generates large C(x_t), mathematically compelling investigation and
adaptation. This function formalizes the intuition that genuine curiosity is not novelty-seeking per
se, but contradiction-seeking: the drive to resolve the gap between expectation and observation.


Aditya Singh  |  Amity Centre for Artificial Intelligence  |
2.3  The Value Equation and Behavioral Thresholds
The unified Information Value V(x_t) is a weighted combination of Doubt and Curiosity:
V(x_t)  =  α · D(x_t)  +  β · C(x_t)     (3)
where α, β ∈ ℝ≥0 are sensitivity coefficients governing the model's relative responsiveness to
uncertainty versus belief revision. When α > β, the system prioritizes inputs where it is internally
confused (high entropy); when β > α, it prioritizes inputs that maximally update its beliefs. In the
base formulation, α and β are treated as hyperparameters; Section 4 addresses their adaptive
formulation.
Incoming stimuli are routed via the Arrogance Threshold τ:
Action =  { Discard (Arrogance)     if V(x_t) < τ
{ Push to Buffer B (Dreaming)  if V(x_t) ≥ τ     (4)
The threshold τ defines the boundary between two behavioral modes formalized in the following
section.
## 3.  Architectural Implications
## 3.1  Arrogance — Redundancy Filtration
When V(x_t) < τ, the system acts with arrogance: the stimulus is discarded without triggering
backpropagation. This has two concrete consequences. First, it provides substantial computational
savings by bypassing the gradient computation pipeline for low-value inputs. Second, and more
importantly, it protects foundational weight pathways from erosion by redundant data — the primary
mechanism of catastrophic forgetting in naive continual learning systems.
The term arrogance is deliberately chosen: the system is not passively ignoring low-value stimuli, but
actively asserting that its current representation is sufficient — that nothing in this stimulus warrants
revision.
## 3.2  Dreaming — Structured Experience Replay
When V(x_t) ≥ τ, the stimulus is placed into an active memory buffer B. During designated offline
computation periods — the Dreaming phase — the model performs structured experience replay:
buffered high-value stimuli are synthesized alongside sampled historical core data and used to
update weights in a controlled, interleaved manner.
This approach is directly motivated by replay-based continual learning methods [5] and the
complementary learning systems theory of biological memory consolidation [8], which proposes that
the hippocampus rapidly encodes new experiences while the neocortex slowly integrates them
during sleep. The Ego Gate formalizes the selection criterion for what enters this replay buffer — a step
that existing replay methods typically handle via random sampling or task boundaries, rather than
principled information-theoretic valuation.



Aditya Singh  |  Amity Centre for Artificial Intelligence  |
Formally, the replay objective during the Dreaming phase minimizes:
L_dream  =  E_{x ~ B} [ L(f_θ, x) ]  +  λ · E_{x ~ P_core} [ L(f_θ, x) ]     (5)
where P_core represents the distribution of historical core training data, λ is a stability coefficient,
and the joint optimization prevents both forgetting and overfitting to the buffered anomalies.
## 3.3  Active Learning — Human Oracle Intervention
A third behavioral mode emerges when D(x_t) approaches its theoretical maximum — near-
uniform predictive entropy indicating complete model confusion. In this regime, the system enters a
quarantine state: parameter updates are paused and human oracle intervention is requested. This
provides a principled mechanism for detecting potential data poisoning or out-of-distribution
adversarial inputs, and creates a human-in-the-loop checkpoint at the boundary of the model's
competence.
## 4.  Positioning Within Continual Learning Literature
The Ego Gate framework synthesizes ideas from several established research threads while
proposing a novel unification.
Continual learning and catastrophic forgetting. The core challenge this framework addresses is
well-documented [2, 3, 4, 5]. EWC [3] protects important weights via a quadratic penalty; the Ego
Gate protects them via input filtration before gradient computation. These approaches are
orthogonal and potentially composable: the Ego Gate determines which inputs are processed, while
EWC determines which weights are updated.
Bayesian deep learning and uncertainty quantification. The Doubt function (Eq. 1) directly
employs predictive entropy, a standard uncertainty measure in Bayesian neural networks and Monte
Carlo Dropout approaches [6, 9]. The Curiosity function (Eq. 2) draws on variational inference and
information-theoretic accounts of learning [7].
Intrinsic motivation in reinforcement learning. The Curiosity function bears formal resemblance
to intrinsic motivation signals in RL — particularly count-based exploration bonuses and prediction-
error-driven curiosity [10, 11]. The key distinction is that the Ego Gate applies this signal as a filter
on supervised or self-supervised learning, not as an exploration bonus in a reward framework.
Replay-based continual learning. Experience replay [5] and generative replay [12] methods are
the closest architectural antecedents to the Dreaming phase. The Ego Gate's contribution is
providing a principled, information-theoretic criterion for buffer population — replacing random or
task-boundary-based selection with stimulus valuation.
## 5.  Open Research Directions
## 5.1  Adaptive Coefficient Learning
The sensitivity coefficients α and β in Equation (3) govern the system's relative weighting of
uncertainty versus information gain. In the base formulation, these are fixed hyperparameters. A
natural extension is to learn them adaptively: treating α and β as outputs of a meta-controller trained
via reinforcement learning, where rewards reflect downstream task performance and memory
stability. The threshold τ itself admits similar treatment — self-adjustment based on hardware

Aditya Singh  |  Amity Centre for Artificial Intelligence  |
constraints, environmental volatility, and current buffer occupancy would move the system toward
genuine autonomy in managing its own cognitive load.
5.2  Application to Retrieval-Augmented Generation
A promising near-term application is adapting the Ego Gate as a response-depth calibration layer
in retrieval-augmented generation (RAG) systems. Current answer engines treat each query as an
isolated retrieval task, with no model of the user's knowledge state or the conceptual dependencies
between topics. The Doubt function could measure the system's uncertainty about user comprehension
rather than output class probability; the Curiosity function could measure how far the current
answer deviates from what a user at this knowledge level actually requires. The Ego Gate would then
determine whether to retrieve deeper, restructure the explanation, or surface a missing conceptual
prerequisite.
## 5.3  Empirical Validation
The framework as presented is theoretical. Empirical validation requires: (1) implementing the Ego
Gate as a modular filter in a continual learning benchmark (Split-CIFAR, Permuted-MNIST, or
CORe50); (2) comparing buffer population quality under information-theoretic selection versus
random selection; (3) measuring the trade-off between compute savings from Arrogance-mode
filtering and downstream task accuracy. These experiments would establish whether the
information-theoretic valuation of Equation (3) produces meaningfully better replay buffers than
simpler heuristics.
## 6.  Conclusion
The transition from narrow AI to general intelligence requires systems that do not merely process
data, but actively evaluate its worth. The Ego Gate provides a rigorous mathematical framework for
continuous learning models to self-regulate cognitive load — discarding the redundant, buffering the
anomalous, and protecting foundational knowledge from both noise and forgetting.
The framework's four psychological primitives — doubt as predictive entropy, curiosity as KL-
divergence, arrogance as principled discarding, and dreaming as structured replay — are not
metaphors. They are formal operations that, taken together, constitute a pre-architectural selectivity layer
absent from current continual learning approaches. Whether implemented as a standalone filter or
integrated into existing EWC or replay frameworks, the Ego Gate proposes that the path to stable,
lifelong learning runs through a question every intelligent system must eventually answer: what is this
worth knowing?

## References
[1] Shumailov, I. et al. (2023). The Curse of Recursion: Training on Generated Data Makes Models
Forget. arXiv:2305.17493.
[2] McCloskey, M., & Cohen, N.J. (1989). Catastrophic interference in connectionist networks.
Psychology of Learning and Motivation, 24, 109–165.
[3] Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS,
## 114(13), 3521–3526.

Aditya Singh  |  Amity Centre for Artificial Intelligence  |
[4] Rusu, A.A. et al. (2016). Progressive Neural Networks. arXiv:1606.04671.
[5] Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. Connection Science,
## 7(2), 123–146.
[6] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. ICML 2016.
[7] Blundell, C. et al. (2015). Weight Uncertainty in Neural Networks. ICML 2015.
[8] McClelland, J.L., McNaughton, B.L., & O'Reilly, R.C. (1995). Why there are complementary
learning systems in the hippocampus and neocortex. Psychological Review, 102(3), 419–457.
[9] Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
NeurIPS 2017.
[10] Bellemare, M.G. et al. (2016). Unifying Count-Based Exploration and Intrinsic Motivation.
NeurIPS 2016.
[11] Pathak, D. et al. (2017). Curiosity-driven Exploration by Self-Supervised Prediction. ICML
## 2017.
[12] Shin, H. et al. (2017). Continual Learning with Deep Generative Replay. NeurIPS 2017.