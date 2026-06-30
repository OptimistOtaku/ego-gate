## Abstract

Artificial intelligence systems intended to learn continuously face a stability-plasticity problem. A learner that receives too little fresh, grounded experience becomes stale and increasingly dependent on incomplete internal representations. A learner trained recursively on synthetic outputs risks distributional degradation. A learner that updates indiscriminately on every new observation is vulnerable to interference, contamination, and catastrophic forgetting. This paper proposes the Ego Gate, a stimulus-valuation and memory-consolidation framework that decides which observations should be discarded, retained in active memory, quarantined for review, or consolidated into model parameters. The initial formulation combines predictive uncertainty (Doubt) and expected parameter influence (Curiosity), then uses offline replay (Dreaming) to integrate selected memories while preserving established knowledge. A reproducible proof of concept on Split-Digits evaluates one component of this vision: fixed-budget replay-buffer selection. Across 20 paired seeds, the equal-weight EgoGate policy reduces forgetting by 39.4% relative to random storage at the same memory and replay-compute budget (Holm-corrected p = 1.41e-5, Cohen's dz = 1.49). Doubt alone produces the best bounded-buffer mean, but does not significantly outperform EgoGate after multiplicity correction. These results support model-state-aware memory selection in a small setting; they do not yet validate the complete online gate or foundation-model-scale continual learning. We define the architectural objective, distinguish the current evidence from the long-term claim, and specify the experiments required to develop the proposal into a task-free, budget-aware continual-learning system.

## 1. Introduction

An intelligent system deployed in the world cannot remain permanently frozen. Environments change, facts become obsolete, new concepts appear, and users generate evidence that was absent from the original training distribution. Yet continuous parameter updating is not automatically beneficial. Sequential learning can overwrite representations needed for earlier tasks, while noisy or adversarial observations can cause harmful updates. At the opposite extreme, a system isolated from fresh evidence cannot correct outdated beliefs and may repeatedly amplify errors in its own generated data.

This tension is the stability-plasticity dilemma: the learner must remain plastic enough to acquire useful knowledge and stable enough to retain prior competence. Existing continual-learning methods address this through parameter regularization, architectural expansion, rehearsal, generative replay, or constrained optimization. The Ego Gate adds a complementary control question before consolidation:

EQUATION: What is this observation worth learning, storing, verifying, or forgetting?

The long-term objective is not a single uncertainty score. It is a cognitive control layer between an environmental stream and long-term model parameters. The control layer values incoming stimuli, maintains bounded active memory, and schedules controlled offline consolidation. The psychological terms Doubt, Curiosity, Arrogance, and Dreaming are organizing metaphors for computable operations; they are not claims of consciousness or a complete theory of human memory.

### 1.1 Scope and contributions

This paper makes four bounded contributions:

- It defines a stimulus-routing architecture linking uncertainty, expected learning value, active memory, quarantine, and offline replay.
- It corrects the tractable Curiosity approximation as a local Fisher-weighted predictive change and states its supervision and compute requirements.
- It provides a reproducible, fixed-compute Split-Digits study with raw seed-level outputs, stronger controls, paired effect sizes, and multiplicity correction.
- It identifies the gap between replay-buffer selection, which is tested here, and task-free online admission, which remains the central thesis-level research problem.

The term AGI motivates the need for lifelong learning, but the empirical claim in this paper is limited to a small classification benchmark.

## 2. Failure modes motivating stimulus control

### 2.1 Stale or weakly grounded models

A frozen model cannot incorporate later evidence into its parameters. Retrieval and external tools can provide current information without weight updates, but they do not by themselves solve long-term representation change, skill acquisition, or consolidation. The relevant failure is not that limited stimulus mechanically causes hallucination. Rather, inadequate grounding and outdated knowledge increase the conditions under which unsupported generation can persist without correction.

### 2.2 Recursive synthetic-data degradation

Training repeatedly on generated samples can distort the learned distribution when synthetic data replace rather than complement representative real data. The Ego Gate does not solve model collapse by itself. It supplies a place to estimate source reliability, redundancy, novelty, and contradiction before generated or observed data are admitted into memory and training.

### 2.3 Catastrophic forgetting and harmful plasticity

When non-stationary data are learned sequentially, gradients from new observations can interfere with parameters supporting earlier competence. Indiscriminate updating also exposes the learner to label noise, outliers, and poisoning. A useful controller must therefore estimate both potential learning value and potential damage.

## 3. Ego Gate formalization

Let x_t be an incoming observation and let the learner have parameters theta_t and active memory B_t. The initial Ego Gate score combines Doubt D and Curiosity C:

EQUATION: V(x_t) = alpha D(x_t) + beta C(x_t), where alpha, beta >= 0.

The score is a baseline, not the final policy. A deployable gate must additionally account for coverage, source reliability, interference risk, memory capacity, and scoring cost.

### 3.1 Doubt: predictive uncertainty

For a C-class predictive distribution p(y = c | x_t, theta), the original Doubt signal is predictive entropy:

EQUATION: D(x_t) = - sum from c=1 to C of p_c log(p_c).

Entropy is low for concentrated predictions and high near a uniform distribution. However, a single softmax entropy is total predictive uncertainty, not a clean measure of epistemic uncertainty. It can reflect ambiguity or noise as well as missing model knowledge, and modern networks can be confidently wrong under distribution shift. Temperature calibration improves probability calibration but does not separate epistemic and aleatoric uncertainty. A stronger implementation should estimate model disagreement with ensembles or MC Dropout and evaluate calibration explicitly.

### 3.2 Curiosity: expected model change

The motivating Bayesian quantity is information gain between a posterior after observing x_t and the prior belief over parameters. A full parameter-posterior KL divergence is unavailable for an ordinary deterministic network. For a small update delta and model predictive distribution p_theta, information geometry gives the local approximation:

EQUATION: KL(p_(theta+delta) || p_theta) is approximately (1/2) delta^T F(theta) delta.

For one SGD step with gradient g_t and learning rate eta:

EQUATION: delta = - eta g_t, so C(x_t) is approximately (eta^2 / 2) g_t^T F(theta) g_t.

The present experiment uses F approximately equal to the identity. After score standardization, constant factors do not affect ranking:

EQUATION: C(x_t) is proportional to || gradient_theta L(f_theta(x_t), y_t) ||_2^2.

This is an exact squared per-example gradient norm for the experimental MLP, but it is only a heuristic proxy for Bayesian information gain. It requires a label and a backward-equivalent computation. Therefore, it cannot simultaneously be described as an unlabeled, forward-only admission score. Future work must test diagonal-Fisher weighting, predictive functional change, pseudo-label variants, and self-supervised losses.

### 3.3 Routing policy

The conceptual controller selects one of four actions:

- Discard: reject a redundant, low-value observation without storing it.
- Retain: place a potentially useful observation in bounded active memory.
- Quarantine: delay learning when reliability or competence is insufficient.
- Immediate update: permit a controlled update when delay is costly and risk is low.

A threshold-only baseline can be written as:

EQUATION: discard if V < tau; retain if V >= tau and D < tau_prime; quarantine if D >= tau_prime.

High entropy alone is not a poisoning detector. Quarantine must ultimately be a calibrated selective-prediction decision using uncertainty, source evidence, outlier detection, and the cost of an incorrect update.

## 4. Memory and Dream consolidation

### 4.1 Active memory

Active memory is a bounded set of observations and metadata available for later replay. Top-K selection by difficulty can overrepresent outliers, mislabeled samples, or a narrow class region. Buffer management must therefore balance value with coverage and diversity. The strengthened experiment includes class-balanced random storage, embedding k-center coverage, and a diversity-constrained EgoGate candidate pool to expose this issue.

### 4.2 Dreaming as controlled offline consolidation

Dreaming denotes a scheduled phase in which selected new observations are interleaved with representative historical memories. A simple objective is:

EQUATION: L_dream = E over new memory of L(theta; x) + lambda E over core memory of L(theta; x).

The deeper claim is about scheduling and control, not merely rehearsal. A complete system should decide when consolidation is needed, which memories to retrieve, how many updates to spend, and which parameters or representations require protection. Event triggers could include memory pressure, detected drift, rising interference, or a fixed offline budget.

### 4.3 Separation of storage, retrieval, and optimization

Three decisions are often conflated:

- Storage selection determines what remains in the bounded buffer.
- Replay retrieval determines what is sampled for a particular update.
- Optimization determines how new and replay gradients are combined.

The current study evaluates storage selection while holding replay sampling and compute fixed. A full thesis must vary these decisions independently.

## 5. Relationship to prior work

The Ego Gate combines established ideas rather than introducing uncertainty or replay selection from first principles. Predictive entropy and Bayesian disagreement are standard uncertainty tools. VIME uses posterior information gain as an intrinsic exploration reward. EWC uses Fisher information to protect important parameters. GSS selects replay samples using gradient-space diversity. MIR retrieves memories predicted to suffer maximal interference from an upcoming update. Gradient Coreset Replay approximates full-data gradients with selected samples. Recent CORE and GRASP methods use cognitive or curriculum-inspired replay selection, while adaptive memory replay treats retrieval as a resource-allocation problem.

The defensible novelty target is therefore the joint control problem: task-free stimulus admission, bounded retention, human-review escalation, and offline consolidation under explicit memory and compute budgets. The psychological terminology aids communication, but does not constitute algorithmic novelty.

## 6. Reproducible proof of concept

### 6.1 Research question

Does model-state-aware selection produce a better fixed-size replay pool than random selection when memory and replay compute are held constant?

This is narrower than testing the complete Ego Gate. Task-A examples are scored after Task-A training and retained for later replay. Incoming Task-B examples are not filtered.

### 6.2 Data, model, and protocol

The study uses scikit-learn Digits: 1,797 grayscale 8 by 8 images. Task A contains classes 0 through 4 and Task B contains classes 5 through 9. Each task is split 80/20 with stratification. The model is a 64 to 64 ReLU to 10 MLP trained with SGD, learning rate 0.5, and batch size 32.

Task A is trained for 60 epochs. At the frozen checkpoint, every Task-A training sample receives Doubt, Curiosity, and equal-weight standardized EgoGate scores. Each bounded selector stores K = 40 of 720 Task-A training examples. During 40 Task-B epochs, every 32-example new-task batch receives two replay examples and one cross-entropy loss is averaged over the combined batch. This gives approximately 40 to 46 replay exposures per epoch. The FullMemory condition stores all Task-A examples but receives the same replay batch size; it tests access to broader coverage, not unlimited replay compute.

All conditions start from the same Task-A checkpoint within each seed. The full pipeline is repeated for 20 seeds. Raw results, configurations, buffer diagnostics, and analysis code are committed with the paper.

[[EXPERIMENT_FIGURE]]

### 6.3 Results

[[RESULTS_TABLE]]

EgoGate reduces mean forgetting from 0.193 under random storage to 0.117, an absolute reduction of 0.076 and a relative reduction of 39.4%. The paired effect is large (t = 6.64, uncorrected p = 2.35e-6, Holm-corrected p = 1.41e-5, Cohen's dz = 1.49). Doubt reduces forgetting by 46.8% relative to random (Holm p = 1.34e-6), and Curiosity reduces it by 38.7% (Holm p = 1.30e-5).

Doubt has the best bounded-buffer mean, but its direct advantage over EgoGate is not significant after correction (uncorrected p = 0.048; Holm p = 0.144). Consequently, the experiment supports information-aware storage but does not show that alpha = beta = 1 is optimal or that combining both signals is superior to either component.

Task-B accuracy remains approximately 99% for the main bounded policies. FullMemory has the best Task-A retention but lower and more variable Task-B performance, reinforcing that retention alone is not a sufficient objective. Stability and plasticity must be optimized jointly.

### 6.4 What the experiment does not establish

The study does not test online admission, unlabeled scoring, threshold adaptation, quarantine accuracy, consolidation scheduling, contamination, or foundation models. It uses one dataset, one architecture, one task boundary, and one replay ratio. The result is evidence that the core mechanism is not obviously wrong, not evidence that Ego Gate already solves lifelong learning.

## 7. Thesis-level research program

The central thesis question is:

EQUATION: Can a task-free learner jointly control admission, bounded memory, and offline consolidation to maximize future performance under explicit memory, compute, and supervision budgets?

A stronger policy should estimate five distinct quantities: epistemic uncertainty, expected learning value, interference risk, memory coverage, and source reliability. Rather than immediately using meta-reinforcement learning, a contextual bandit can learn routing coefficients from delayed validation reward while satisfying hard resource constraints.

The recommended empirical progression is:

- Reproduce storage and retrieval baselines on Split-CIFAR-10/100, including reservoir replay, class-balanced replay, herding, GSS, MIR, gradient coresets, and GRASP-style curricula.
- Implement true single-pass, task-free admission with running score normalization and no future task boundaries.
- Add redundant data, label noise, rare classes, covariate drift, and poisoning attempts; measure harmful-update prevention and false rejection of valid data.
- Compare concurrent, periodic, and event-triggered Dream consolidation under equal total gradient updates.
- Extend only after these controls to CORe50 and a small pretrained model with parameter-efficient updates.

Primary metrics should include average accuracy, forgetting, forward transfer, area under the online accuracy curve, calibration, contamination detection, false discard rate, memory, score-computation cost, update count, and wall-clock time.

## 8. Limitations and responsible interpretation

The biological analogy is inspiration, not evidence that the implementation replicates human cognition or sleep. Human memory is reconstructive, multimodal, and shaped by attention, emotion, goals, and social context; the four labels used here are not an exhaustive taxonomy.

The present Curiosity score is supervised and gradient-based. The compute saved by rejecting a sample must therefore be measured after including score computation. The predictive-entropy Doubt score is not a reliable stand-alone detector for epistemic uncertainty, out-of-distribution inputs, or poisoning. Top-K value selection can sacrifice coverage, and a bounded replay buffer cannot guarantee preservation of arbitrary details.

Finally, continual weight updating is not the only route to current or personalized behavior. Retrieval, external memory, tools, adapters, and modular parameter stores may be safer for many kinds of knowledge. Ego Gate should be evaluated as a controller across these memory mechanisms, not assumed to route every observation into base-model weights.

## 9. Conclusion

The long-term Ego Gate vision is a control architecture for lifelong learning: real-world observations are evaluated before they influence active memory or model parameters; important and representative experiences are retained; uncertain or unreliable observations are quarantined; and offline consolidation integrates new evidence while protecting established competence.

The strengthened Split-Digits study supplies reproducible evidence for one narrow component. At equal memory and replay compute, model-state-aware storage substantially reduces forgetting relative to random storage. The result survives paired testing and multiplicity correction, but Doubt alone remains competitive with the combined score and the benchmark is intentionally small.

The next scientific step is not a larger claim. It is a harder test: an online, task-free, budget-aware policy that jointly decides what to learn, what to remember, what to verify, and when to consolidate.

## References

1. Shumailov, I. et al. The Curse of Recursion: Training on Generated Data Makes Models Forget. arXiv:2305.17493, 2023.
2. McCloskey, M. and Cohen, N. Catastrophic Interference in Connectionist Networks. Psychology of Learning and Motivation 24, 1989.
3. Kirkpatrick, J. et al. Overcoming Catastrophic Forgetting in Neural Networks. PNAS 114(13), 2017.
4. Rusu, A. et al. Progressive Neural Networks. arXiv:1606.04671, 2016.
5. Robins, A. Catastrophic Forgetting, Rehearsal and Pseudorehearsal. Connection Science 7(2), 1995.
6. Gal, Y. and Ghahramani, Z. Dropout as a Bayesian Approximation. ICML, 2016.
7. Blundell, C. et al. Weight Uncertainty in Neural Networks. ICML, 2015.
8. McClelland, J., McNaughton, B., and O'Reilly, R. Why There Are Complementary Learning Systems in the Hippocampus and Neocortex. Psychological Review 102(3), 1995.
9. Kendall, A. and Gal, Y. What Uncertainties Do We Need in Bayesian Deep Learning? NeurIPS, 2017.
10. Houthooft, R. et al. VIME: Variational Information Maximizing Exploration. NeurIPS, 2016.
11. Guo, C. et al. On Calibration of Modern Neural Networks. ICML, 2017.
12. Amari, S. Natural Gradient Works Efficiently in Learning. Neural Computation 10(2), 1998.
13. Aljundi, R. et al. Gradient Based Sample Selection for Online Continual Learning. NeurIPS, 2019.
14. Aljundi, R. et al. Online Continual Learning with Maximally Interfered Retrieval. NeurIPS, 2019.
15. Tiwari, R. et al. GCR: Gradient Coreset Based Replay Buffer Selection for Continual Learning. CVPR, 2022.
16. Zhang, J. et al. CORE: Mitigating Catastrophic Forgetting through Cognitive Replay. arXiv:2402.01348, 2024.
17. Smith, J. et al. Adaptive Memory Replay for Continual Learning. arXiv:2404.12526, 2024.
18. Yoo, J. et al. Layerwise Proximal Replay: A Proximal Point Method for Online Continual Learning. ICML, 2024.
19. Harun, M. et al. GRASP: A Rehearsal Policy for Efficient Online Continual Learning. CoLLAs, 2025.
20. Wimmer, L. et al. Quantifying Aleatoric and Epistemic Uncertainty in Machine Learning. UAI, 2023.
