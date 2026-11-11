# Additional benchmark protocol (fixed before new outcomes)

This study strengthens the current paper's **offline storage selection** evidence.
It does not redesign the online controller or replace earlier negative results.
It is a dated local protocol, not an externally registered preregistration.

## Questions and design

Does the original sum of standardized entropy and squared gradient norm improve
retention and final accuracy beyond random, entropy alone, and representative
selection? How does the answer depend on dataset and memory budget?

- Datasets: Digits (stratified 80/20 split per seed), full official Fashion-MNIST
  and CIFAR-10 train/test splits. No leaderboard scores imported as comparisons.
  CIFAR is attached from Kaggle's `pankrzysiu/cifar10-python` mirror; torchvision
  checks the official batch-file MD5 hashes. The public source is a data mirror,
  not a source of baseline performance numbers.
- Ten paired seeds 100--109. Each seed randomizes the five-class/five-class
  partition, initialization, training order, and random selection. No task label
  or class mask is supplied at inference; predictions always use ten outputs.
- Digits retains the original 64-hidden-unit MLP, SGD 0.5, 60/40 epochs,
  batches of 32 new and 2 replay examples. Memory sizes 20, 40, 100.
- Fashion uses a 128-hidden-unit MLP, SGD 0.1, 10/10 epochs, batches of 128
  new and 8 replay examples. Memory sizes 50, 200, 500.
- CIFAR uses a three-block convolutional network without batch normalization,
  SGD 0.05 with momentum 0.9, 20/20 epochs, batches of 128 new and 8 replay
  examples, normalized pixels, no augmentation. Memory sizes 50, 200, 500.
  This is a transparent controlled CNN experiment, not an optimized CIFAR entry.
- Fixed configurations, no test-driven tuning or winner-only reporting. Any
  amendment must be recorded explicitly before additional runs.
- Nine bounded selectors: random, class-balanced random, entropy, gradient
  norm, original EgoGate, embedding k-center, EgoGate-diverse (existing rule),
  high training loss, and class-balanced herding. Herding is an adaptation of
  the exemplar component of iCaRL, not the complete iCaRL learner. Full-pool
  and no-replay anchors are run once per seed, separately from bounded methods.
- Base Task-A checkpoints are shared within a seed. Task-B training order and
  replay-position random streams are independent and shared across selectors.
  This removes the archived code's coupling between shuffling and replay RNG.
- Score normalization is global over the Task-A training pool, as in the
  original offline rule. CNN gradient norms use exact per-example autograd.
- Equal update and replay exposure budgets, plus actual selection and training
  wall times, persistent exemplar bytes, CUDA peak allocations, selected-class
  counts, and hashes of selected indices. Timing includes synchronization.
  Selection may access the entire training pool: this is not bounded online RAM.

## Inference

Primary endpoint: final equally weighted mean of Task-A and Task-B accuracy.
Primary contrasts: EgoGate minus random, entropy, and herding in each of the
nine dataset/budget cells (27 tests, a single Holm family). Report seed-paired
95% t intervals, all raw values, and sample standard deviations. Intervals are
pointwise, not multiplicity-adjusted. Seeds vary several nuisance factors and
are not independent datasets. Forgetting, new-task accuracy, other selectors,
cost, and coverage are secondary/descriptive. No significance-based stopping.

Expected bounded rows: 3 datasets x 3 budgets x 10 seeds x 9 selectors = 810.
Expected anchor rows: 3 datasets x 10 seeds x 2 = 60. Total: 870.

GPU execution must be verified in the environment manifest. Tests and a
non-inferential execution check can use synthetic inputs locally. Genuine
benchmark evidence must come from Kaggle. Preserve source hashes and raw rows;
never label a partial run complete.

## Limits even after completion

This does not establish state of the art, general AGI progress, efficacy in
agent memory, or superiority to complete GSS/GCR/MIR/GRASP systems. The added
published-component baseline is herding. Multi-phase state-of-the-art systems,
architecture sweeps, and an independently reviewed release remain further work.
