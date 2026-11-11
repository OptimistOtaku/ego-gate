# Research review after Kaggle execution

The 20-seed independent reproduction supports the original small offline replay result (40.1% reduction in forgetting versus random storage). The 200-run streaming study does not support the tested online extension. These are different implementations and settings, so the negative result does not falsify the original offline rule or establish that no useful admission controller exists.

## Main technical findings

- The gradient norm is exact for this MLP, but is a supervised, parameterization-dependent proxy, not Bayesian information gain.
- Constant-probability bypass replacement has a retention upper bound (1-epsilon/M)^n. At the tested values, an existing item survives one 12,000-arrival phase with probability at most 0.0025. A protected reservoir or separate exploratory store should be evaluated next.
- Periodic processing changes order and the available replay pool as well as delay. It is not a pure timing intervention.
- All 200 condition-seed combinations are present and share update/exposure budgets within each dataset/noise/seed group. Total costs and temporary storage differ.
- None of the eight specified primary contrasts rejects the null after Holm correction. Five seeds are inadequate to infer equivalence.
- Noise flags are diagnostic metadata only; the controller never uses them. Final test outcomes do not feed admission decisions.
- Source hashes match the submitted implementation. CPU model and torchvision version were not captured. Archived local outputs lack an original environment manifest.

## Publication assessment

The manuscript is a complete IEEE-format empirical framework paper with a reproduced component result and a documented failed online extension. It is ready for author/colleague review, not a claim of acceptance or validation of the full vision. Stronger published baselines, validation-controlled hyperparameters, isolated scheduler ablations, and larger seed counts would strengthen a competitive submission. The commercial agent-memory application and developmental/personality hypothesis remain untested.

The final evidence is in results/kaggle_ieee/ego_gate/. No experiment was changed after inspecting its outcomes. The next design should be evaluated in a separate, clearly labeled study, preserving these negative results.
