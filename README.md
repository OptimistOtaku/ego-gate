# Ego Gate

Ego Gate studies stimulus valuation and offline memory consolidation for continual-learning systems. The long-term objective is an architecture that admits, buffers, quarantines, or discards real-world inputs before controlled consolidation.

## Current evidence

The included Split-Digits experiment tests one narrow component of that vision: whether information-based selection produces a better fixed-size replay buffer than random storage. It does **not** validate the complete online admission gate or AGI-scale claims.

The current result and staged path to a full thesis are documented in [THESIS_ROADMAP.md](THESIS_ROADMAP.md).

The publication source is `paper/ego_gate_paper_source.md`. Regenerate the paper and its data-driven figure with:

```powershell
python scripts/build_paper.py
```

The build updates both `output/pdf/ego_gate_paper.pdf` and the stable root-level `ego_gate_paper.pdf`.

The experiment runner improves the original study in four ways:

- Every bounded condition receives the same memory and replay-compute budget.
- Results are saved per seed before aggregation.
- Paired tests include effect sizes and relative forgetting reduction.
- Stronger controls test class balance, representation diversity, and a diversity-aware EgoGate buffer.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.run_digits --num-seeds 20 --output-dir results/digits
```

Regenerate summary statistics from the committed seed-level results without retraining:

```powershell
python -m experiments.run_digits --summarize-only --output-dir results/digits
```

For a quick validation run:

```powershell
python -m experiments.run_digits --num-seeds 2 --epochs-a 5 --epochs-b 5 --output-dir results/smoke
```

The benchmark defaults to CPU because the dataset and network are too small to benefit materially from GPU execution. `--device cuda` is available for compatibility checks.

By default, each 32-example Task-B batch receives two replay examples and a single loss is averaged over the combined batch. This produces approximately 40-46 replay exposures per epoch, closely matching one pass over the paper's 40-example buffer while keeping replay compute identical across selectors, including the full-memory oracle.

## Outputs

- `per_seed.csv`: the auditable unit of evidence.
- `summary.csv`: mean and sample standard deviation by condition.
- `paired_comparisons.csv`: paired t-tests and Cohen's dz against random storage.
- `buffer_diagnostics.csv`: class coverage and selected-score diagnostics.
- `config.json`: the exact experiment configuration.

The `results/` directory is intentionally not ignored: completed thesis experiments should be committed with their configurations and raw per-seed metrics.
