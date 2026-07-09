# Ego Gate

**Models that keep learning shouldn't have to keep forgetting. Ego Gate is a replay buffer selection strategy built on one bet: a model that knows what it doesn't know can decide what's worth remembering — no retraining cycle required.**

[![Status](https://img.shields.io/badge/status-research--preview-orange)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)

---

## Why This Exists

Every large model today learns in one shot, then freezes. New knowledge means a new training run. That's not how intelligence is supposed to work — humans update on the fly, without wiping out everything they knew before.

**Ego Gate is the first experimental piece of [The Stimulus](#the-stimulus) — a research program betting that continual learning, done right, is a load-bearing piece of the path to AGI.** Not a side quest. The actual mechanism.

The idea starts small and concrete: teach a model to select *what to remember* using uncertainty and influence signals, so it can keep learning without catastrophic forgetting — and without a full retrain every time the world changes. Ego Gate is that selection engine.

## What's Inside

- **Doubt** — uncertainty-aware replay selection. The model keeps what it's least sure about.
- **Ego Gate (combined)** — influence-weighted uncertainty selection, layering *how much an example would change the model* on top of how uncertain it is.
- **Router (experimental)** — a mechanism for dynamically switching selection strategy mid-training, the architectural bet this repo is building toward.

Full benchmarks, ablations, and the write-up on what worked and what didn't are in `/paper` and `/experiments`.

## The Stimulus

Ego Gate is step one of a larger thesis: that AGI doesn't emerge from bigger one-shot training runs, but from systems that learn continuously, gate their own memory, and never need to stop and retrain from scratch. Everything in this repo is built to be a real, falsifiable piece of that argument — not a metaphor for it.

## Quick Start

```bash
git clone https://github.com/OptimistOtaku/ego-gate.git
cd ego-gate
pip install -r requirements.txt

# Run the Doubt baseline
python train.py --strategy doubt --dataset <your-benchmark>

# Run the full Ego Gate combined strategy
python train.py --strategy ego-gate --dataset <your-benchmark>
```

## Repo Structure

```
ego-gate/
├── strategies/
│   ├── doubt.py            # Uncertainty-only selection
│   ├── ego_gate.py         # Combined uncertainty + influence selection
│   └── router.py           # Experimental: dynamic strategy routing (untested)
├── benchmarks/             # Evaluation scripts and metrics
├── experiments/            # Configs for reproducing reported results
├── paper/                  # Preprint and supplementary material
└── train.py
```

## Roadmap

- [ ] Validate the dynamic router across longer task sequences
- [ ] Expand benchmark coverage beyond current task suite
- [ ] Publish full comparisons against established continual learning baselines
- [ ] Scale toward the broader Stimulus architecture

## Citation

If you use this work, please cite the preprint (link in `/paper`).

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built by [Aditya](https://optimistotaku.tech) — one piece of The Stimulus, a research program on continual learning as a path to AGI.*
