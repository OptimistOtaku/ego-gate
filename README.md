# Ego Gate

Research on selective replay, memory admission, and periodic consolidation.

**[Read the IEEE paper](output/pdf/ego_gate_ieee.pdf)** · [LaTeX source](paper/ieee/main.tex) · [Research assessment](paper/ieee/RESEARCH_REVIEW.md)

This is an author-review manuscript, not a peer-reviewed publication. The small offline Digits result reproduced on Kaggle, while the tested online extension failed to generalize its advantage. Both findings are retained.

An additional **870-run GPU benchmark** on Digits, Fashion-MNIST, and CIFAR-10 has completed and passed its grid/exposure audit. Its results are included; integration into the PDF is still pending. These experiments do not establish state of the art or a route to AGI.

## Quick start

Install Python 3.10+ and a compatible PyTorch/torchvision pair for your hardware.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.run_digits --num-seeds 20 --output-dir results/local/digits
```

## Reproduce the analyses

```bash
python scripts/analyze_stream.py results/kaggle_ieee/ego_gate/stream_results
python scripts/analyze_offline_benchmark.py results/kaggle_gpu/benchmark --output results/offline_benchmark
python scripts/prepare_ieee_assets.py
python scripts/prepare_offline_assets.py results/offline_benchmark
```

Build the paper with Tectonic:

```bash
tectonic --outdir output/pdf paper/ieee/main.tex
```

The compiler names its output `main.pdf`; the reviewed release is `output/pdf/ego_gate_ieee.pdf`. Review rendered pages before replacing that release.

## GPU execution

Read the [fixed protocol](paper/ieee/BENCHMARK_PROTOCOL.md). Run `python scripts/prepare_gpu_benchmark.py`, set your own notebook ID in the generated metadata, then use `kaggle kernels push -p kaggle/gpu_benchmark --accelerator NvidiaTeslaT4`. The generator defaults to the author's private notebook. Keep credentials outside this repository.

## File tree

```text
experiments/                 Experiment implementations
scripts/                     Analysis and notebook-packaging tools
tests/                       Mathematical and evidence-integrity checks
kaggle/                      Execution notebooks and source hashes
paper/ieee/                  Canonical manuscript, figures, and protocol
output/pdf/                  Current IEEE paper
results/digits/              Original observations cited by the paper
results/kaggle_ieee/         Reproduction and online-study evidence
results/kaggle_gpu/          Expanded GPU benchmark evidence
results/offline_benchmark/   Audited summaries and comparisons
```

Historical observations and frozen execution sources remain because they substantiate the paper. Superseded manuscripts are available in Git history. Temporary files, credentials, local drafts, and downloads are excluded.

## License and citation

[MIT](LICENSE). Author: Aditya Singh. Cite the manuscript version or commit. IEEE formatting does not imply IEEE publication.
