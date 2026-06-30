"""Reproducible Split-Digits continual-learning benchmark for Ego Gate.

The benchmark separates *storage selection* from *replay compute*. Every bounded
buffer condition receives the same memory budget and the same number of replay
examples per Task-B update. Results are written per seed before aggregation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return (logits, hidden) if return_hidden else logits


@dataclass(frozen=True)
class Config:
    num_seeds: int = 20
    data_seed: int = 0
    hidden_size: int = 64
    epochs_a: int = 60
    epochs_b: int = 40
    learning_rate: float = 0.5
    batch_size: int = 32
    replay_batch_size: int = 2
    buffer_size: int = 40
    candidate_multiplier: int = 5
    device: str = "cpu"


CONDITIONS = (
    "none",
    "random",
    "stratified_random",
    "doubt",
    "curiosity",
    "egogate",
    "embedding_kcenter",
    "egogate_diverse",
    "full",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_digits(data_seed: int):
    digits = load_digits()
    x = digits.data.astype(np.float32) / 16.0
    y = digits.target.astype(np.int64)

    task_a = y <= 4
    task_b = y >= 5
    xa_train, xa_test, ya_train, ya_test = train_test_split(
        x[task_a], y[task_a], test_size=0.2, random_state=data_seed, stratify=y[task_a]
    )
    xb_train, xb_test, yb_train, yb_test = train_test_split(
        x[task_b], y[task_b], test_size=0.2, random_state=data_seed, stratify=y[task_b]
    )
    to_x = lambda value: torch.from_numpy(value)
    to_y = lambda value: torch.from_numpy(value).long()
    return tuple(
        fn(value)
        for value, fn in (
            (xa_train, to_x),
            (ya_train, to_y),
            (xa_test, to_x),
            (ya_test, to_y),
            (xb_train, to_x),
            (yb_train, to_y),
            (xb_test, to_x),
            (yb_test, to_y),
        )
    )


def iter_batches(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
):
    order = torch.randperm(len(x), generator=generator)
    for start in range(0, len(x), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def train_task_a(model: MLP, x: torch.Tensor, y: torch.Tensor, cfg: Config, seed: int) -> None:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(seed + 10_000)
    for _ in range(cfg.epochs_a):
        for xb, yb in iter_batches(x, y, cfg.batch_size, generator):
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            optimizer.step()


def train_task_b(
    model: MLP,
    x_new: torch.Tensor,
    y_new: torch.Tensor,
    replay_x: torch.Tensor,
    replay_y: torch.Tensor,
    cfg: Config,
    seed: int,
) -> None:
    """Train with fixed replay exposure, independent of buffer population size."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(seed + 20_000)
    for _ in range(cfg.epochs_b):
        for xb, yb in iter_batches(x_new, y_new, cfg.batch_size, generator):
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            train_x, train_y = xb, yb
            if len(replay_x):
                ridx = torch.randint(
                    len(replay_x), (cfg.replay_batch_size,), generator=generator
                )
                rx = replay_x[ridx].to(cfg.device)
                ry = replay_y[ridx].to(cfg.device)
                train_x = torch.cat((xb, rx), dim=0)
                train_y = torch.cat((yb, ry), dim=0)
            loss = F.cross_entropy(model(train_x), train_y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model: MLP, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    model.eval()
    prediction = model(x.to(device)).argmax(dim=1).cpu()
    return float((prediction == y).float().mean())


@torch.no_grad()
def score_examples(model: MLP, x: torch.Tensor, y: torch.Tensor, device: str):
    """Compute entropy and the exact squared per-example gradient norm for this MLP.

    For an outer product ab^T, ||ab^T||_F^2 = ||a||^2 ||b||^2. This lets us
    compute the full parameter-gradient norm without one backward pass per sample.
    """
    model.eval()
    x_device, y_device = x.to(device), y.to(device)
    logits, hidden = model(x_device, return_hidden=True)
    probabilities = logits.softmax(dim=1)
    doubt = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)

    output_error = probabilities.clone()
    output_error[torch.arange(len(y_device), device=device), y_device] -= 1.0
    hidden_error = (output_error @ model.fc2.weight) * (hidden > 0)
    output_error_sq = output_error.square().sum(dim=1)
    hidden_error_sq = hidden_error.square().sum(dim=1)
    curiosity = output_error_sq * (hidden.square().sum(dim=1) + 1.0)
    curiosity += hidden_error_sq * (x_device.square().sum(dim=1) + 1.0)

    def zscore(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-12)

    egogate = zscore(doubt) + zscore(curiosity)
    return {
        "doubt": doubt.cpu().numpy(),
        "curiosity": curiosity.cpu().numpy(),
        "egogate": egogate.cpu().numpy(),
        "embedding": hidden.cpu().numpy(),
    }


def stratified_random_indices(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    classes = np.unique(y)
    base, remainder = divmod(k, len(classes))
    selected: list[int] = []
    for position, label in enumerate(classes):
        count = base + int(position < remainder)
        candidates = np.flatnonzero(y == label)
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def kcenter_indices(features: np.ndarray, k: int, candidates: np.ndarray | None = None) -> np.ndarray:
    pool = np.arange(len(features)) if candidates is None else np.asarray(candidates)
    if k >= len(pool):
        return pool.copy()
    values = features[pool].astype(np.float64, copy=True)
    scale = values.std(axis=0, keepdims=True)
    values = (values - values.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-8)
    first = int(np.square(values).sum(axis=1).argmax())
    chosen = [first]
    min_distance = np.square(values - values[first]).sum(axis=1)
    min_distance[first] = -1.0
    for _ in range(1, k):
        nxt = int(min_distance.argmax())
        chosen.append(nxt)
        distance = np.square(values - values[nxt]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[chosen] = -1.0
    return pool[np.asarray(chosen)]


def select_indices(
    condition: str,
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    cfg: Config,
    seed: int,
) -> np.ndarray:
    n, k = len(y), min(cfg.buffer_size, len(y))
    rng = np.random.default_rng(seed + 30_000)
    if condition == "none":
        return np.empty(0, dtype=np.int64)
    if condition == "full":
        return np.arange(n)
    if condition == "random":
        return rng.choice(n, size=k, replace=False)
    if condition == "stratified_random":
        return stratified_random_indices(y, k, rng)
    if condition in {"doubt", "curiosity", "egogate"}:
        return np.argsort(scores[condition])[-k:][::-1].copy()
    if condition == "embedding_kcenter":
        return kcenter_indices(scores["embedding"], k)
    if condition == "egogate_diverse":
        candidate_count = min(n, cfg.candidate_multiplier * k)
        candidates = np.argsort(scores["egogate"])[-candidate_count:]
        return kcenter_indices(scores["embedding"], k, candidates)
    raise ValueError(f"Unknown condition: {condition}")


def buffer_diagnostics(
    seed: int,
    condition: str,
    indices: np.ndarray,
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "seed": seed,
        "condition": condition,
        "buffer_size": len(indices),
        "label_coverage": len(np.unique(labels[indices])) if len(indices) else 0,
    }
    for label in range(10):
        row[f"class_{label}"] = int(np.sum(labels[indices] == label)) if len(indices) else 0
    for name in ("doubt", "curiosity", "egogate"):
        row[f"mean_{name}"] = float(np.mean(scores[name][indices])) if len(indices) else math.nan
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_existing(output_dir: Path) -> None:
    path = output_dir / "per_seed.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "seed": int(raw["seed"]),
                    "condition": raw["condition"],
                    "task_a_pre": float(raw["task_a_pre"]),
                    "task_a_post": float(raw["task_a_post"]),
                    "forgetting": float(raw["forgetting"]),
                    "task_b_post": float(raw["task_b_post"]),
                }
            )
    conditions = tuple(dict.fromkeys(row["condition"] for row in rows))
    summary, comparisons = aggregate(rows, conditions)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)


def aggregate(rows: list[dict], conditions: tuple[str, ...]):
    summary: list[dict] = []
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        record: dict[str, float | int | str] = {"condition": condition, "n": len(subset)}
        for metric in ("task_a_pre", "task_a_post", "forgetting", "task_b_post"):
            values = np.asarray([row[metric] for row in subset], dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        summary.append(record)

    comparisons: list[dict] = []
    requested_pairs = []
    if "random" in conditions:
        requested_pairs.extend(("random", condition) for condition in conditions if condition != "random")
    requested_pairs.extend(
        pair
        for pair in (
            ("egogate", "doubt"),
            ("egogate", "curiosity"),
            ("egogate", "egogate_diverse"),
        )
        if pair[0] in conditions and pair[1] in conditions
    )
    for baseline_name, condition in requested_pairs:
        baseline = {
            row["seed"]: row["forgetting"]
            for row in rows
            if row["condition"] == baseline_name
        }
        candidate = {
            row["seed"]: row["forgetting"]
            for row in rows
            if row["condition"] == condition
        }
        seeds = sorted(set(baseline) & set(candidate))
        baseline_values = np.asarray([baseline[seed] for seed in seeds])
        condition_values = np.asarray([candidate[seed] for seed in seeds])
        improvement = baseline_values - condition_values
        if len(seeds) > 1:
            test = ttest_rel(baseline_values, condition_values)
            std_delta = improvement.std(ddof=1)
            cohen_dz = improvement.mean() / std_delta if std_delta > 0 else math.inf
        else:
            test = type("Test", (), {"statistic": math.nan, "pvalue": math.nan})()
            cohen_dz = math.nan
        comparisons.append(
            {
                "baseline": baseline_name,
                "condition": condition,
                "n": len(seeds),
                "mean_forgetting_reduction": float(improvement.mean()),
                "relative_reduction": float(improvement.mean() / baseline_values.mean())
                if baseline_values.mean() != 0
                else math.nan,
                "paired_t": float(test.statistic),
                "p_value": float(test.pvalue),
                "cohen_dz": float(cohen_dz),
            }
        )
    # Holm correction controls family-wise error across the declared comparisons.
    order = sorted(range(len(comparisons)), key=lambda index: comparisons[index]["p_value"])
    running_max = 0.0
    for rank, index in enumerate(order):
        adjusted = (len(order) - rank) * comparisons[index]["p_value"]
        running_max = max(running_max, adjusted)
        comparisons[index]["holm_p_value"] = min(1.0, running_max)
    return summary, comparisons


def run(cfg: Config, output_dir: Path, conditions: tuple[str, ...]) -> None:
    data = load_split_digits(cfg.data_seed)
    xa_train, ya_train, xa_test, ya_test, xb_train, yb_train, xb_test, yb_test = data
    rows: list[dict] = []
    diagnostics: list[dict] = []

    for seed in range(cfg.num_seeds):
        seed_everything(seed)
        base_model = MLP(cfg.hidden_size).to(cfg.device)
        train_task_a(base_model, xa_train, ya_train, cfg, seed)
        task_a_pre = accuracy(base_model, xa_test, ya_test, cfg.device)
        scores = score_examples(base_model, xa_train, ya_train, cfg.device)
        labels = ya_train.numpy()

        for condition in conditions:
            indices = select_indices(condition, labels, scores, cfg, seed)
            diagnostics.append(
                buffer_diagnostics(seed, condition, indices, labels, scores)
            )
            model = copy.deepcopy(base_model)
            train_task_b(
                model,
                xb_train,
                yb_train,
                xa_train[indices],
                ya_train[indices],
                cfg,
                seed,
            )
            task_a_post = accuracy(model, xa_test, ya_test, cfg.device)
            task_b_post = accuracy(model, xb_test, yb_test, cfg.device)
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "task_a_pre": task_a_pre,
                    "task_a_post": task_a_post,
                    "forgetting": task_a_pre - task_a_post,
                    "task_b_post": task_b_post,
                }
            )
            print(
                f"seed={seed:02d} condition={condition:20s} "
                f"A_pre={task_a_pre:.3f} A_post={task_a_post:.3f} "
                f"forget={task_a_pre - task_a_post:.3f} B={task_b_post:.3f}",
                flush=True,
            )

        write_csv(output_dir / "per_seed.csv", rows)
        write_csv(output_dir / "buffer_diagnostics.csv", diagnostics)

    summary, comparisons = aggregate(rows, conditions)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "paired_comparisons.csv", comparisons)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "conditions": conditions}, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/digits"))
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--epochs-a", type=int, default=60)
    parser.add_argument("--epochs-b", type=int, default=40)
    parser.add_argument("--buffer-size", type=int, default=40)
    parser.add_argument("--replay-batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Regenerate aggregate CSVs from an existing per_seed.csv without training.",
    )
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        summarize_existing(args.output_dir)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    cfg = Config(
        num_seeds=args.num_seeds,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        buffer_size=args.buffer_size,
        replay_batch_size=args.replay_batch_size,
        device=args.device,
    )
    run(cfg, args.output_dir, tuple(args.conditions))


if __name__ == "__main__":
    main()
