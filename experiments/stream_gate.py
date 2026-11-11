"""Single-pass supervised admission/consolidation study; no task IDs enter the gate.

Test boundaries are evaluator-only. Run on Kaggle through the generated notebook.
Scores are computed before each block's updates. Comparisons share initialization,
stream, corruption, batch size and optimization-update counts, not total compute.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class Config:
    dataset: str = "digits"
    seed: int = 0
    memory: int = 200
    batch: int = 32
    period: int = 8
    lr: float = 0.05
    hidden: int = 128
    noise: float = 0.0
    epsilon: float = 0.1
    device: str = "cpu"


POLICIES = ("reservoir", "balanced", "doubt", "curiosity", "egogate", "egogate_audit")


class Net(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


@torch.no_grad()
def scores(model, x, y, mode):
    h = F.relu(model.fc1(x))
    p = model.fc2(h).softmax(1)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(1) / np.log(10)
    if mode == "doubt":
        return entropy
    err = p.clone()
    err[torch.arange(len(y), device=y.device), y] -= 1
    back = (err @ model.fc2.weight) * (h > 0)
    grad2 = err.square().sum(1) * (h.square().sum(1) + 1)
    grad2 += back.square().sum(1) * (x.square().sum(1) + 1)
    # Bounded monotone transform avoids full-stream normalization or future data.
    influence = grad2 / (1 + grad2)
    return influence if mode == "curiosity" else (entropy + influence) / 2


class Memory:
    def __init__(self, capacity, policy, seed, epsilon=0.1):
        self.capacity, self.policy, self.epsilon = capacity, policy, epsilon
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.seen = 0
        self.admitted = 0
        self.score_seconds = 0.0
        self.scored_examples = 0

    def add(self, x, y, corrupt, model):
        self.seen += 1
        item = (x.detach().cpu().clone(), int(y), bool(corrupt))
        if len(self.items) < self.capacity:
            self.items.append(item)
            self.admitted += 1
            return True
        if self.policy == "reservoir":
            j = int(self.rng.integers(self.seen))
            if j < self.capacity:
                self.items[j] = item
                self.admitted += 1
                return True
            return False
        if self.policy == "balanced":
            # Online class-balanced random replacement, using observed labels only.
            labels = np.array([v[1] for v in self.items])
            counts = np.bincount(labels, minlength=10)
            largest = np.flatnonzero(counts == counts.max())
            victim_class = int(self.rng.choice(largest))
            if counts[int(y)] <= counts[victim_class]:
                candidates = np.flatnonzero(labels == victim_class)
                self.items[int(self.rng.choice(candidates))] = item
                self.admitted += 1
                return True
            return False
        # Audit bypass keeps some low-score evidence accessible. This is exploratory,
        # not a calibrated poisoning defense or an adaptive/meta-learned controller.
        j = int(self.rng.integers(self.capacity))
        accept = self.policy == "egogate_audit" and self.rng.random() < self.epsilon
        if not accept:
            device = next(model.parameters()).device
            pair_x = torch.stack((item[0], self.items[j][0])).to(device)
            pair_y = torch.tensor((item[1], self.items[j][1]), device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            s = scores(model, pair_x, pair_y, self.policy)
            accept = bool(s[0] > s[1])
            if device.type == "cuda":
                torch.cuda.synchronize()
            self.score_seconds += time.perf_counter() - start
            self.scored_examples += 2
        if accept:
            self.items[j] = item
            self.admitted += 1
        return accept

    def sample(self, size, rng, device):
        idx = rng.integers(len(self.items), size=size)
        return (torch.stack([self.items[i][0] for i in idx]).to(device),
                torch.tensor([self.items[i][1] for i in idx], device=device))


def dataset(cfg, root):
    if cfg.dataset == "digits":
        d = load_digits()
        x, y = d.data.astype("float32") / 16, d.target
        a, b = train_test_split(np.arange(len(y)), test_size=.2,
                                stratify=y, random_state=cfg.seed)
        return x[a], y[a], x[b], y[b]
    from torchvision.datasets import FashionMNIST
    train = FashionMNIST(root, train=True, download=True)
    test = FashionMNIST(root, train=False, download=True)
    return (train.data.numpy().reshape(-1, 784).astype("float32") / 255,
            train.targets.numpy(), test.data.numpy().reshape(-1, 784).astype("float32") / 255,
            test.targets.numpy())


def run_one(cfg, policy, schedule, root, admission=False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x, y, tx, ty = dataset(cfg, root)
    stream_rng = np.random.default_rng(cfg.seed + 1000)
    classes = stream_rng.permutation(10).reshape(5, 2)
    stream = np.concatenate([stream_rng.permutation(np.flatnonzero(np.isin(y, c))) for c in classes])
    ends = np.cumsum([np.isin(y, c).sum() for c in classes])
    clean = y[stream].copy()
    noisy = clean.copy()
    corruption = stream_rng.random(len(noisy)) < cfg.noise
    noisy[corruption] = (noisy[corruption] + stream_rng.integers(1, 10, corruption.sum())) % 10
    x = torch.tensor(x[stream])
    noisy = torch.tensor(noisy, dtype=torch.long)
    model = Net(x.shape[1], cfg.hidden).to(cfg.device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    memory = Memory(cfg.memory, policy, cfg.seed + 2000, cfg.epsilon)
    replay_rng = np.random.default_rng(cfg.seed + 3000)
    order_rng = np.random.default_rng(cfg.seed + 4000)
    pending, trace = [], []
    updates, exposures, peak_pending = 0, 0, 0
    period = 1 if schedule == "immediate" else cfg.period
    start_time = time.perf_counter()

    def consolidate():
        nonlocal updates, exposures
        if not pending:
            return
        # Retain the same number of updates and new examples across schedules.
        for index in order_rng.permutation(len(pending)):
            bx, by, original_count = pending[index]
            rx, ry = memory.sample(cfg.batch + original_count - len(by), replay_rng, cfg.device)
            allx = torch.cat((bx.to(cfg.device), rx))
            ally = torch.cat((by.to(cfg.device), ry))
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(allx), ally).backward()
            opt.step()
            updates += 1
            exposures += len(ally)
        pending.clear()

    # Only the evaluator uses ends; minibatches continue across task transitions.
    evaluated = set()
    for lo in range(0, len(x), cfg.batch):
        hi = min(lo + cfg.batch, len(x))
        admitted = [memory.add(x[i], noisy[i], corruption[i], model) for i in range(lo, hi)]
        mask = torch.tensor(admitted if admission else [True]*(hi-lo), dtype=torch.bool)
        pending.append((x[lo:hi][mask], noisy[lo:hi][mask], hi-lo))
        peak_pending = max(peak_pending, sum(len(v[0]) for v in pending))
        if len(pending) >= period or hi == len(x):
            consolidate()
        for task, end in enumerate(ends):
            if hi >= end and task not in evaluated:
                evaluated.add(task)
                with torch.no_grad():
                    pred = torch.cat([model(torch.tensor(tx[j:j+512], device=cfg.device)).argmax(1).cpu()
                                      for j in range(0, len(tx), 512)]).numpy()
                acc = [float((pred[np.isin(ty,c)] == ty[np.isin(ty,c)]).mean()) for c in classes]
                trace.append({"task":task, "observations":hi, "accuracy":acc,
                              "pending_examples":sum(len(v[0]) for v in pending)})
    matrix = np.asarray([v["accuracy"] for v in trace])
    final = float(matrix[-1].mean())
    forgetting = float(np.mean([matrix[j:-1,j].max() - matrix[-1,j] for j in range(4)]))
    row = {**asdict(cfg), "policy":policy, "schedule":schedule, "admission":admission,
           "final_accuracy":final, "forgetting":forgetting,
           "mean_seen_accuracy":float(np.mean([matrix[j,:j+1].mean() for j in range(5)])),
           "updates":updates, "training_exposures":exposures,
           "admission_fraction":memory.admitted / memory.seen,
           "buffer_corruption":float(np.mean([v[2] for v in memory.items])),
           "peak_pending_examples":peak_pending, "buffer_examples":len(memory.items),
           "score_seconds":memory.score_seconds, "scored_examples":memory.scored_examples,
           "wall_seconds":time.perf_counter()-start_time,
           "class_order":classes.tolist(), "trace":trace}
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["digits", "fashion"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--noise", type=float, nargs="+", default=[0., .2])
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/stream_results"))
    p.add_argument("--data-root", default="/kaggle/working/data")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    # Study is fixed before execution: six selectors + schedule ablation.
    conditions = [(p,"immediate",False) for p in POLICIES] + [
        ("reservoir","periodic",False),("egogate_audit","periodic",False),
        ("egogate_audit","immediate",True),("egogate_audit","periodic",True)]
    manifest = {"python":platform.python_version(), "torch":torch.__version__,
                "numpy":np.__version__, "device":args.device,
                "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                "conditions":conditions}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    results = []
    for name in args.datasets:
        for noise in args.noise:
            for seed in range(args.seeds):
                for policy,schedule,admission in conditions:
                    cfg = Config(dataset=name, seed=seed, noise=noise, device=args.device)
                    row = run_one(cfg,policy,schedule,args.data_root,admission)
                    results.append(row)
                    (args.output/"raw.json").write_text(json.dumps(results,indent=2))
                    print(name,noise,seed,policy,schedule,admission,round(row["final_accuracy"],4),flush=True)


if __name__ == "__main__":
    main()
