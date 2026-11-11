"""Fixed-protocol, GPU offline selection benchmark; see BENCHMARK_PROTOCOL.md."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call, grad, vmap
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

METHODS = ('random', 'stratified_random', 'doubt', 'curiosity', 'egogate',
           'embedding_kcenter', 'egogate_diverse', 'loss', 'herding')


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, 10)

    def forward(self, x, features=False):
        h = F.relu(self.fc1(x.flatten(1)))
        return (self.fc2(h), h) if features else self.fc2(h)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1, self.fc2 = nn.Linear(128 * 4 * 4, 128), nn.Linear(128, 10)

    def forward(self, x, features=False):
        h = F.relu(self.fc1(self.conv(x).flatten(1)))
        return (self.fc2(h), h) if features else self.fc2(h)


def settings(dataset):
    if dataset == 'digits':
        return dict(lr=.5, momentum=0., epochs_a=60, epochs_b=40, batch=32,
                    replay=2, budgets=[20, 40, 100])
    return dict(lr=.05 if dataset == 'cifar10' else .1,
                momentum=.9 if dataset == 'cifar10' else 0.,
                epochs_a=20 if dataset == 'cifar10' else 10,
                epochs_b=20 if dataset == 'cifar10' else 10,
                batch=128, replay=8, budgets=[50, 200, 500])


def load_data(name, root, seed):
    if name == 'digits':
        d = load_digits()
        ix, it = train_test_split(np.arange(len(d.target)), test_size=.2,
                                 stratify=d.target, random_state=seed)
        return (torch.tensor(d.data[ix] / 16, dtype=torch.float32), torch.tensor(d.target[ix]),
                torch.tensor(d.data[it] / 16, dtype=torch.float32), torch.tensor(d.target[it]))
    from torchvision.datasets import FashionMNIST, CIFAR10
    cls = CIFAR10 if name == 'cifar10' else FashionMNIST
    download = True
    if name == 'cifar10' and Path('/kaggle/input').exists():
        mirrors = sorted(Path('/kaggle/input').glob('**/cifar-10-batches-py'))
        if mirrors:
            # torchvision validates the official per-batch MD5 checksums.
            root, download = mirrors[0].parent, False
    train, test = cls(root, train=True, download=download), cls(root, train=False, download=download)
    def convert(d):
        x = torch.as_tensor(np.asarray(d.data).copy(), dtype=torch.float32) / 255
        if name == 'cifar10':
            x = x.permute(0, 3, 1, 2).contiguous()
            x = (x - torch.tensor([.4914, .4822, .4465])[None, :, None, None]) / torch.tensor([.247, .243, .261])[None, :, None, None]
        return x, torch.as_tensor(d.targets, dtype=torch.long)
    return (*convert(train), *convert(test))


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def train(model, x, y, cfg, seed, epochs, rx=None, ry=None):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])
    # Separate generators prevent replay policy from altering new-example order.
    order_rng = torch.Generator().manual_seed(seed + 20000)
    replay_rng = torch.Generator().manual_seed(seed + 30000)
    updates = exposures = replay_exposures = 0
    for epoch in range(epochs):
        order = torch.randperm(len(y), generator=order_rng)
        for start in range(0, len(y), cfg['batch']):
            ix = order[start:start + cfg['batch']].to(y.device)
            bx, by = x[ix], y[ix]
            if rx is not None and len(rx):
                ri = torch.randint(len(rx), (cfg['replay'],), generator=replay_rng).to(y.device)
                bx, by = torch.cat((bx, rx[ri])), torch.cat((by, ry[ri]))
                replay_exposures += len(ri)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(bx), by).backward()
            opt.step()
            updates += 1
            exposures += len(by)
    return dict(updates=updates, exposures=exposures, replay_exposures=replay_exposures)


@torch.no_grad()
def accuracy(model, x, y):
    model.eval()
    correct = torch.zeros((), device=y.device)
    for start in range(0, len(y), 512):
        correct += (model(x[start:start+512]).argmax(1) == y[start:start+512]).sum()
    return float(correct.item() / len(y))


def score_pool(model, x, y, mode):
    """Only compute signals needed by the selector, charging their actual cost."""
    model.eval()
    want_grad = mode in ('curiosity', 'egogate', 'egogate_diverse')
    want_embed = mode in ('embedding_kcenter', 'egogate_diverse', 'herding')
    parts = {k: [] for k in ('doubt', 'curiosity', 'embedding', 'loss')}
    params = {k: v.detach() for k, v in model.named_parameters()}
    def single_loss(p, sample, label):
        return F.cross_entropy(functional_call(model, p, (sample.unsqueeze(0),)), label.unsqueeze(0))
    per_grad = vmap(grad(single_loss), in_dims=(None, 0, 0))
    batch = 16 if isinstance(model, CNN) and want_grad else 512
    for start in range(0, len(y), batch):
        bx, by = x[start:start+batch], y[start:start+batch]
        with torch.no_grad():
            logits, h = model(bx, features=True)
            p = logits.softmax(1)
            parts['doubt'].append((-(p * p.clamp_min(1e-12).log()).sum(1)).cpu())
            parts['loss'].append(F.cross_entropy(logits, by, reduction='none').cpu())
            if want_embed:
                parts['embedding'].append(h.cpu())
            if want_grad and isinstance(model, MLP):
                err = p.clone()
                err[torch.arange(len(by), device=by.device), by] -= 1
                back = (err @ model.fc2.weight) * (h > 0)
                g2 = err.square().sum(1) * (h.square().sum(1) + 1)
                g2 += back.square().sum(1) * (bx.flatten(1).square().sum(1) + 1)
                parts['curiosity'].append(g2.cpu())
        if want_grad and isinstance(model, CNN):
            gradients = per_grad(params, bx, by)
            norm = sum(v.flatten(1).square().sum(1) for v in gradients.values())
            parts['curiosity'].append(norm.detach().cpu())
            del gradients
    out = {k: torch.cat(v).numpy() for k, v in parts.items() if v}
    if want_grad:
        def z(v):
            return (v - v.mean()) / max(float(v.std()), 1e-12)
        out['egogate'] = z(out['doubt']) + z(out['curiosity'])
    return out


def kcenter(features, k, candidates=None):
    pool = np.arange(len(features)) if candidates is None else candidates
    z = features[pool].astype(np.float64)
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-8)
    chosen = [int((z*z).sum(1).argmax())]
    distance = ((z - z[chosen[0]])**2).sum(1)
    distance[chosen] = -1
    for _ in range(1, k):
        idx = int(distance.argmax())
        chosen.append(idx)
        distance = np.minimum(distance, ((z-z[idx])**2).sum(1))
        distance[chosen] = -1
    return pool[np.asarray(chosen)]


def herding(features, labels, k):
    """Balanced herding adaptation: L2-normalized features, no duplicate exemplars.

    Greedily minimize distance of the selected arithmetic mean to the class mean.
    This is a component baseline, not full iCaRL or its nearest-mean classifier.
    """
    result = []
    classes = np.unique(labels)
    for pos, label in enumerate(classes):
        pool = np.flatnonzero(labels == label)
        z = features[pool].astype(np.float64)
        z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        target, running = z.mean(0), np.zeros(z.shape[1])
        used = np.zeros(len(pool), dtype=bool)
        count = k // len(classes) + int(pos < k % len(classes))
        for i in range(count):
            distances = ((target - (z + running)/(i+1))**2).sum(1)
            distances[used] = np.inf
            nxt = int(distances.argmin())
            used[nxt] = True
            running += z[nxt]
            result.append(pool[nxt])
    return np.asarray(result, dtype=np.int64)


def select(model, x, y, method, k, seed):
    rng = np.random.default_rng(seed + 40000)
    labels = y.cpu().numpy()
    if method == 'none':
        return np.empty(0, dtype=np.int64)
    if method == 'full':
        return np.arange(len(y))
    if method == 'random':
        return rng.choice(len(y), k, replace=False)
    if method == 'stratified_random':
        classes = np.unique(labels)
        return np.concatenate([rng.choice(np.flatnonzero(labels == c),
                               k//len(classes) + int(i < k % len(classes)), replace=False)
                               for i, c in enumerate(classes)])
    scores = score_pool(model, x, y, method)
    if method == 'herding':
        return herding(scores['embedding'], labels, k)
    if method == 'embedding_kcenter':
        return kcenter(scores['embedding'], k)
    if method == 'egogate_diverse':
        candidates = np.argsort(scores['egogate'], kind='stable')[-min(len(y), 5*k):]
        return kcenter(scores['embedding'], k, candidates)
    return np.argsort(scores[method], kind='stable')[-k:][::-1].copy()


def run_seed(dataset, seed, output, root, device, smoke=False):
    cfg = settings(dataset)
    if smoke:
        cfg.update(epochs_a=1, epochs_b=1, budgets=[10])
    # Never silently skip a partial seed on resume.
    path = output / f'{dataset}_{seed}.json'
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing evidence: {path}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x, y, tx, ty = load_data(dataset, root, seed)
    classes = np.random.default_rng(seed + 50000).permutation(10)
    a = torch.isin(y, torch.tensor(classes[:5]))
    ta = torch.isin(ty, torch.tensor(classes[:5]))
    arrays = [x[a], y[a], x[~a], y[~a], tx[ta], ty[ta], tx[~ta], ty[~ta]]
    if smoke:
        arrays = [v[:256] for v in arrays]
    xa, ya, xb, yb, xta, yta, xtb, ytb = [v.to(device) for v in arrays]
    del x, y, tx, ty, arrays
    model = (CNN() if dataset == 'cifar10' else MLP(xa[0].numel(), 64 if dataset == 'digits' else 128)).to(device)
    sync(device)
    start = time.perf_counter()
    train(model, xa, ya, cfg, seed, cfg['epochs_a'])
    sync(device)
    base_seconds = time.perf_counter() - start
    before = accuracy(model, xta, yta)
    rows = []
    jobs = [(budget, method) for budget in cfg['budgets'] for method in METHODS]
    jobs += [(0, 'none'), (len(ya), 'full')]
    # Randomized condition execution order reduces systematic timing-order bias.
    np.random.default_rng(seed + 60000).shuffle(jobs)
    for budget, method in jobs:
        candidate = copy.deepcopy(model)
        sync(device)
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        idx = select(model, xa, ya, method, budget, seed)
        sync(device)
        selection_seconds = time.perf_counter() - start
        assert len(idx) == budget and len(np.unique(idx)) == budget
        selected = torch.as_tensor(idx, device=device)
        rx, ry = xa[selected], ya[selected]
        sync(device)
        start = time.perf_counter()
        counts = train(candidate, xb, yb, cfg, seed, cfg['epochs_b'], rx, ry)
        sync(device)
        training_seconds = time.perf_counter() - start
        after, new = accuracy(candidate, xta, yta), accuracy(candidate, xtb, ytb)
        row = dict(dataset=dataset, seed=seed, budget=budget, method=method,
                   train_a_count=len(ya), train_b_count=len(yb),
                   test_a_count=len(yta), test_b_count=len(ytb),
                   parameter_count=sum(p.numel() for p in model.parameters()),
                   task_a_pre=before, task_a_post=after, task_b_post=new,
                   final_accuracy=(after+new)/2, forgetting=before-after,
                   selection_seconds=selection_seconds, training_seconds=training_seconds,
                   base_seconds=base_seconds, memory_bytes=rx.numel()*rx.element_size()+ry.numel()*ry.element_size(),
                   cuda_peak_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
                   coverage=int(len(torch.unique(ry))), class_counts=torch.bincount(ry, minlength=10).cpu().tolist(),
                   selection_sha256=hashlib.sha256(np.asarray(idx, dtype='<i8').tobytes()).hexdigest(),
                   selected_indices=idx.tolist(), **counts)
        rows.append(row)
        (output / f'{dataset}_{seed}.partial.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps({k: row[k] for k in ('dataset','seed','budget','method','final_accuracy','selection_seconds','training_seconds')}), flush=True)
        del candidate, rx, ry
    path.write_text(json.dumps(dict(config=cfg, classes=classes.tolist(), seed=seed,
                                    dataset=dataset, rows=rows, complete=True), indent=2))
    (output / f'{dataset}_{seed}.partial.json').unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['digits', 'fashion', 'cifar10'], required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(100,110)))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('/kaggle/working/data'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested Kaggle GPU is unavailable; no silent CPU fallback')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    import torchvision
    import scipy
    import sklearn
    manifest = dict(python=platform.python_version(), torch=torch.__version__,
                    torchvision=torchvision.__version__, numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__,
                    cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    device=args.device, gpu=torch.cuda.get_device_name(args.device) if torch.cuda.is_available() else None,
                    platform=platform.platform(), cpu=platform.processor(), threads=torch.get_num_threads(),
                    gpu_memory_bytes=torch.cuda.get_device_properties(args.device).total_memory if torch.cuda.is_available() else None,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    cifar_mirror='pankrzysiu/cifar10-python; torchvision official MD5 validation',
                    seeds=args.seeds, dataset=args.dataset, smoke=args.smoke,
                    kaggle=bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE')))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for seed in args.seeds:
        run_seed(args.dataset, seed, args.output, args.data, args.device, args.smoke)
    (args.output/'COMPLETE.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
