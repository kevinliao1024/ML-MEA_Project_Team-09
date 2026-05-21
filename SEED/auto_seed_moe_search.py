import argparse
import copy
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from course_project.TEST_DATASET import TestDataset, TrainDataset


DATA_NAME = "SEED"
DEFAULT_DATA_DIR = Path("G:/MLproject/course_project") / DATA_NAME
CHANNELS = 62
CLASSES = 3
BASELINE_VAL_ACC = 0.5571
DEFAULT_SEEDS = (3407, 2024, 42, 777, 1001)
REQUIRED_METHOD_KINDS = {"single", "multi_seed", "topk", "augment", "ema", "swa"}
KIND_PRIORITY = {
    "single": 0,
    "topk": 1,
    "multi_seed": 2,
    "augment": 3,
    "ema": 4,
    "swa": 5,
}


@dataclass
class Candidate:
    name: str
    kind: str
    config: dict
    seeds: tuple[int, ...] = (3407,)
    augmentation: dict | None = None
    top_k: int = 1
    use_ema: bool = False
    use_swa: bool = False


@dataclass
class SearchStopper:
    baseline: float = BASELINE_VAL_ACC
    min_gain: float = 1 / 350
    patience: int = 10
    best_acc: float = field(init=False)
    no_improve_count: int = 0

    def __post_init__(self):
        self.best_acc = float(self.baseline)

    def observe(self, val_acc):
        val_acc = float(val_acc)
        if val_acc >= self.best_acc + self.min_gain:
            self.best_acc = val_acc
            self.no_improve_count = 0
            return False
        self.no_improve_count += 1
        return self.no_improve_count >= self.patience


class SimpleMoEClassifier(nn.Module):
    def __init__(self, chans=CHANNELS, num_classes=CLASSES, hidden_dim=72, num_experts=4, dropout=0.30):
        super().__init__()
        kernels = (3, 7, 15, 31)[:num_experts]
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(chans, hidden_dim, kernel_size=kernel, padding=kernel // 2, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
                    nn.BatchNorm1d(hidden_dim),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(hidden_dim, num_classes),
                )
                for kernel in kernels
            ]
        )
        self.router = nn.Sequential(
            nn.Linear(chans * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.experts)),
        )

    def forward(self, x):
        route_logits = self.router(eeg_stat_features(x))
        route_weights = torch.softmax(route_logits, dim=1)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(route_weights.unsqueeze(-1) * expert_logits, dim=1)

    def clip_gradients(self, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


class ExponentialMovingAverage:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if torch.is_floating_point(param)
        }

    def update(self, model):
        state = model.state_dict()
        for name, shadow_value in self.shadow.items():
            shadow_value.mul_(self.decay).add_(state[name].detach(), alpha=1.0 - self.decay)

    def state_dict(self, model):
        averaged = copy.deepcopy(model.state_dict())
        for name, value in self.shadow.items():
            averaged[name] = value.detach().clone()
        return averaged


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_h5_y(path):
    with h5py.File(path, "r") as f:
        return f["y"][()].astype(np.int64)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def eeg_stat_features(x, eps=1e-6):
    mean = x.mean(dim=-1)
    std = x.std(dim=-1, unbiased=False)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1) + eps)
    peak_to_peak = x.amax(dim=-1) - x.amin(dim=-1)
    return torch.cat([mean, std, rms, peak_to_peak], dim=1)


def write_labels(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{int(label)}\n")


def baseline_config():
    return {
        "hidden_dim": 72,
        "num_experts": 4,
        "dropout": 0.30,
        "lr": 6e-4,
        "weight_decay": 1e-3,
        "label_smoothing": 0.05,
        "scheduler_patience": 6,
    }


def fine_grid_configs():
    base = baseline_config()
    specs = [
        ("single_h72_dp30_base", 72, 0.30, 6e-4, 1e-3, 0.05, 6),
        ("single_h68_dp30", 68, 0.30, 6e-4, 1e-3, 0.05, 6),
        ("single_h76_dp30", 76, 0.30, 6e-4, 1e-3, 0.05, 6),
        ("single_h72_dp28", 72, 0.28, 6e-4, 1e-3, 0.05, 6),
        ("single_h72_dp32", 72, 0.32, 6e-4, 1e-3, 0.05, 6),
        ("single_h72_lr55", 72, 0.30, 5.5e-4, 1e-3, 0.05, 6),
        ("single_h72_lr65", 72, 0.30, 6.5e-4, 1e-3, 0.05, 6),
        ("single_h72_wd08_ls04", 72, 0.30, 6e-4, 8e-4, 0.04, 6),
        ("single_h72_wd12_ls06", 72, 0.30, 6e-4, 1.2e-3, 0.06, 6),
    ]
    configs = []
    for name, hidden_dim, dropout, lr, weight_decay, label_smoothing, scheduler_patience in specs:
        config = dict(base)
        config.update(
            {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "lr": lr,
                "weight_decay": weight_decay,
                "label_smoothing": label_smoothing,
                "scheduler_patience": scheduler_patience,
            }
        )
        configs.append(Candidate(name=name, kind="single", config=config))
    return configs


def build_candidate_plan(max_candidates=30):
    candidates = fine_grid_configs()
    base = baseline_config()
    for name, augmentation in [
        ("augment_noise", {"noise_std": 0.015}),
        ("augment_shift", {"max_shift": 12}),
        ("augment_channel_dropout", {"channel_drop_prob": 0.08}),
        ("augment_combo", {"noise_std": 0.01, "max_shift": 10, "channel_drop_prob": 0.06}),
    ]:
        candidates.append(Candidate(name=name, kind="augment", config=dict(base), augmentation=augmentation))
    candidates.append(Candidate("topk_h72_dp30", "topk", dict(base), top_k=5))
    candidates.append(Candidate("multi_seed_h72_dp30", "multi_seed", dict(base), seeds=DEFAULT_SEEDS))
    candidates.append(Candidate("ema_h72_dp30", "ema", dict(base), use_ema=True))
    candidates.append(Candidate("swa_h72_dp30", "swa", dict(base), use_swa=True, top_k=5))
    return candidates[:max_candidates]


def should_allow_early_stop(seen_kinds, required_kinds=REQUIRED_METHOD_KINDS):
    return set(required_kinds).issubset(set(seen_kinds))


def sort_results(results):
    return sorted(results, key=lambda result: float(result["val_acc"]), reverse=True)


def select_final_result(results):
    if not results:
        raise ValueError("No results to select from.")
    return min(
        results,
        key=lambda result: (
            -float(result["val_acc"]),
            KIND_PRIORITY.get(result.get("kind", "single"), 99),
            result.get("name", ""),
        ),
    )


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def make_results_payload(results, best, baseline_acc=BASELINE_VAL_ACC):
    json_results = [to_jsonable(result) for result in results]
    json_best = to_jsonable(best)
    return {
        "baseline_val_acc": float(baseline_acc),
        "exceeded_baseline": float(best["val_acc"]) >= baseline_acc + (1 / 350),
        "best": json_best,
        "results": json_results,
    }


def make_loaders(data_dir, batch_size, test_batch_size, seed, use_cuda):
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"
    test_path = data_dir / "test_x_only.h5"
    train_ds = TrainDataset(str(train_path))
    val_ds = TrainDataset(str(val_path))
    test_ds = TestDataset(str(test_path))
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
        pin_memory=use_cuda,
    )
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, pin_memory=use_cuda)
    labels = (read_h5_y(train_path), read_h5_y(val_path))
    return (train_loader, train_eval_loader, val_loader, test_loader), labels, test_ds


def apply_training_augmentation(data, augmentation):
    if not augmentation:
        return data
    augmented = data
    noise_std = augmentation.get("noise_std", 0.0)
    if noise_std:
        augmented = augmented + torch.randn_like(augmented) * float(noise_std)
    max_shift = int(augmentation.get("max_shift", 0) or 0)
    if max_shift:
        shift = random.randint(-max_shift, max_shift)
        augmented = torch.roll(augmented, shifts=shift, dims=-1)
    channel_drop_prob = float(augmentation.get("channel_drop_prob", 0.0) or 0.0)
    if channel_drop_prob:
        mask = torch.rand(augmented.shape[0], augmented.shape[1], 1, device=augmented.device) > channel_drop_prob
        augmented = augmented * mask
    return augmented


def collect_prob(model, loader, device, use_cuda):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            data = batch[0] if isinstance(batch, (tuple, list)) else batch
            data = data.to(device, dtype=torch.float32, non_blocking=use_cuda)
            probs.append(torch.softmax(model(data), dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def average_prob(prob_list):
    return np.mean(np.stack(prob_list, axis=0), axis=0)


def train_single_seed(candidate, args, loaders, labels, test_ds, device, use_cuda, seed, suffix):
    train_loader, train_eval_loader, val_loader, test_loader = loaders
    train_y, val_y = labels
    set_global_seed(seed)
    config = candidate.config
    model = SimpleMoEClassifier(
        hidden_dim=config["hidden_dim"],
        num_experts=config["num_experts"],
        dropout=config["dropout"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=config["scheduler_patience"],
        min_lr=args.min_lr,
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay) if candidate.use_ema else None
    best_state = None
    top_states = []
    best_val_acc = -1.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data, label in train_loader:
            data = data.to(device, dtype=torch.float32, non_blocking=use_cuda)
            label = label.to(device, dtype=torch.long, non_blocking=use_cuda)
            data = apply_training_augmentation(data, candidate.augmentation)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            model.clip_gradients(args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        model.eval()
        val_correct = val_num = 0
        with torch.no_grad():
            for val_data, val_label in val_loader:
                val_data = val_data.to(device, dtype=torch.float32, non_blocking=use_cuda)
                val_label = val_label.to(device, dtype=torch.long, non_blocking=use_cuda)
                val_output = model(val_data)
                val_correct += (torch.argmax(val_output, dim=1) == val_label).sum().item()
                val_num += val_label.size(0)
        epoch_val_acc = val_correct / val_num
        scheduler.step(epoch_val_acc)

        current_state = ema.state_dict(model) if ema is not None else copy.deepcopy(model.state_dict())
        if candidate.top_k > 1 or candidate.use_swa:
            top_states.append((epoch_val_acc, epoch, current_state))
            top_states = sorted(top_states, key=lambda item: item[0], reverse=True)[: candidate.top_k]
        if epoch_val_acc > best_val_acc + args.min_delta:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            bad_epochs = 0
            best_state = current_state
        else:
            bad_epochs += 1
        print(
            f"{candidate.name} seed={seed} epoch={epoch:03d}/{args.epochs} "
            f"val_acc={epoch_val_acc:.4f} best={best_val_acc:.4f}@{best_epoch:03d} "
            f"bad_epochs={bad_epochs}",
            flush=True,
        )
        if bad_epochs >= args.patience:
            break

    checkpoint_dir = Path(args.data_dir) / "auto_moe_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{candidate.name}_{suffix}.pt"
    if candidate.top_k > 1 or candidate.use_swa:
        states = [state for _, _, state in top_states]
        averaged = average_state_dicts(states)
        model.load_state_dict(averaged)
        torch.save(averaged, checkpoint_path)
    else:
        model.load_state_dict(best_state)
        torch.save(best_state, checkpoint_path)

    train_prob = collect_prob(model, train_eval_loader, device, use_cuda)
    val_prob = collect_prob(model, val_loader, device, use_cuda)
    test_prob = collect_prob(model, test_loader, device, use_cuda)
    checkpoint_labels = np.argmax(test_prob, axis=1).astype(int).tolist()
    assert len(checkpoint_labels) == len(test_ds), f"Prediction count {len(checkpoint_labels)} != test sample count {len(test_ds)}"
    prediction_path = checkpoint_path.with_suffix(".txt")
    write_labels(str(prediction_path), checkpoint_labels)
    return {
        "seed": seed,
        "train_prob": train_prob,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "train_acc": accuracy(train_y, np.argmax(train_prob, axis=1)),
        "val_acc": accuracy(val_y, np.argmax(val_prob, axis=1)),
        "best_epoch": best_epoch,
        "checkpoint": checkpoint_path,
        "prediction_path": prediction_path,
    }


def average_state_dicts(states):
    if not states:
        raise ValueError("No states to average.")
    averaged = copy.deepcopy(states[0])
    for key in averaged:
        if not torch.is_floating_point(averaged[key]):
            continue
        averaged[key] = torch.stack([state[key].float() for state in states], dim=0).mean(dim=0)
    return averaged


def train_candidate(candidate, args, loaders, labels, test_ds, device, use_cuda):
    runs = [
        train_single_seed(candidate, args, loaders, labels, test_ds, device, use_cuda, seed, f"seed{seed}")
        for seed in candidate.seeds
    ]
    val_prob = average_prob([run["val_prob"] for run in runs])
    test_prob = average_prob([run["test_prob"] for run in runs])
    train_prob = average_prob([run["train_prob"] for run in runs])
    train_y, val_y = labels
    val_acc = accuracy(val_y, np.argmax(val_prob, axis=1))
    test_labels = np.argmax(test_prob, axis=1).astype(int).tolist()
    output_dir = Path(args.data_dir) / "auto_moe_predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{candidate.name}.txt"
    write_labels(str(prediction_path), test_labels)
    return {
        "name": candidate.name,
        "kind": candidate.kind,
        "config": candidate.config,
        "seeds": candidate.seeds,
        "augmentation": candidate.augmentation,
        "train_acc": accuracy(train_y, np.argmax(train_prob, axis=1)),
        "val_acc": val_acc,
        "best_epoch": max(run["best_epoch"] for run in runs),
        "checkpoint": runs[0]["checkpoint"],
        "prediction_path": prediction_path,
        "val_prob": val_prob,
        "test_prob": test_prob,
    }


def run_search(args):
    data_dir = Path(args.data_dir)
    set_global_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda and args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if use_cuda else "cpu")
    loaders, labels, test_ds = make_loaders(data_dir, args.batch_size, args.test_batch_size, args.seed, use_cuda)
    candidates = build_candidate_plan(args.max_candidates)
    stopper = SearchStopper(BASELINE_VAL_ACC, 1 / len(labels[1]), args.no_improve_patience)
    results = []
    seen_kinds = set()
    for index, candidate in enumerate(candidates, start=1):
        print(f"\n[{index}/{len(candidates)}] Training {candidate.name} ({candidate.kind})")
        result = train_candidate(candidate, args, loaders, labels, test_ds, device, use_cuda)
        results.append(result)
        seen_kinds.add(candidate.kind)
        print(f"{candidate.name} val_acc={result['val_acc']:.4f}", flush=True)
        should_stop = stopper.observe(result["val_acc"])
        if should_stop and should_allow_early_stop(seen_kinds):
            print(f"Stopping after {stopper.no_improve_count} consecutive non-improving candidates.")
            break

    best = select_final_result(results)
    final_labels = np.argmax(best["test_prob"], axis=1).astype(int).tolist()
    output_path = Path(args.output)
    write_labels(str(output_path), final_labels)
    serializable_results = []
    for result in results:
        item = dict(result)
        item.pop("val_prob", None)
        item.pop("test_prob", None)
        serializable_results.append(item)
    serializable_best = dict(best)
    serializable_best.pop("val_prob", None)
    serializable_best.pop("test_prob", None)
    payload = make_results_payload(serializable_results, serializable_best, BASELINE_VAL_ACC)
    results_path = Path(args.results_json)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_best_notebook(Path(args.notebook_output), best)
    print(f"\nBest: {best['name']} ({best['kind']}) val_acc={best['val_acc']:.4f}")
    print(f"Wrote final predictions to {output_path}")
    print(f"Wrote results to {results_path}")
    print(f"Wrote notebook to {args.notebook_output}")
    return best, results


def notebook_source_for_best(best):
    config = best.get("config", baseline_config())
    checkpoint = Path(best.get("checkpoint", "")).as_posix()
    val_acc = float(best.get("val_acc", 0.0))
    name = best.get("name", "unknown")
    return f'''# SEED Auto Best Pure MoE

This notebook is generated from `auto_seed_moe_search.py` and contains only the selected pure MoE solution.

```python
import copy
import os
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from course_project.TEST_DATASET import TrainDataset, TestDataset

DATA_NAME = "SEED"
DATA_DIR = Path("G:/MLproject/course_project") / DATA_NAME
TRAIN_PATH = DATA_DIR / "train.h5"
VAL_PATH = DATA_DIR / "val.h5"
TEST_PATH = DATA_DIR / "test_x_only.h5"
OUTPUT_PATH = DATA_DIR / "SEED.txt"

SEED = 3407
CHANNELS = 62
CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 160
PATIENCE = 24
MIN_DELTA = 1e-4
MIN_LR = 1e-5
GRAD_CLIP = 1.0
BEST_CONFIG = {json.dumps(config, indent=4)}
BEST_NAME = {json.dumps(name)}
BEST_VAL_ACCURACY = {val_acc}
BEST_CHECKPOINT_PATH = Path({json.dumps(checkpoint)})

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.enabled = False
device = torch.device("cuda" if USE_CUDA else "cpu")

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_h5_y(path):
    with h5py.File(path, "r") as f:
        return f["y"][()].astype(np.int64)

def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def write_labels(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{{int(label)}}\\n")

def eeg_stat_features(x, eps=1e-6):
    mean = x.mean(dim=-1)
    std = x.std(dim=-1, unbiased=False)
    rms = torch.sqrt(torch.mean(x.square(), dim=-1) + eps)
    peak_to_peak = x.amax(dim=-1) - x.amin(dim=-1)
    return torch.cat([mean, std, rms, peak_to_peak], dim=1)

class SimpleMoEClassifier(nn.Module):
    def __init__(self, chans=CHANNELS, num_classes=CLASSES, hidden_dim=72, num_experts=4, dropout=0.30):
        super().__init__()
        kernels = (3, 7, 15, 31)[:num_experts]
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(chans, hidden_dim, kernel_size=kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden_dim, num_classes),
            )
            for kernel in kernels
        ])
        self.router = nn.Sequential(
            nn.Linear(chans * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.experts)),
        )

    def forward(self, x):
        route_logits = self.router(eeg_stat_features(x))
        route_weights = torch.softmax(route_logits, dim=1)
        expert_logits = torch.stack([expert(x) for expert in self.experts], dim=1)
        return torch.sum(route_weights.unsqueeze(-1) * expert_logits, dim=1)

    def clip_gradients(self, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)

def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device, dtype=torch.float32, non_blocking=USE_CUDA)
            label = label.to(device, dtype=torch.long, non_blocking=USE_CUDA)
            output = model(data)
            correct += (torch.argmax(output, dim=1) == label).sum().item()
            total += label.size(0)
    return correct / total

set_global_seed(SEED)
train_y = read_h5_y(TRAIN_PATH)
val_y = read_h5_y(VAL_PATH)
train_ds = TrainDataset(str(TRAIN_PATH))
val_ds = TrainDataset(str(VAL_PATH))
test_ds = TestDataset(str(TEST_PATH))
train_generator = torch.Generator().manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=train_generator, pin_memory=USE_CUDA)
train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=USE_CUDA)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=USE_CUDA)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, pin_memory=USE_CUDA)

model = SimpleMoEClassifier(
    hidden_dim=BEST_CONFIG["hidden_dim"],
    num_experts=BEST_CONFIG["num_experts"],
    dropout=BEST_CONFIG["dropout"],
).to(device)
if BEST_CHECKPOINT_PATH.exists():
    state = torch.load(BEST_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    loaded_val_acc = evaluate_accuracy(model, val_loader)
    print(f"Loaded {{BEST_NAME}} from {{BEST_CHECKPOINT_PATH}}")
    print(f"Checkpoint Val Accuracy: {{loaded_val_acc:.4f}} (search record: {{BEST_VAL_ACCURACY:.4f}})")
else:
    print(f"Checkpoint not found at {{BEST_CHECKPOINT_PATH}}; retraining from BEST_CONFIG.")
    criterion = nn.CrossEntropyLoss(label_smoothing=BEST_CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=BEST_CONFIG["lr"], weight_decay=BEST_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=BEST_CONFIG["scheduler_patience"],
        min_lr=MIN_LR,
    )
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, label in train_loader:
            data = data.to(device, dtype=torch.float32, non_blocking=USE_CUDA)
            label = label.to(device, dtype=torch.long, non_blocking=USE_CUDA)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            model.clip_gradients(GRAD_CLIP)
            optimizer.step()
        epoch_val_acc = evaluate_accuracy(model, val_loader)
        scheduler.step(epoch_val_acc)
        if epoch_val_acc > best_val_acc + MIN_DELTA:
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
        print(f"Epoch {{epoch:03d}}/{{EPOCHS}} | Val Acc: {{epoch_val_acc:.4f}} | Best: {{best_val_acc:.4f}} @ {{best_epoch:03d}}")
        if bad_epochs >= PATIENCE:
            break
    model.load_state_dict(best_state)
model.eval()
test_probs = []
with torch.no_grad():
    for batch in test_loader:
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        data = data.to(device, dtype=torch.float32, non_blocking=USE_CUDA)
        test_probs.append(torch.softmax(model(data), dim=1).cpu().numpy())
test_prob = np.concatenate(test_probs, axis=0)
all_test_labels = np.argmax(test_prob, axis=1).astype(int).tolist()
assert len(all_test_labels) == len(test_ds), f"Prediction count {{len(all_test_labels)}} != test sample count {{len(test_ds)}}"
write_labels(str(OUTPUT_PATH), all_test_labels)
print(f"Wrote {{len(all_test_labels)}} predictions to {{OUTPUT_PATH}}")
```
'''


def write_best_notebook(path, best):
    source = notebook_source_for_best(best)
    code = source.split("```python\n", 1)[1].rsplit("\n```", 1)[0]
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SEED Auto Best Pure MoE\n",
                    "\n",
                    "Generated from `auto_seed_moe_search.py`. Contains only the selected pure MoE solution.\n",
                ],
            },
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in code.splitlines()]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-search pure MoE SEED solutions.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output", default=str(DEFAULT_DATA_DIR / "SEED.txt"))
    parser.add_argument("--results-json", default=str(DEFAULT_DATA_DIR / "auto_moe_search_results.json"))
    parser.add_argument("--notebook-output", default="train_seed_auto_best_moe.ipynb")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--no-improve-patience", type=int, default=10)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--disable-cudnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable cuDNN while still using CUDA to avoid local cudnnCreate errors.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_search(parse_args())
