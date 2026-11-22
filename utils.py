"""Question-independent helpers for the CW3 notebooks.

Kept here, as the brief allows: environment checks, seeding, data loading,
batching, evaluation, result files and plotting. Code that matters for the
research question (the residual block, initialisation, calibration, the
training loop and the Jacobian metrics) is written and documented in the
notebooks.

Run `python utils.py` on a lab machine for a quick smoke test.
"""
from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Environment and reproducibility
# ---------------------------------------------------------------------------
def env_report(verbose: bool = True) -> dict:
    """Collect (and optionally print) versions and the compute device."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None))
        and torch.backends.mps.is_available(),
        "device": str(get_device()),
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except ImportError:
        info["torchvision"] = "not installed"
    try:
        import matplotlib
        info["matplotlib"] = matplotlib.__version__
    except ImportError:
        info["matplotlib"] = "not installed"
    if verbose:
        for k, v in info.items():
            print(f"{k:>15}: {v}")
    return info


def get_device(allow_mps: bool = True) -> torch.device:
    """CUDA if available, else Apple MPS (local development), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if allow_mps and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int) -> torch.Generator:
    """A CPU generator, e.g. for reproducible shuffling order."""
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_fashion_mnist(device: torch.device | str | None = None,
                       n_val: int = 10_000, split_seed: int = 0,
                       n_calib: int = 512, calib_seed: int = 1,
                       data_dir: Path | str = DATA_DIR) -> dict:
    """Load Fashion-MNIST as flat, standardised tensors.

    - Images are flattened to 784 values in [0, 1], then standardised with the
      mean and std (single scalars) of the training split only.
    - The 60k official training images are split into train (60k - n_val) and
      validation (n_val) with a fixed seed, independent of the run seed.
    - `x_calib` is a fixed batch of `n_calib` training images for calibration.

    Returns a dict with x_train, y_train, x_val, y_val, x_test, y_test,
    x_calib, mean, std.
    """
    import torchvision  # imported here so `import utils` works without it

    train = torchvision.datasets.FashionMNIST(str(data_dir), train=True, download=True)
    test = torchvision.datasets.FashionMNIST(str(data_dir), train=False, download=True)

    x_all = train.data.reshape(len(train.data), -1).float() / 255.0
    y_all = train.targets.clone().long()
    x_test = test.data.reshape(len(test.data), -1).float() / 255.0
    y_test = test.targets.clone().long()

    perm = torch.randperm(len(x_all), generator=make_generator(split_seed))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]

    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    calib_idx = torch.randperm(len(x_train), generator=make_generator(calib_seed))[:n_calib]
    x_calib = x_train[calib_idx].clone()

    out = {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "x_calib": x_calib,
        "mean": float(mean), "std": float(std),
    }
    if device is not None:
        for k, v in out.items():
            if torch.is_tensor(v):
                out[k] = v.to(device)
    return out


def iterate_minibatches(x: torch.Tensor, y: torch.Tensor, batch_size: int,
                        generator: torch.Generator | None = None,
                        shuffle: bool = True, drop_last: bool = False):
    """Yield (xb, yb) mini-batches. Pass a seeded generator for a reproducible order."""
    n = len(x)
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(x.device)
    else:
        idx = torch.arange(n, device=x.device)
    stop = n - (n % batch_size) if drop_last else n
    for start in range(0, stop, batch_size):
        b = idx[start:start + batch_size]
        yield x[b], y[b]


@torch.no_grad()
def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             batch_size: int = 4096) -> dict:
    """Mean cross-entropy and accuracy over a whole dataset, in eval mode.

    Use this (not a running average over training batches) for every reported
    loss, so loss and accuracy always describe the same model state.
    """
    was_training = model.training
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in iterate_minibatches(x, y, batch_size, shuffle=False):
        logits = model(xb)
        total_loss += torch.nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=1) == yb).sum().item()
    model.train(was_training)
    n = len(x)
    return {"loss": total_loss / n, "acc": correct / n}


# ---------------------------------------------------------------------------
# Result files (one JSON per training run)
# ---------------------------------------------------------------------------
_KEY_FIELDS = ("depth", "alpha", "lr", "seed", "epochs")
DEFAULT_WIDTH = 256


def run_name(depth: int, alpha: float, lr: float, seed: int, epochs: int, tag: str = "",
             width: int = DEFAULT_WIDTH) -> str:
    """Canonical file stem for one run, e.g. 'L32_w256_a0.7071_lr0.003_s1_e6'."""
    stem = f"L{depth}_w{width}_a{alpha:.4f}_lr{lr:g}_s{seed}_e{epochs}"
    return f"{stem}_{tag}" if tag else stem


def result_path(depth, alpha, lr, seed, epochs, tag="", width=DEFAULT_WIDTH,
                results_dir=RESULTS_DIR) -> Path:
    return Path(results_dir) / f"{run_name(depth, alpha, lr, seed, epochs, tag, width)}.json"


def result_exists(depth, alpha, lr, seed, epochs, tag="", width=DEFAULT_WIDTH,
                  results_dir=RESULTS_DIR) -> bool:
    """True if this run is already saved (lets long sweeps resume after interruption)."""
    return result_path(depth, alpha, lr, seed, epochs, tag, width, results_dir).exists()


def git_commit() -> str | None:
    """Short hash of the current commit, or None outside a git repository."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(v):
    if torch.is_tensor(v):
        v = v.detach().cpu()
        v = v.item() if v.ndim == 0 else v.tolist()
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)  # JSON has no NaN/inf; stored as "nan" / "inf"
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def save_result(record: dict, tag: str = "", results_dir: Path | str = RESULTS_DIR,
                overwrite: bool = False) -> Path:
    """Save one run as JSON. `record` must contain depth, alpha, lr, seed, epochs
    (and width, if it is not the default 256).

    Adds the time, git commit and library versions so every cached number can
    be traced back to the code that produced it.
    """
    missing = [k for k in _KEY_FIELDS if k not in record]
    if missing:
        raise KeyError(f"record is missing {missing}")
    record = {"width": DEFAULT_WIDTH, **record}
    path = result_path(*(record[k] for k in _KEY_FIELDS), tag=tag, width=record["width"],
                       results_dir=results_dir)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
            "torch": torch.__version__, "tag": tag}
    payload = {**_jsonable(record), "_meta": meta}
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_results(results_dir: Path | str = RESULTS_DIR, tag: str | None = None, **filters) -> list[dict]:
    """Load saved runs, optionally filtered, e.g. load_results(depth=32, seed=1).

    Float filters (alpha, lr) match within 1e-6. `tag=None` ignores tags.
    """
    records = []
    for p in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(p.read_text())
        if tag is not None and r.get("_meta", {}).get("tag", "") != tag:
            continue
        ok = True
        for k, want in filters.items():
            have = r.get(k)
            if isinstance(want, float) or isinstance(have, float):
                ok = have is not None and abs(float(have) - float(want)) < 1e-6
            else:
                ok = have == want
            if not ok:
                break
        if ok:
            records.append(r)
    return records


def summarise(records: list[dict], metric: str, by=("depth", "width", "alpha")) -> list[dict]:
    """Group runs by `by` and report n, mean and sample sd of `metric`.

    Diverged runs (record['diverged'] true) are counted but not averaged.
    Records saved without a width count as width 256.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        r = {"width": DEFAULT_WIDTH, **r}
        key = tuple(round(r[k], 6) if isinstance(r[k], float) else r[k] for k in by)
        groups.setdefault(key, []).append(r)
    rows = []
    for key in sorted(groups):
        rs = groups[key]
        vals = [float(r[metric]) for r in rs if not r.get("diverged") and metric in r]
        rows.append({
            **dict(zip(by, key)),
            "n": len(vals),
            "n_diverged": sum(bool(r.get("diverged")) for r in rs),
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan"),
            "values": vals,
        })
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_mean_sd(ax, x, runs, label=None, **kwargs):
    """Mean line with a +/- 1 sd band over seeds. `runs`: array-like (n_seeds, len(x))."""
    runs = np.asarray(runs, dtype=float)
    mean = runs.mean(axis=0)
    line, = ax.plot(x, mean, label=label, **kwargs)
    if len(runs) > 1:
        sd = runs.std(axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.2, color=line.get_color(), linewidth=0)
    return line


def plot_seed_points(ax, x, values_per_x, label=None, jitter=0.0, seed=0, **kwargs):
    """One dot per seed at each x, plus a line through the means."""
    rng = np.random.default_rng(seed)
    means = []
    line = None
    for xi, vals in zip(x, values_per_x):
        vals = np.asarray(vals, dtype=float)
        xs = xi + jitter * rng.uniform(-1, 1, size=len(vals))
        pts = ax.scatter(xs, vals, s=14, alpha=0.6,
                         color=None if line is None else line.get_color())
        means.append(vals.mean() if len(vals) else np.nan)
        if line is None:
            line, = ax.plot([], [], label=label, color=pts.get_facecolor()[0], **kwargs)
    line.set_data(x, means)
    ax.relim()
    ax.autoscale_view()
    return line


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("== environment ==")
    env_report()
    print("\n== data ==")
    d = load_fashion_mnist()
    for k in ("x_train", "x_val", "x_test", "x_calib"):
        print(f"{k:>8}: {tuple(d[k].shape)}")
    print(f"train mean/std after standardising: {d['x_train'].mean():.3f} / {d['x_train'].std():.3f}")
    xb, yb = next(iterate_minibatches(d["x_train"], d["y_train"], 128, generator=make_generator(0)))
    print("first batch:", tuple(xb.shape), tuple(yb.shape))
    print("\n== result files ==")
    with tempfile.TemporaryDirectory() as tmp:
        save_result({"depth": 32, "alpha": 0.5, "lr": 0.003, "seed": 1, "epochs": 6,
                     "final_train_loss": 0.3, "diverged": False}, results_dir=tmp)
        print(load_results(tmp, depth=32)[0]["final_train_loss"], "-> round trip ok")
    print("\nsmoke test passed")
