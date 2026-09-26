"""Experiment-independent seeds, diffusion schedules and artifact I/O."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


SEED = 20260912


ORDERS = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3]


SEEDS = [42, 43, 44, 45, 46]


ROOT = Path(__file__).resolve().parent


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False


def gen(seed, device="cuda"):
    return torch.Generator(device=device).manual_seed(seed)


def schedule(T=256, device="cuda", start=1e-5, end=0.08):
    beta = torch.linspace(start, end, T, device=device)
    return beta, (1 - beta).cumprod(0)


def activate_workspace():
    import os

    ROOT.mkdir(exist_ok=True)
    os.chdir(ROOT)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)
