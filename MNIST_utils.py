"""Stacked-MNIST data, networks, evaluation and sampling helpers."""

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

from RenyiSampler import image_predictor, image_ratio_gradient, mobility
from sampling_utils import ROOT, SEED, gen, schedule, seed_all, sha


T = 1000


DEFAULT_CORRECTION_LEVELS = 64


def configure_compile_cache():
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", "/tmp/renyi_sampling_runtime/compile_cache"
    )
    os.environ.setdefault(
        "TRITON_CACHE_DIR", "/tmp/renyi_sampling_runtime/triton_cache"
    )
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "4")


def correction_anchors(levels=DEFAULT_CORRECTION_LEVELS):
    if not 1 <= levels <= T:
        raise ValueError(f"levels must be between 1 and {T}")
    return np.linspace(0, T - 1, levels).round().astype(int).tolist()


def save_sample_images(samples, run):
    pixels = ((samples[:100].detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    destination = Path(run) / "generated_samples"
    destination.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("L", (840, 28 * math.ceil(len(pixels) / 10)))
    for index, channels in enumerate(pixels):
        image = Image.fromarray(torch.cat(list(channels), dim=1).numpy())
        image.save(destination / f"{index:03d}.png")
        canvas.paste(image, ((index % 10) * 84, (index // 10) * 28))
    canvas.resize((canvas.width * 2, canvas.height * 2)).save(
        Path(run) / "generated.png"
    )


MNIST_ROOT = Path("outputs/MNIST")


RARE_PROBABILITY = 0.01


def configure_experiment(args):
    global MNIST_ROOT
    MNIST_ROOT = (ROOT / args.output_dir).resolve()


def stacked_weights():
    p = torch.full((1000,), (1 - RARE_PROBABILITY) / 900)
    p[torch.arange(1000) % 10 == 7] = RARE_PROBABILITY / 100
    return p


def make_composition(n, rng):
    """Pin digit triples; handwriting instances remain fresh on each access."""
    rare = rng.random_sample(n) < RARE_PROBABILITY
    first = rng.randint(0, 10, size=n)
    second = rng.randint(0, 10, size=n)
    common = np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9])
    third = common[rng.randint(0, len(common), size=n)]
    third[rare] = 7
    triples = torch.from_numpy(np.stack([first, second, third], axis=1)).long()
    labels = triples[:, 0] * 100 + triples[:, 1] * 10 + triples[:, 2]
    return triples, labels


def prepare(n=1000000):
    root = MNIST_ROOT
    root.mkdir(parents=True, exist_ok=True)
    if (root / "data.pt").exists():
        payload = torch.load(root / "data.pt", map_location="cpu", weights_only=True)
        if payload.get("rare_probability", 0.01) != RARE_PROBABILITY:
            raise ValueError("Saved composition prior differs from requested prior")
        if len(payload["train_digits"]) != n:
            raise ValueError(
                "Saved composition size differs from requested training size"
            )
        return
    train = MNIST(root / "raw", train=True, download=True)
    test = MNIST(root / "raw", train=False, download=True)
    rng = np.random.RandomState(SEED)
    train_digits, train_modes = make_composition(n, rng)
    test_digits, test_modes = make_composition(50000, rng)
    torch.save(
        {
            "samples": [tuple(row) for row in train_digits.tolist()],
            "labels": train_modes.tolist(),
        },
        root / "composition.pt",
    )
    torch.save(
        {
            "train_images": train.data,
            "train_labels": train.targets,
            "test_images": test.data,
            "test_labels": test.targets,
            "train_digits": train_digits,
            "train_modes": train_modes,
            "test_digits": test_digits,
            "test_modes": test_modes,
            "rare_probability": RARE_PROBABILITY,
            "seed": SEED,
        },
        root / "data.pt",
    )
    print(
        f"Saved {n} digit triples; rare fraction={(train_modes % 10 == 7).float().mean().item():.6f}",
        flush=True,
    )


class StackedData:
    """Saved composition with independent handwriting redraws per access."""

    def __init__(self, device="cuda"):
        payload = torch.load(
            str(MNIST_ROOT / "data.pt"), map_location="cpu", weights_only=True
        )
        if payload.get("rare_probability", 0.01) != RARE_PROBABILITY:
            raise ValueError("Saved dataset prior differs from configured prior")
        self.device = device
        for name in [
            "train_images",
            "test_images",
            "train_digits",
            "test_digits",
            "train_modes",
            "test_modes",
            "train_labels",
            "test_labels",
        ]:
            setattr(self, name, payload[name].to(device))
        for split in ["train", "test"]:
            labels = getattr(self, split + "_labels")
            pools = [(labels == digit).nonzero().flatten() for digit in range(10)]
            sizes = torch.tensor([len(p) for p in pools], device=device)
            padded = torch.zeros(
                10, max(len(p) for p in pools), dtype=torch.long, device=device
            )
            for digit, pool in enumerate(pools):
                padded[digit, : len(pool)] = pool
            setattr(self, split + "_pools", padded)
            setattr(self, split + "_sizes", sizes)

    def from_indices(self, ids, test=False, g=None):
        prefix = "test" if test else "train"
        digits = getattr(self, prefix + "_digits")[ids]
        pools = getattr(self, prefix + "_pools")
        sizes = getattr(self, prefix + "_sizes")
        draws = torch.rand(digits.shape, device=self.device, generator=g)
        slots = (draws * sizes[digits]).long()
        indices = pools[digits, slots]
        return getattr(self, prefix + "_images")[indices].float() / 127.5 - 1

    def sample(self, n, g, test=False):
        recipes = self.test_digits if test else self.train_digits
        ids = torch.randint(len(recipes), (n,), device=self.device, generator=g)
        return self.from_indices(ids, test=test, g=g)


class TimeEmbedding(torch.nn.Module):
    def __init__(self, width=128, T=1000):
        super().__init__()
        self.T = T
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), 32)))
        self.net = torch.nn.Sequential(
            torch.nn.Linear(64, width), torch.nn.SiLU(), torch.nn.Linear(width, width)
        )

    def forward(self, t, n, device):
        t = torch.as_tensor(t, device=device)
        if t.ndim == 0:
            t = t.expand(n)
        p = t[:, None] / (self.T - 1) * self.freq
        return self.net(torch.cat([p.sin(), p.cos()], 1))


class Block(torch.nn.Module):
    def __init__(self, cin, cout, td=128):
        super().__init__()
        self.n1 = torch.nn.GroupNorm(8, cin)
        self.c1 = torch.nn.Conv2d(cin, cout, 3, padding=1)
        self.time = torch.nn.Linear(td, cout)
        self.n2 = torch.nn.GroupNorm(8, cout)
        self.c2 = torch.nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = (
            torch.nn.Conv2d(cin, cout, 1) if cin != cout else torch.nn.Identity()
        )

    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x))) + self.time(F.silu(t))[:, :, None, None]
        return (self.c2(F.silu(self.n2(h))) + self.skip(x)) / math.sqrt(2)


class UNet(torch.nn.Module):
    def __init__(self, T=1000, base=32):
        super().__init__()
        self.time = TimeEmbedding(T=T)
        b = base
        self.inp = torch.nn.Conv2d(3, b, 3, padding=1)
        self.e1 = Block(b, b)
        self.d1 = torch.nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.e2 = Block(2 * b, 2 * b)
        self.d2 = torch.nn.Conv2d(2 * b, 3 * b, 4, stride=2, padding=1)
        self.mid1 = Block(3 * b, 3 * b)
        self.mid2 = Block(3 * b, 3 * b)
        self.u2 = Block(5 * b, 2 * b)
        self.u1 = Block(3 * b, b)
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(8, b),
            torch.nn.SiLU(),
            torch.nn.Conv2d(b, 3, 3, padding=1),
        )

    def forward(self, x, t):
        emb = self.time(t, len(x), x.device)
        a = self.e1(self.inp(x), emb)
        b = self.e2(self.d1(a), emb)
        m = self.mid2(self.mid1(self.d2(b), emb), emb)
        u = self.u2(
            torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), b], 1), emb
        )
        return self.out(
            self.u1(
                torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), a], 1), emb
            )
        )


class RatioCNN(torch.nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.time = TimeEmbedding(64, T)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(32, 48, 4, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(48, 64, 4, stride=2, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(64, 96, 4, stride=2, padding=1),
            torch.nn.SiLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(96 * 3 * 3 + 64, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x, t):
        return self.head(
            torch.cat([self.conv(x).flatten(1), self.time(t, len(x), x.device)], 1)
        ).flatten()


class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def mnist_schedule():
    return schedule(T, start=1e-4, end=0.02)


def train_classifier(data):
    path = MNIST_ROOT / "classifier.pt"
    model = DigitClassifier().cuda()
    if path.exists():
        model.load_state_dict(
            torch.load(path, weights_only=True, map_location="cuda")["state"]
        )
        return model.eval().requires_grad_(False)
    seed_all()
    model = DigitClassifier().cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    g = gen(SEED + 10)
    images = data.train_images.float()[:, None] / 255
    labels = data.train_labels
    hist = []
    for epoch in range(8):
        model.train()
        perm = torch.randperm(len(images), device="cuda", generator=g)
        for idx in perm.split(256):
            loss = F.cross_entropy(model(images[idx]), labels[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat(
                [
                    model(q.float()[:, None] / 255).argmax(1)
                    for q in data.test_images.split(512)
                ]
            )
        acc = (pred == data.test_labels).float().mean().item()
        hist.append({"epoch": epoch + 1, "test_accuracy": acc})
        print(json.dumps({"event": "classifier", **hist[-1]}), flush=True)
    confusion = torch.bincount(data.test_labels * 10 + pred, minlength=100).reshape(
        10, 10
    )
    torch.save({"state": model.state_dict(), "seed": SEED, "epochs": 8}, path)
    torch.save(
        {"test_accuracy": acc, "confusion_matrix": confusion.cpu()},
        MNIST_ROOT / "classifier_validation.pt",
    )
    return model.eval().requires_grad_(False)


def load_ddpm():
    saved = torch.load(
        str(MNIST_ROOT / "ddpm.pt"), map_location="cuda", weights_only=False
    )
    assert saved["dataset_sha256"] == sha(MNIST_ROOT / "data.pt")
    assert saved.get("rare_probability", 0.01) == RARE_PROBABILITY
    model = UNet(T).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(saved["ema"])
    model.eval().requires_grad_(False)

    configure_compile_cache()
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    def predictor(x, t):
        torch.compiler.cudagraph_mark_step_begin()
        time_index = torch.as_tensor(t, device=x.device)
        if time_index.ndim == 0:
            time_index = time_index.expand(len(x))
        return compiled(x, time_index).clone()

    return predictor


def evaluator():
    model = DigitClassifier().cuda()
    model.load_state_dict(
        torch.load(
            str(MNIST_ROOT / "classifier.pt"), map_location="cuda", weights_only=True
        )["state"]
    )
    return model.eval().requires_grad_(False)


@torch.no_grad()
def evaluate_modes(x, classifier):
    digits = []
    confidence = []
    for q in x.split(256):
        crop = ((q + 1) / 2).clamp(0, 1)
        probs = classifier(crop.reshape(-1, 1, 28, 28)).softmax(1)
        conf, idx = probs.max(1)
        digits.append(idx.reshape(-1, 3))
        confidence.append(conf.reshape(-1, 3))
    digits = torch.cat(digits)
    confidence = torch.cat(confidence)
    modes = digits[:, 0] * 100 + digits[:, 1] * 10 + digits[:, 2]
    valid = confidence.min(1).values >= 0.9
    n = len(x)
    counts = torch.bincount(modes, minlength=1000)
    accepted = torch.bincount(modes[valid], minlength=1000)
    truth = stacked_weights().to(x.device)
    rare = torch.arange(1000, device=x.device) % 10 == 7
    freq = counts.float() / n
    accepted_freq = accepted.float() / n
    reject = (~valid).float().mean()
    p = truth.cpu().numpy().astype(float)
    expected = 1 - (1 - p) ** n
    result = {
        "samples": n,
        "tv_distance": (0.5 * (freq - truth).abs().sum()).item(),
        "confidence_filtered_tv": (
            0.5 * ((accepted_freq - truth).abs().sum() + reject)
        ).item(),
        "mode_coverage_recall": (counts > 0).float().mean().item(),
        "rare_mode_coverage_recall": (counts[rare] > 0).float().mean().item(),
        "common_mode_coverage_recall": (counts[~rare] > 0).float().mean().item(),
        "mode_recall_at5": (counts >= 5).float().mean().item(),
        "rare_mode_recall_at5": (counts[rare] >= 5).float().mean().item(),
        "confident_mode_coverage_recall": (accepted > 0).float().mean().item(),
        "confident_rare_mode_coverage_recall": (accepted[rare] > 0)
        .float()
        .mean()
        .item(),
        "rare_mass": freq[rare].sum().item(),
        "confident_rare_mass": accepted_freq[rare].sum().item(),
        "low_confidence_fraction": reject.item(),
        "expected_exact_mode_coverage": float(expected.mean()),
        "expected_exact_rare_mode_coverage": float(
            expected[np.arange(1000) % 10 == 7].mean()
        ),
        "mode_counts": counts.cpu().tolist(),
        "confident_mode_counts": accepted.cpu().tolist(),
        "nonfinite_count": int((~torch.isfinite(x)).sum()),
    }
    return result, modes.cpu(), confidence.cpu()


@torch.no_grad()
def baseline_bank(model, n=4096, levels=DEFAULT_CORRECTION_LEVELS):
    anchors = correction_anchors(levels)
    beta, abar = mnist_schedule()
    g = gen(SEED + 20)
    x = torch.randn(n, 3, 28, 28, device="cuda", generator=g)
    bank = {}
    for t in reversed(range(T)):
        if t in anchors:
            bank[t] = x.cpu().half()
        x = image_predictor(model, x, t, beta, abar, g, chunk=1024)
    return bank, x


def discriminator_loss(d, fake, real, t, regularize=False):
    hf = d(fake, t)
    hr = d(real, t)
    loss = F.softplus(-hf).mean() + F.softplus(hr).mean()
    if regularize:
        q = real[:32].detach().requires_grad_(True)
        hq = d(q, t)
        grad = torch.autograd.grad(hq.sum(), q, create_graph=True)[0]
        loss = loss + 0.01 * grad.flatten(1).square().sum(1).mean()
    return loss, hf, hr


def online_adapter(data, abar, g):
    state = {"optimizer": None}

    def adapt(d, reference, t, substep):
        d.train().requires_grad_(True)
        if state["optimizer"] is None:
            state["optimizer"] = torch.optim.Adam(d.parameters(), lr=1e-4)
        opt = state["optimizer"]
        train = reference[:768]
        validation = reference[768:]
        for step in range(12):
            idx = torch.randint(len(train), (128,), device="cuda", generator=g)
            fake = train[idx]
            clean = data.sample(128, g)
            real = abar[t].sqrt() * clean + (1 - abar[t]).sqrt() * torch.randn(
                clean.shape, device="cuda", generator=g
            )
            loss, _, _ = discriminator_loss(d, fake, real, t, regularize=step == 11)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), 5.0)
            opt.step()
        d.eval().requires_grad_(False)
        with torch.no_grad():
            real = data.sample(len(validation), g, test=True)
            real = abar[t].sqrt() * real + (1 - abar[t]).sqrt() * torch.randn(
                real.shape, device="cuda", generator=g
            )
            val, ff, rr = discriminator_loss(d, validation, real, t)
        return {
            "ratio_train_bce": loss.item(),
            "ratio_validation_bce": val.item(),
            "ratio_validation_accuracy": 0.5
            * ((ff > 0).float().mean() + (rr < 0).float().mean()).item(),
        }

    return adapt


def validate_mnist():
    prior = stacked_weights()
    rare = torch.arange(1000) % 10 == 7
    assert abs(prior[rare].sum().item() - RARE_PROBABILITY) < 1e-6
    payload = torch.load(
        ROOT / str(MNIST_ROOT / "data.pt"), map_location="cpu", weights_only=True
    )
    assert (
        abs((payload["train_modes"] % 10 == 7).float().mean().item() - RARE_PROBABILITY)
        < 0.001
    )
    composition = torch.load(
        ROOT / str(MNIST_ROOT / "composition.pt"), weights_only=True
    )
    assert torch.equal(torch.as_tensor(composition["samples"]), payload["train_digits"])
    assert torch.equal(torch.as_tensor(composition["labels"]), payload["train_modes"])
    assert len(payload["train_modes"]) == 1000000
    model = UNet()
    x = torch.randn(2, 3, 28, 28)
    assert model(x, 3).shape == x.shape
    discriminator = RatioCNN().double()
    x = x.double()
    h, gradient = image_ratio_gradient(discriminator, x, 3)
    direction = torch.randn_like(x)
    direction /= direction.norm()
    with torch.no_grad():
        delta = (
            discriminator(x + 0.01 * direction, 3).sum()
            - discriminator(x - 0.01 * direction, 3).sum()
        ) / 0.02
    torch.testing.assert_close(
        delta, (gradient * direction).sum(), rtol=0.03, atol=1e-5
    )
    a, da, _ = mobility(h, gradient, 1.0, 1)
    assert torch.equal(a, torch.ones_like(a)) and torch.equal(da, torch.zeros_like(da))
    print("MNIST validation passed")


class SpatialAttention(torch.nn.Module):
    """Global spatial attention at the 14x14 and 7x7 UNet resolutions."""

    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError("Channels must be divisible by attention heads")
        self.heads = heads
        self.norm = torch.nn.GroupNorm(8, channels)
        self.qkv = torch.nn.Conv2d(channels, channels * 3, 1)
        self.proj = torch.nn.Conv2d(channels, channels, 1)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        n, c, height, width = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            n, 3, self.heads, c // self.heads, height * width
        )
        q, k, v = [part.transpose(-1, -2).contiguous() for part in qkv.unbind(1)]
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attended = attended.transpose(-1, -2).reshape(n, c, height, width)
        return x + self.proj(attended)


class AttentionUNet(torch.nn.Module):
    """A deeper epsilon-prediction UNet with global attention; native 28x28."""

    def __init__(self, T=1000, base=64):
        super().__init__()
        b = base
        self.time = TimeEmbedding(T=T)
        self.inp = torch.nn.Conv2d(3, b, 3, padding=1)
        self.e1 = torch.nn.ModuleList([Block(b, b), Block(b, b)])
        self.d1 = torch.nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.e2 = torch.nn.ModuleList([Block(2 * b, 2 * b), Block(2 * b, 2 * b)])
        self.a2 = SpatialAttention(2 * b)
        self.d2 = torch.nn.Conv2d(2 * b, 3 * b, 4, stride=2, padding=1)
        self.mid1 = Block(3 * b, 3 * b)
        self.amid = SpatialAttention(3 * b)
        self.mid2 = Block(3 * b, 3 * b)
        self.u2 = torch.nn.ModuleList([Block(5 * b, 2 * b), Block(2 * b, 2 * b)])
        self.au2 = SpatialAttention(2 * b)
        self.u1 = torch.nn.ModuleList([Block(3 * b, b), Block(b, b)])
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(8, b),
            torch.nn.SiLU(),
            torch.nn.Conv2d(b, 3, 3, padding=1),
        )

    def forward(self, x, t):
        emb = self.time(t, len(x), x.device)
        a = self.inp(x)
        for block in self.e1:
            a = block(a, emb)
        b = self.d1(a)
        for block in self.e2:
            b = block(b, emb)
        b = self.a2(b)
        m = self.mid2(self.amid(self.mid1(self.d2(b), emb)), emb)
        u = torch.cat([F.interpolate(m, size=b.shape[-2:], mode="nearest"), b], dim=1)
        for block in self.u2:
            u = block(u, emb)
        u = self.au2(u)
        u = self.u1[0](
            torch.cat([F.interpolate(u, size=a.shape[-2:], mode="nearest"), a], dim=1),
            emb,
        )
        return self.out(self.u1[1](u, emb))


def make_fidelity_model(base=64, architecture="plain"):
    if architecture == "plain":
        return UNet(T, base=base)
    if architecture == "attention":
        return AttentionUNet(T, base=base)
    raise ValueError(f"Unknown denoiser architecture: {architecture}")
