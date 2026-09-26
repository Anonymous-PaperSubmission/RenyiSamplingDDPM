"""GMM models, analytic mixture kernels, training and sampling orchestration."""

import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from sampling_utils import (
    ROOT,
    SEED,
    SEEDS,
    ORDERS,
    activate_workspace,
    atomic_torch_save,
    gen,
    save_json,
    schedule,
    seed_all,
    sha,
)


class VectorDDPM(torch.nn.Module):
    def __init__(self, dim, width=192, T=256):
        super().__init__()
        self.T = T
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.first = torch.nn.Linear(dim + 32, width)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.SiLU(),
                    torch.nn.Linear(width, width),
                    torch.nn.SiLU(),
                    torch.nn.Linear(width, width),
                )
                for _ in range(3)
            ]
        )
        self.last = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(width, dim))

    def forward(self, x, t):
        tt = torch.as_tensor(t, device=x.device)
        if tt.ndim == 0:
            tt = tt.expand(len(x))
        p = tt[:, None] / (self.T - 1) * self.freq
        h = self.first(torch.cat([x, p.sin(), p.cos()], 1))
        for b in self.blocks:
            h = (h + b(h)) / math.sqrt(2)
        return self.last(h)


class GaussianMixtureBase:
    """Shared sampling and noisy-density kernel for finite Gaussian mixtures."""

    def to(self, device):
        self.centers = self.centers.to(device)
        self.weights = self.weights.to(device)
        self.device = device
        return self

    def sample(self, n, generator):
        idx = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        return self.centers[idx] + self.std * torch.randn(
            n, self.dim, device=self.device, generator=generator
        )

    def log_score(self, x, abar=1.0, chunk=1024):
        centers = self.centers * torch.as_tensor(abar, device=x.device).sqrt()
        var = abar * self.std**2 + 1 - abar
        cs = centers.square().sum(1)
        logs = []
        scores = []
        for q in x.split(chunk):
            d2 = (q.square().sum(1)[:, None] + cs[None] - 2 * q @ centers.T).clamp_min(
                0
            )
            lc = -d2 / (2 * var) + self.weights.log()[None]
            logs.append(
                torch.logsumexp(lc, 1)
                - self.dim
                / 2
                * torch.log(torch.as_tensor(2 * math.pi * var, device=x.device))
            )
            scores.append((lc.softmax(1) @ centers - q) / var)
        return torch.cat(logs), torch.cat(scores)


def train_vector_ddpm(target, folder, steps):
    folder.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    data_path = folder / "dataset.pt"
    if not data_path.exists():
        target.to("cpu")
        g = gen(SEED, "cpu")
        data = {
            split: target.sample(n, g)
            for split, n in [("train", 100000), ("validation", 10000), ("test", 50000)]
        }
        torch.save(data, data_path)
        torch.save(
            {"centers": target.centers, "weights": target.weights, "std": target.std},
            folder / "target.pt",
        )
    target.to(device)
    data = torch.load(data_path, weights_only=True, map_location=device)
    T = 256
    beta, abar = schedule(T)
    model = VectorDDPM(target.dim, T=T).cuda()
    if (folder / "model.pt").exists():
        saved = torch.load(folder / "model.pt", weights_only=False, map_location=device)
        model.load_state_dict(saved["ema"])
        return model.eval().requires_grad_(False), beta, abar, data
    seed_all()
    model = VectorDDPM(target.dim, T=T).cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    g = gen(SEED + 1)
    vg = gen(SEED + 2)
    clean = data["validation"]
    vt = torch.randint(T, (len(clean),), device=device, generator=vg)
    eps = torch.randn(clean.shape, device=device, generator=vg)
    vx = abar[vt, None].sqrt() * clean + (1 - abar[vt, None]).sqrt() * eps
    history = []
    start = time.perf_counter()
    rolling = 0.0
    for step in range(1, steps + 1):
        lr = min(1, step / 300) * (
            1e-4 + 4.5e-4 * (1 + math.cos(math.pi * step / steps))
        )
        for p in optim.param_groups:
            p["lr"] = lr
        idx = torch.randint(len(data["train"]), (1024,), device=device, generator=g)
        x = data["train"][idx]
        t = torch.randint(T, (len(x),), device=device, generator=g)
        noise = torch.randn(x.shape, device=device, generator=g)
        xt = abar[t, None].sqrt() * x + (1 - abar[t, None]).sqrt() * noise
        loss = (model(xt, t) - noise).square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optim.step()
        with torch.no_grad():
            for p, q in zip(ema.parameters(), model.parameters()):
                p.lerp_(q, 1 - min(0.999, (1 + step) / (10 + step)))
        rolling += loss.item()
        if step % 1000 == 0:
            with torch.no_grad():
                vl = (ema(vx, vt) - eps).square().mean().item()
            row = {
                "step": step,
                "train_loss": rolling / 1000,
                "validation_loss": vl,
                "seconds": time.perf_counter() - start,
            }
            history.append(row)
            rolling = 0.0
            print(json.dumps({"target": folder.name, **row}), flush=True)
        if step % 10000 == 0 or step == steps:
            state = {
                "ema": ema.state_dict(),
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "rng": g.get_state(),
                "seed": SEED,
                "step": step,
                "dim": target.dim,
                "T": T,
                "dataset_sha256": sha(data_path),
            }
            torch.save(state, folder / f"step_{step}.pt")
    torch.save(state, folder / "model.pt")
    return ema, beta, abar, data


def run_vector_experiment(target, folder, steps, name, train_only=False):
    seed_all()
    activate_workspace()
    from RenyiSampler import vector_sample

    model, beta, abar, data = train_vector_ddpm(target, folder, steps)
    if train_only:
        return
    settings = [(0, 1.0)] + [(k, a) for k in [1, 2, 3] for a in ORDERS]
    if not (folder / "target_calibration.json").exists():
        truth = []
        for seed in SEEDS:
            x = target.sample(5000, gen(3000000 + seed))
            truth.append({"seed": seed, **target.metrics(x)})
        save_json(folder / "target_calibration.json", truth)
    completed = 0
    for score_type in ["learned", "oracle"]:
        for k, a in settings:
            for seed in SEEDS:
                run = folder / "runs" / score_type / f"K{k}_a{a:g}_s{seed}"
                if (run / "metrics.json").exists():
                    completed += 1
                    continue
                run.mkdir(parents=True, exist_ok=True)
                m, x, ref, trace = vector_sample(
                    model, target, beta, abar, k, a, seed, score_type
                )
                torch.save({"samples": x, "reference": ref}, run / "samples.pt")
                save_json(run / "metrics.json", m)
                completed += 1
                print(
                    json.dumps(
                        {
                            "event": "geometry_run",
                            "target": name,
                            "completed": completed,
                            "score": score_type,
                            "K": k,
                            "alpha": a,
                            "seed": seed,
                        }
                    ),
                    flush=True,
                )


# Shared orchestration for the paper-faithful vector sweeps.


def plot_vector_run(samples, target, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = samples.detach().cpu().numpy()
    c = target.centers.detach().cpu().numpy()
    finite = np.isfinite(x).all(1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax in axes:
        ax.scatter(x[finite, 0], x[finite, 1], s=2, alpha=0.3)
        ax.plot(c[:, 0], c[:, 1], "r.", ms=3)
        ax.set(xlabel="Coordinate 1", ylabel="Coordinate 2")
    pad = max(float(np.ptp(c[:, :2], axis=0).max()) * 0.15, target.std * 6)
    axes[0].set(
        xlim=(c[:, 0].min() - pad, c[:, 0].max() + pad),
        ylim=(c[:, 1].min() - pad, c[:, 1].max() + pad),
        title="Target-scale view",
    )
    axes[1].set_title("Full extent; no sample filtering beyond nonfinite coordinates")
    fig.suptitle(
        title
        + (
            f"; {target.dim}D projected onto first two coordinates"
            if target.dim > 2
            else ""
        )
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def paper_vector_sweep(
    target,
    model,
    beta,
    abar,
    folder,
    checkpoint,
    dataset,
    metrics_fn,
    reference_count=512,
    bandwidth_floor=0.01,
    estimator="kde",
    samples=5000,
    seeds=SEEDS,
    score_types=("learned", "oracle"),
    settings=None,
):
    from RenyiSampler import (
        paper_transport_sample,
        gaussian_kde_log_prob_score,
        CurrentVectorRatio,
        PaperNumericalFailure,
    )

    seed_all()
    folder, checkpoint, dataset = Path(folder), Path(checkpoint), Path(dataset)
    folder.mkdir(parents=True, exist_ok=True)
    settings = settings or [(0, 1.0)] + [(k, a) for k in (1, 2, 3) for a in ORDERS]
    hashes = dict(checkpoint_sha256=sha(checkpoint), dataset_sha256=sha(dataset))
    core_hash = sha(ROOT / "RenyiSampler.py")
    rows = []
    for score_type in score_types:

        def score(q, t):
            with torch.no_grad():
                return (
                    target.log_score(q, abar[t])[1]
                    if score_type == "oracle"
                    else model(q, t)
                )

        for k, alpha in settings:
            for seed in seeds:
                run = folder / "runs" / score_type / f"K{k}_a{alpha:g}_s{seed}"
                run.mkdir(parents=True, exist_ok=True)
                config = dict(
                    version=1,
                    estimator=estimator,
                    score_type=score_type,
                    K=k,
                    alpha=alpha,
                    seed=seed,
                    samples=samples,
                    reference=reference_count,
                    snr=0.1,
                    clip=None,
                    step_caps=False,
                    fixed_inner_step=True,
                    predictor_variance="beta",
                    final_predictor_noise=False,
                    bandwidth_floor=bandwidth_floor,
                    bandwidth_factor=0.30,
                    core_sha256=core_hash,
                    **hashes,
                )
                saved_path = run / "result.pt"
                if saved_path.exists():
                    saved = torch.load(
                        saved_path, map_location="cpu", weights_only=False
                    )
                    if saved["config"] != config:
                        raise ValueError(f"Saved configuration mismatch: {run}")
                    rows.append(saved["metrics"])
                    continue
                # K0 and alpha1 contain no ratio evaluations; reuse identical phase-1 outputs.
                baseline = folder.parent / "paper_kde" / "runs" / score_type / run.name
                if (
                    estimator == "discriminator"
                    and (k == 0 or alpha == 1)
                    and (baseline / "result.pt").exists()
                ):
                    prior = torch.load(
                        baseline / "result.pt", map_location="cpu", weights_only=False
                    )
                    compare = {
                        key: value
                        for key, value in prior["config"].items()
                        if key != "estimator"
                    }
                    assert compare == {
                        key: value
                        for key, value in config.items()
                        if key != "estimator"
                    }
                    row = dict(
                        prior["metrics"], estimator=estimator, reused_from=str(baseline)
                    )
                    atomic_torch_save(
                        dict(
                            config=config,
                            metrics=row,
                            samples_from=str(baseline / "samples.pt"),
                        ),
                        saved_path,
                    )
                    rows.append(row)
                    continue
                ratio = None
                if estimator == "discriminator" and k and alpha != 1:
                    ratio = CurrentVectorRatio(target, abar, reference_count)

                def ratio_fn(x, t, substep):
                    if ratio is not None:
                        return ratio(x, t, substep)
                    variance = abar[t] * target.std**2 + 1 - abar[t]
                    bandwidth = max(bandwidth_floor, 0.30 * math.sqrt(float(variance)))
                    lp, sp = gaussian_kde_log_prob_score(
                        x.double(),
                        x[:reference_count].double(),
                        bandwidth,
                        1024,
                        reference_count,
                    )
                    lpi, spi = target.log_score(x, abar[t])
                    return (
                        lp - lpi.double(),
                        sp - spi.double(),
                        {"bandwidth": bandwidth},
                    )

                print("START", folder, score_type, k, alpha, seed, flush=True)
                started = time.perf_counter()
                try:
                    out, ref, trace = paper_transport_sample(
                        score,
                        ratio_fn,
                        beta,
                        abar,
                        (reference_count + samples, target.dim),
                        k,
                        alpha,
                        seed,
                        reference_count,
                    )
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    atomic_torch_save(
                        dict(
                            samples=out, reference=ref, diagnostics=trace, config=config
                        ),
                        run / "samples.pt",
                    )
                    row = metrics_fn(out)
                    row.update(
                        status="complete",
                        sampling_seconds=elapsed,
                        total_correction_time=sum(d["eta"] for d in trace),
                        max_mobility=max(
                            (d["mobility_max"] for d in trace), default=1.0
                        ),
                        max_mobility_gradient=max(
                            (d["gradient_max"] for d in trace), default=0.0
                        ),
                    )
                    plot_vector_run(
                        out,
                        target,
                        run / "generated.png",
                        f"{folder.parent.name}: {estimator}, {score_type}, K={k}, alpha={alpha}, seed={seed}",
                    )
                except (PaperNumericalFailure, FloatingPointError) as exc:
                    row = dict(
                        status="numerical_failure",
                        reason=str(exc),
                        failure_t=getattr(exc, "t", -1),
                        failure_substep=getattr(exc, "substep", -1),
                        sampling_seconds=time.perf_counter() - started,
                    )
                    payload = dict(
                        reason=str(exc),
                        config=config,
                        state=getattr(exc, "state", None),
                        diagnostics=getattr(exc, "diagnostics", []),
                    )
                    atomic_torch_save(payload, run / "failure.pt")
                row.update(
                    K=k,
                    alpha=alpha,
                    seed=seed,
                    score_type=score_type,
                    estimator=estimator,
                    samples=samples,
                )
                if ratio is not None:
                    atomic_torch_save(
                        dict(
                            state=ratio.model.state_dict(),
                            optimizer=ratio.opt.state_dict(),
                            target_rng=ratio.generator.get_state(),
                            updates=ratio.updates,
                            training_seed=SEED,
                            matches_final_law=False,
                        ),
                        run / "discriminator.pt",
                    )
                atomic_torch_save(dict(config=config, metrics=row), saved_path)
                rows.append(row)
                print("DONE", len(rows), row["status"], flush=True)

    expected = len(score_types) * len(settings) * len(seeds)
    assert len(rows) == expected
    atomic_torch_save(
        dict(
            complete=True,
            attempted=len(rows),
            succeeded=sum(r["status"] == "complete" for r in rows),
            failed=sum(r["status"] != "complete" for r in rows),
            source_sha256=core_hash,
            **hashes,
        ),
        folder / "completion.pt",
    )
    print("SWEEP_COMPLETE", folder, len(rows), flush=True)
