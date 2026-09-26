"""Train and sample stacked-MNIST diffusion experiments."""

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import MNIST_utils as U
from RenyiSampler import image_corrector, image_predictor
from sampling_utils import (
    ROOT,
    SEED,
    SEEDS,
    ORDERS,
    activate_workspace,
    atomic_torch_save,
    gen,
    save_json,
    seed_all,
    sha,
)


def train_models(args):
    activate_workspace()
    seed_all()
    torch.backends.cudnn.benchmark = True
    U.prepare(args.train_size)
    data = U.StackedData()
    root = U.MNIST_ROOT
    seed_all()
    if (
        not (root / "ddpm.pt").exists()
        or torch.load(root / "ddpm.pt", map_location="cpu", weights_only=False)["step"]
        < args.steps
    ):
        model = U.UNet(U.T).cuda().to(memory_format=torch.channels_last)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        beta, abar = U.mnist_schedule()
        g = gen(SEED + 1)
        vg = gen(SEED + 2)
        vx = data.sample(1024, vg, test=True)
        vt = torch.randint(U.T, (len(vx),), device="cuda", generator=vg)
        ve = torch.randn(vx.shape, device="cuda", generator=vg)
        vx = (
            abar[vt, None, None, None].sqrt() * vx
            + (1 - abar[vt, None, None, None]).sqrt() * ve
        )
        start = time.perf_counter()
        rolling = 0.0
        start_step = 0
        row = {}
        if (root / "ddpm_latest.pt").exists():
            saved = torch.load(
                root / "ddpm_latest.pt", map_location="cuda", weights_only=False
            )
            if saved["dataset_sha256"] != sha(root / "data.pt"):
                raise ValueError("Resume checkpoint dataset mismatch")
            if saved.get("rare_probability", 0.01) != U.RARE_PROBABILITY:
                raise ValueError("Resume checkpoint prior mismatch")
            if (
                saved.get("planned_steps", args.steps) != args.steps
                or saved["batch"] != args.batch
            ):
                raise ValueError("Resume schedule or batch mismatch")
            model.load_state_dict(saved["model"])
            ema.load_state_dict(saved["ema"])
            opt.load_state_dict(saved["optimizer"])
            g.set_state(saved["rng"].cpu())
            start_step = saved["step"]
            state = saved

        U.configure_compile_cache()
        train_forward = torch.compile(model, mode="reduce-overhead")
        for step in range(start_step + 1, args.steps + 1):
            torch.compiler.cudagraph_mark_step_begin()
            for group in opt.param_groups:
                group["lr"] = min(1, step / 500) * (
                    2e-5 + 9e-5 * (1 + math.cos(math.pi * step / args.steps))
                )
            x = data.sample(args.batch, g)
            t = torch.randint(U.T, (len(x),), device="cuda", generator=g)
            e = torch.randn(x.shape, device="cuda", generator=g)
            xt = (
                abar[t, None, None, None].sqrt() * x
                + (1 - abar[t, None, None, None]).sqrt() * e
            ).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = train_forward(xt, t)
                loss = (pred.float() - e).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                decay = min(0.9999, (1 + step) / (10 + step))
                torch._foreach_lerp_(
                    list(ema.parameters()), list(model.parameters()), 1 - decay
                )
            rolling += loss.item()
            if step % 100 == 0:
                row = {
                    "step": step,
                    "loss": rolling / 100,
                    "seconds": time.perf_counter() - start,
                    "resume_from_step": start_step,
                }
                rolling = 0.0
                if step % 500 == 0:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = torch.cat(
                            [ema(q, t) for q, t in zip(vx.split(128), vt.split(128))]
                        )
                    row["validation_mse"] = (pred.float() - ve).square().mean().item()
                print(json.dumps({"event": "ddpm", **row}), flush=True)
            if step % 5000 == 0 or step == args.steps:
                state = {
                    "ema": ema.state_dict(),
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "rng": g.get_state(),
                    "seed": SEED,
                    "step": step,
                    "T": U.T,
                    "beta_start": 1e-4,
                    "beta_end": 0.02,
                    "image_size": 28,
                    "batch": args.batch,
                    "train_size": len(data.train_digits),
                    "rare_probability": U.RARE_PROBABILITY,
                    "planned_steps": args.steps,
                    "validation_mse": row.get("validation_mse"),
                    "training_loss": row.get("loss"),
                    "dataset_sha256": sha(root / "data.pt"),
                }
                torch.save(state, root / f"ddpm_step_{step}.pt")
                torch.save(state, root / "ddpm_latest.pt")
        torch.save(state, root / "ddpm.pt")
    U.train_classifier(data)


def pretrain_discriminator(levels=U.DEFAULT_CORRECTION_LEVELS):
    anchors = U.correction_anchors(levels)
    activate_workspace()
    seed_all()
    root = U.MNIST_ROOT
    if (root / "discriminator.pt").exists():
        saved = torch.load(
            root / "discriminator.pt", map_location="cpu", weights_only=False
        )
        if saved["anchors"] != anchors:
            raise ValueError(
                "Discriminator anchors differ from --levels; use a fresh output directory"
            )
        if saved["ddpm_sha256"] != sha(root / "ddpm.pt") or saved[
            "dataset_sha256"
        ] != sha(root / "data.pt"):
            raise ValueError("Discriminator model or dataset mismatch")
        return
    model = U.load_ddpm()
    data = U.StackedData()
    beta, abar = U.mnist_schedule()
    if not (root / "baseline_bank.pt").exists():
        bank, x = U.baseline_bank(model, levels=levels)
        torch.save(bank, root / "baseline_bank.pt")
        torch.save(x.cpu().half(), root / "baseline_bank_final.pt")
        result, _, _ = U.evaluate_modes(x, U.evaluator())
        torch.save(result, root / "baseline_bank_quality.pt")
    bank = torch.load(root / "baseline_bank.pt", map_location="cpu", weights_only=True)
    if sorted(bank) != anchors:
        raise ValueError(
            "Baseline bank anchors differ from --levels; use a fresh output directory"
        )

    d = U.RatioCNN(U.T).cuda()
    opt = torch.optim.Adam(d.parameters(), lr=2e-4)
    g = gen(SEED + 21)
    start = time.perf_counter()
    for step in range(1, 3001):
        t = anchors[(step - 1) % len(anchors)]
        idx = torch.randint(3072, (256,), generator=g, device="cuda")
        fake = bank[t][idx.cpu()].cuda().float()
        clean = data.sample(256, g)
        real = abar[t].sqrt() * clean + (1 - abar[t]).sqrt() * torch.randn(
            clean.shape, device="cuda", generator=g
        )
        loss, hf, hr = U.discriminator_loss(d, fake, real, t, regularize=step % 16 == 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), 5.0)
        opt.step()
        if step % 100 == 0:
            d.eval()
            with torch.no_grad():
                f = bank[t][3072:3584].cuda().float()
                r = data.sample(512, g, test=True)
                r = abar[t].sqrt() * r + (1 - abar[t]).sqrt() * torch.randn(
                    r.shape, device="cuda", generator=g
                )
                lf, ff, rr = U.discriminator_loss(d, f, r, t)
            row = {
                "step": step,
                "t": t,
                "train_loss": loss.item(),
                "heldout_bce": lf.item(),
                "heldout_accuracy": 0.5
                * ((ff > 0).float().mean() + (rr < 0).float().mean()).item(),
                "seconds": time.perf_counter() - start,
            }

            print(json.dumps({"event": "ratio_pretrain", **row}), flush=True)
            d.train()
    torch.save(
        {
            "state": d.state_dict(),
            "optimizer": opt.state_dict(),
            "seed": SEED,
            "steps": 3000,
            "ddpm_sha256": sha(root / "ddpm.pt"),
            "dataset_sha256": sha(root / "data.pt"),
            "anchors": anchors,
        },
        root / "discriminator.pt",
    )


def run_fidelity_check(samples=2000, seeds=SEEDS):
    """Paired predictor-only variance ablation using the frozen DDPM."""

    activate_workspace()
    seed_all()
    model = U.load_ddpm()
    classifier = U.evaluator()
    beta, abar = U.mnist_schedule()
    folder = ROOT / str(U.MNIST_ROOT / "fidelity_check")
    for variance in ["beta", "posterior"]:
        for seed in seeds:
            run = folder / "runs" / f"{variance}_s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            result_path = run / "metrics.json"
            if result_path.exists():

                continue
            x = torch.randn(samples, 3, 28, 28, device="cuda", generator=gen(seed))
            gp = gen(2000000 + seed)
            with torch.no_grad():
                for t in reversed(range(U.T)):
                    x = image_predictor(
                        model, x, t, beta, abar, gp, chunk=1024, variance=variance
                    )
                result, pred, confidence = U.evaluate_modes(x, classifier)
            result.update({"variance": variance, "seed": seed, "K": 0, "alpha": 1.0})
            torch.save(
                {
                    "samples": x.cpu().half(),
                    "predicted_modes": pred,
                    "digit_confidence": confidence,
                },
                run / "samples.pt",
            )
            U.save_sample_images(x, run)
            save_json(result_path, result)

            print(variance, seed, result["low_confidence_fraction"], flush=True)


@torch.no_grad()
def run_fidelity_study(samples=2048, seeds=(42, 43, 44)):
    """Paired frozen-checkpoint ablations; keep the main experiment unchanged."""

    activate_workspace()
    seed_all()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    U.configure_compile_cache()
    root = U.MNIST_ROOT
    folder = root / "fidelity_study"
    checkpoint = torch.load(root / "ddpm.pt", map_location="cuda", weights_only=False)
    model = (
        U.UNet(U.T)
        .cuda()
        .to(memory_format=torch.channels_last)
        .eval()
        .requires_grad_(False)
    )
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    classifier = U.evaluator()
    beta, abar = U.mnist_schedule()
    previous = torch.cat([torch.ones_like(abar[:1]), abar[:-1]])
    coef0 = beta * previous.sqrt() / (1 - abar)
    coeft = (1 - previous) * (1 - beta).sqrt() / (1 - abar)
    posterior = beta * (1 - previous) / (1 - abar)
    configs = [
        ("ema_beta_bf16", "ema", "beta", False, "bf16"),
        ("ema_posterior_bf16", "ema", "posterior", False, "bf16"),
        ("ema_beta_clip_bf16", "ema", "beta", True, "bf16"),
        ("ema_posterior_clip_bf16", "ema", "posterior", True, "bf16"),
        ("ema_beta_fp32", "ema", "beta", False, "fp32"),
        ("raw_beta_bf16", "model", "beta", False, "bf16"),
    ]
    for name, weights, variance, clip, precision in configs:
        model.load_state_dict(checkpoint[weights])
        for seed in seeds:
            run = folder / "runs" / f"{name}_s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            if (run / "metrics.json").exists():
                previous_result = json.loads((run / "metrics.json").read_text())
                if previous_result["samples"] != samples or previous_result[
                    "checkpoint_sha256"
                ] != sha(root / "ddpm.pt"):
                    raise ValueError(
                        "Existing fidelity result has a different sample count or checkpoint"
                    )

                continue
            x = torch.randn(samples, 3, 28, 28, device="cuda", generator=gen(seed))
            gp = gen(2000000 + seed)
            begin = time.perf_counter()
            for t in reversed(range(U.T)):
                parts = []
                for q in x.split(1024):
                    ti = torch.full((len(q),), t, device="cuda", dtype=torch.long)
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
                    ):
                        eps = (
                            compiled(
                                q.contiguous(memory_format=torch.channels_last), ti
                            )
                            .clone()
                            .float()
                        )
                    parts.append(eps)
                eps = torch.cat(parts)
                if clip:
                    x0 = ((x - (1 - abar[t]).sqrt() * eps) / abar[t].sqrt()).clamp(
                        -1, 1
                    )
                    mean = coef0[t] * x0 + coeft[t] * x
                else:
                    mean = (x - beta[t] * eps / (1 - abar[t]).sqrt()) / (
                        1 - beta[t]
                    ).sqrt()
                if t:
                    var = beta[t] if variance == "beta" else posterior[t]
                    mean = mean + var.sqrt() * torch.randn(
                        x.shape, device=x.device, generator=gp
                    )
                x = mean
            result, pred, confidence = U.evaluate_modes(x, classifier)
            result.update(
                {
                    "config": name,
                    "seed": seed,
                    "weights": weights,
                    "variance": variance,
                    "clip_x0": clip,
                    "precision": precision,
                    "sampling_seconds": time.perf_counter() - begin,
                    "pixel_out_of_range_fraction": ((x < -1) | (x > 1))
                    .float()
                    .mean()
                    .item(),
                    "channel_low_confidence": (confidence < 0.9)
                    .float()
                    .mean(0)
                    .tolist(),
                    "checkpoint_sha256": sha(root / "ddpm.pt"),
                }
            )
            torch.save(
                {
                    "samples": x.cpu().half(),
                    "predicted_modes": pred,
                    "digit_confidence": confidence,
                },
                run / "samples.pt",
            )
            U.save_sample_images(x, run)
            save_json(run / "metrics.json", result)

            print(
                json.dumps(
                    {
                        "config": name,
                        "seed": seed,
                        "low_confidence": result["low_confidence_fraction"],
                        "confidence_tv": result["confidence_filtered_tv"],
                        "seconds": result["sampling_seconds"],
                    }
                ),
                flush=True,
            )


def run_capacity_pilot(steps=50000, base=64, architecture="plain"):
    """Isolate denoiser width using the existing dataset and training schedule."""

    activate_workspace()
    seed_all()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    U.configure_compile_cache()
    root = U.MNIST_ROOT
    folder = root / "fidelity_capacity"
    folder.mkdir(exist_ok=True)
    data = U.StackedData()
    model = (
        U.make_fidelity_model(base, architecture)
        .cuda()
        .to(memory_format=torch.channels_last)
    )
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    beta, abar = U.mnist_schedule()
    g = gen(SEED + 1)
    vg = gen(SEED + 2)
    vx = data.sample(1024, vg, test=True)
    vt = torch.randint(U.T, (len(vx),), device="cuda", generator=vg)
    ve = torch.randn(vx.shape, device="cuda", generator=vg)
    vx = (
        abar[vt, None, None, None].sqrt() * vx
        + (1 - abar[vt, None, None, None]).sqrt() * ve
    )
    start_step = 0
    prefix = f"width{base}" if architecture == "plain" else f"attention{base}"
    latest = folder / f"{prefix}_latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location="cuda", weights_only=False)
        assert state["dataset_sha256"] == sha(root / "data.pt")
        assert state["base"] == base and state["seed"] == SEED
        assert state.get("architecture", "plain") == architecture
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        g.set_state(state["rng"].cpu())
        start_step = state["step"]
    compiled = torch.compile(model, mode="reduce-overhead")
    rolling = 0.0
    begin = time.perf_counter()
    for step in range(start_step + 1, steps + 1):
        torch.compiler.cudagraph_mark_step_begin()
        lr = min(1, step / 500) * (
            2e-5 + 9e-5 * (1 + math.cos(math.pi * step / 200000))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        x = data.sample(128, g)
        t = torch.randint(U.T, (len(x),), device="cuda", generator=g)
        eps = torch.randn(x.shape, device="cuda", generator=g)
        xt = (
            abar[t, None, None, None].sqrt() * x
            + (1 - abar[t, None, None, None]).sqrt() * eps
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = compiled(xt.contiguous(memory_format=torch.channels_last), t)
            loss = (pred.float() - eps).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            decay = min(0.9999, (1 + step) / (10 + step))
            torch._foreach_lerp_(
                list(ema.parameters()), list(model.parameters()), 1 - decay
            )
        rolling += loss.item()
        if step % 1000 == 0:
            print(
                json.dumps(
                    {
                        "width": base,
                        "step": step,
                        "loss": rolling / 1000,
                        "seconds": time.perf_counter() - begin,
                    }
                ),
                flush=True,
            )
            rolling = 0.0
        if step % 10000 == 0 or step == steps:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred = torch.cat(
                    [ema(q, tq) for q, tq in zip(vx.split(128), vt.split(128))]
                )
                validation = (pred.float() - ve).square().mean().item()
            state = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": g.get_state(),
                "step": step,
                "base": base,
                "architecture": architecture,
                "T": U.T,
                "seed": SEED,
                "beta_start": 1e-4,
                "beta_end": 0.02,
                "image_size": 28,
                "batch": 128,
                "train_size": len(data.train_digits),
                "training_schedule_steps": 200000,
                "validation_mse": validation,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "dataset_sha256": sha(root / "data.pt"),
            }
            torch.save(state, folder / f"{prefix}_step{step}.pt")
            torch.save(state, latest)
            print(
                json.dumps({"width": base, "step": step, "validation_mse": validation}),
                flush=True,
            )
    print("CAPACITY_TRAINING_COMPLETE", flush=True)


@torch.no_grad()
def evaluate_capacity_checkpoint(
    name, checkpoint_path, base, samples=2048, seeds=(42, 43, 44)
):
    """Evaluate a width ablation with exactly the original uncorrected sampler."""

    activate_workspace()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    U.configure_compile_cache()
    path = Path(checkpoint_path)
    state = torch.load(path, map_location="cuda", weights_only=False)
    assert state["dataset_sha256"] == sha(U.MNIST_ROOT / "data.pt")
    architecture = state.get("architecture", "plain")
    assert state.get("base", 32) == base
    model = (
        U.make_fidelity_model(base, architecture)
        .cuda()
        .to(memory_format=torch.channels_last)
        .eval()
        .requires_grad_(False)
    )
    model.load_state_dict(state["ema"])
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    def predict(x, t):
        torch.compiler.cudagraph_mark_step_begin()
        ti = torch.full((len(x),), t, device=x.device, dtype=torch.long)
        return compiled(x, ti).clone()

    classifier = U.evaluator()
    beta, abar = U.mnist_schedule()
    folder = U.MNIST_ROOT / "fidelity_capacity"
    for seed in seeds:
        run = folder / "runs" / f"{name}_s{seed}"
        run.mkdir(parents=True, exist_ok=True)
        if (run / "metrics.json").exists():
            previous_result = json.loads((run / "metrics.json").read_text())
            if previous_result["samples"] != samples or previous_result[
                "checkpoint_sha256"
            ] != sha(path):
                raise ValueError(
                    "Use a new run name for a different sample count or checkpoint"
                )
            continue
        x = torch.randn(samples, 3, 28, 28, device="cuda", generator=gen(seed))
        g = gen(2000000 + seed)
        begin = time.perf_counter()
        for t in reversed(range(U.T)):
            x = image_predictor(predict, x, t, beta, abar, g, chunk=1024)
        metrics, pred, confidence = U.evaluate_modes(x, classifier)
        metrics.update(
            {
                "config": name,
                "seed": seed,
                "base": base,
                "architecture": architecture,
                "training_step": state["step"],
                "checkpoint_sha256": sha(path),
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "sampling_seconds": time.perf_counter() - begin,
                "channel_low_confidence": (confidence < 0.9).float().mean(0).tolist(),
            }
        )
        torch.save(
            {
                "samples": x.cpu().half(),
                "predicted_modes": pred,
                "digit_confidence": confidence,
            },
            run / "samples.pt",
        )
        U.save_sample_images(x, run)
        save_json(run / "metrics.json", metrics)

        print(
            json.dumps(
                {
                    "config": name,
                    "seed": seed,
                    "low_confidence": metrics["low_confidence_fraction"],
                    "confidence_tv": metrics["confidence_filtered_tv"],
                    "rare_mass": metrics["rare_mass"],
                }
            ),
            flush=True,
        )
    print("CAPACITY_EVALUATION_COMPLETE", name, flush=True)


def run_sampling(samples=10000, seeds=SEEDS, levels=U.DEFAULT_CORRECTION_LEVELS):
    anchors = U.correction_anchors(levels)
    if samples < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(
            "samples must be positive and seeds must be nonempty and unique"
        )
    activate_workspace()
    seed_all()
    torch.backends.cudnn.benchmark = True
    root = U.MNIST_ROOT
    folder = root / f"correction_levels{levels}"
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha(root / "ddpm.pt")
    initial = torch.load(
        root / "discriminator.pt", map_location="cpu", weights_only=False
    )
    if initial["anchors"] != anchors:
        raise ValueError("Discriminator anchors differ from sampling --levels")
    assert initial["ddpm_sha256"] == checkpoint_hash
    assert initial["dataset_sha256"] == sha(root / "data.pt")
    model = U.load_ddpm()
    classifier = U.evaluator()
    data = U.StackedData()
    beta, abar = U.mnist_schedule()
    m = 1024
    rows = []
    settings = [(0, 1.0)] + [(k, a) for k in (1, 2, 3) for a in ORDERS]
    for k, alpha in settings:
        for seed in seeds:
            run = folder / "runs" / f"K{k}_a{alpha:g}_s{seed}"
            run.mkdir(parents=True, exist_ok=True)
            config = dict(
                version=1,
                rare_probability=U.RARE_PROBABILITY,
                K=k,
                alpha=alpha,
                seed=seed,
                levels=anchors,
                samples=samples,
                reference=m,
                checkpoint_sha256=checkpoint_hash,
                discriminator_sha256=sha(root / "discriminator.pt"),
                sampler_sha256=sha(ROOT / "RenyiSampler.py"),
                dataset_sha256=initial["dataset_sha256"],
                snr=0.08,
                clip=4.0,
                step_caps=True,
                predictor_variance="beta",
            )
            if (run / "result.pt").exists():
                old = torch.load(
                    run / "result.pt", map_location="cpu", weights_only=False
                )
                assert old["config"] == config
                rows.append(old["metrics"])
                continue
            print("MNIST_START", k, alpha, seed, flush=True)
            g0, gc, gp, gd = (
                gen(seed),
                gen(1000000 + seed),
                gen(2000000 + seed),
                gen(4000000 + seed),
            )
            x = torch.randn(m + samples, 3, 28, 28, device="cuda", generator=g0)
            critic = U.RatioCNN(U.T).cuda()
            critic.load_state_dict(initial["state"])
            critic.eval().requires_grad_(False)
            adapt = U.online_adapter(data, abar, gd)
            trace = []
            begun = time.perf_counter()
            try:
                for t in reversed(range(U.T)):
                    if k and t in anchors:
                        x, diag = image_corrector(
                            model,
                            critic,
                            x,
                            t,
                            beta,
                            abar,
                            k,
                            alpha,
                            gc,
                            m,
                            adapt,
                            chunk=1024,
                        )
                        trace.extend(diag)
                    x = image_predictor(model, x, t, beta, abar, gp, chunk=1024)
                    if not torch.isfinite(x).all():
                        raise FloatingPointError(f"Nonfinite MNIST particles at t={t}")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begun
                row, pred, conf = U.evaluate_modes(x[m:], classifier)
                row.update(
                    status="complete",
                    sampling_seconds=elapsed,
                    total_correction_time=sum(d["eta"] for d in trace),
                    step_capped_fraction=(
                        float(np.mean([d["step_capped"] for d in trace]))
                        if trace
                        else 0.0
                    ),
                    weight_clipped_fraction=(
                        float(np.mean([d["clipped_fraction"] for d in trace]))
                        if trace
                        else 0.0
                    ),
                )
                outputs = x[m:].cpu()
                atomic_torch_save(
                    dict(
                        samples=outputs.half(),
                        reference=x[:m].cpu(),
                        predicted_modes=pred,
                        digit_confidence=conf,
                        diagnostics=trace,
                        config=config,
                    ),
                    run / "samples.pt",
                )
                U.save_sample_images(outputs, run)
            except FloatingPointError as exc:
                row = dict(
                    status="numerical_failure",
                    reason=str(exc),
                    failure_t=t,
                    sampling_seconds=time.perf_counter() - begun,
                )
                atomic_torch_save(
                    dict(
                        state=x.cpu(), diagnostics=trace, config=config, reason=str(exc)
                    ),
                    run / "failure.pt",
                )
            row.update(
                K=k,
                alpha=alpha,
                seed=seed,
                score_type="learned",
                levels=levels,
                samples=samples,
            )
            atomic_torch_save(dict(config=config, metrics=row), run / "result.pt")
            if k and alpha != 1:
                atomic_torch_save(
                    dict(
                        state=critic.state_dict(),
                        config=config,
                        matches_final_law=False,
                    ),
                    run / "discriminator.pt",
                )
            rows.append(row)

            print("MNIST_DONE", len(rows), row["status"], flush=True)
    assert len(rows) == len(settings) * len(seeds)
    atomic_torch_save(
        dict(
            complete=True,
            attempted=len(rows),
            succeeded=sum(r["status"] == "complete" for r in rows),
            failed=sum(r["status"] != "complete" for r in rows),
        ),
        folder / "completion.pt",
    )
    print("MNIST_SWEEP_COMPLETE", len(rows), flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/MNIST"))
    parser.add_argument(
        "action",
        choices=[
            "data",
            "train",
            "discriminator",
            "sample",
            "all",
            "test",
            "fidelity",
            "fidelity-study",
            "capacity-train",
            "capacity-sample",
        ],
        default="all",
        nargs="?",
    )
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--train-size", type=int, default=1000000)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--levels",
        type=int,
        default=U.DEFAULT_CORRECTION_LEVELS,
        help="anchors for discriminator/sample/all (default: 64)",
    )
    parser.add_argument("--base", type=int)
    parser.add_argument("--architecture", choices=["plain", "attention"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-name")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    U.configure_experiment(args)
    if not 1 <= args.levels <= U.T:
        parser.error(f"--levels must be between 1 and {U.T}")
    if args.samples < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--samples must be positive and --seeds must be unique")
    if (
        args.checkpoint is not None or args.run_name is not None
    ) and args.action != "capacity-sample":
        parser.error("--checkpoint and --run-name apply only to capacity-sample")
    if args.base is not None and args.action not in (
        "capacity-train",
        "capacity-sample",
    ):
        parser.error("--base applies only to capacity-train or capacity-sample")
    if args.architecture is not None and args.action != "capacity-train":
        parser.error("--architecture applies only to capacity-train")
    args.base = 64 if args.base is None else args.base
    args.architecture = args.architecture or "plain"
    activate_workspace()
    if args.action == "test":
        U.validate_mnist()
    elif args.action == "data":
        U.prepare(args.train_size)
    elif args.action == "fidelity":
        run_fidelity_check(samples=args.samples, seeds=args.seeds)
    elif args.action == "fidelity-study":
        run_fidelity_study(samples=args.samples, seeds=args.seeds)
    elif args.action == "capacity-train":
        run_capacity_pilot(
            steps=args.steps, base=args.base, architecture=args.architecture
        )
    elif args.action == "capacity-sample":
        if args.checkpoint is None or args.run_name is None:
            parser.error("capacity-sample requires --checkpoint and --run-name")
        evaluate_capacity_checkpoint(
            args.run_name, args.checkpoint, args.base, args.samples, args.seeds
        )
    else:
        if args.action in ("train", "all"):
            train_models(args)
        if args.action in ("discriminator", "all"):
            pretrain_discriminator(levels=args.levels)
        if args.action in ("sample", "all"):
            run_sampling(samples=args.samples, seeds=args.seeds, levels=args.levels)


if __name__ == "__main__":
    main()
