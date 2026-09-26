import math, time
import torch
import numpy as np
from sampling_utils import gen


def kde_ratio(x, ref, target, abar, bandwidth):
    # Full-dimensional KDE: no claim of accuracy in high dimension.
    logs = []
    scores = []
    h2 = bandwidth**2
    m = len(ref)
    for offset in range(0, len(x), 1024):
        q = x[offset : offset + 1024]
        d2 = (
            q.square().sum(1)[:, None] + ref.square().sum(1)[None] - 2 * q @ ref.T
        ).clamp_min(0)
        lc = -d2 / (2 * h2)
        count = torch.full((len(q),), m, device=x.device, dtype=x.dtype)
        upto = min(len(q), max(0, m - offset))
        if upto:
            row = torch.arange(upto, device=x.device)
            lc[row, row + offset] = -torch.inf
            count[row] = m - 1
        logs.append(
            torch.logsumexp(lc, 1)
            - count.log()
            - target.dim * math.log(bandwidth * math.sqrt(2 * math.pi))
        )
        scores.append((lc.softmax(1) @ ref - q) / h2)
    lp, sp = target.log_score(x, abar)
    return torch.cat(logs) - lp, torch.cat(scores) - sp


def mobility(h, dh, alpha, m, clip=4.0):
    flat = [-1] + [1] * (dh.ndim - 1)
    if alpha == 1:
        return torch.ones_like(h), torch.zeros_like(dh), 0.0
    z = (alpha - 1) * h
    loga = z - (torch.logsumexp(z[:m], 0) - math.log(m))
    active = loga.abs() < math.log(clip)
    b = loga.clamp(-math.log(clip), math.log(clip)).exp()
    a = b / b[:m].mean()
    da = a.reshape(flat) * (alpha - 1) * dh * active.reshape(flat)
    return a, da, (~active).float().mean().item()


@torch.inference_mode()
def vector_sample(model, target, beta, abar, k, alpha, seed, score_type, N=5000, M=512):
    device = "cuda"
    dim = target.dim
    g0, gc, gp = gen(seed), gen(1000000 + seed), gen(2000000 + seed)
    x = torch.randn((M + N, dim), device=device, generator=g0)

    def score(q, t):
        return (
            target.log_score(q, abar[t])[1]
            if score_type == "oracle"
            else -model(q, t) / (1 - abar[t]).sqrt()
        )

    diag = []
    trace = []
    start = time.perf_counter()
    for t in range(len(beta) - 1, -1, -1):
        var = abar[t] * target.std**2 + 1 - abar[t]
        if k:
            base_score = score(x, t)
            expected_z = math.sqrt(2) * math.exp(
                math.lgamma((dim + 1) / 2) - math.lgamma(dim / 2)
            )
            nominal = (
                2
                * (1 - beta[t])
                * (0.1 * expected_z / base_score[:M].norm(dim=1).mean().clamp_min(1e-8))
                ** 2
                / k
            )
            bridge = torch.randn(6, len(x), dim, device=device, generator=gc)
            bw = max(target.std * 0.45, 0.30 * math.sqrt(var.item()))
            for j in range(k):
                sc = base_score if j == 0 else score(x, t)
                if alpha == 1:
                    a = torch.ones(len(x), device=device)
                    da = torch.zeros_like(x)
                    clipped = 0.0
                else:
                    h, dh = kde_ratio(x, x[:M], target, abar[t], bw)
                    a, da, clipped = mobility(h, dh, alpha, M)
                drift = a[:, None] * sc + da
                eta = torch.minimum(nominal, 0.1 * var / a.max())
                eta = torch.minimum(
                    eta,
                    0.25
                    * torch.sqrt(var * dim)
                    / drift.norm(dim=1).max().clamp_min(1e-8),
                )
                z = bridge[j * (6 // k) : (j + 1) * (6 // k)].sum(0) / math.sqrt(6 // k)
                x = x + eta * drift + (2 * eta * a[:, None]).sqrt() * z
                diag.append(
                    [
                        eta.item(),
                        float(eta < nominal * 0.999999),
                        clipped,
                        a.max().item(),
                        da.norm(dim=1).max().item(),
                    ]
                )
        sc = score(x, t)
        x = (x + beta[t] * sc) / (1 - beta[t]).sqrt()
        if t:
            x += beta[t].sqrt() * torch.randn(x.shape, device=device, generator=gp)
        if t % 16 == 0:
            if not torch.isfinite(x).all():
                raise FloatingPointError("Nonfinite geometry samples")
            trace.append({"t": t, "max_norm": x.norm(dim=1).max().item()})
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    out = x[M:].cpu()
    metrics = target.metrics(out)
    d = np.array(diag) if diag else np.zeros((1, 5))
    metrics.update(
        {
            "K": k,
            "alpha": alpha,
            "seed": seed,
            "score_type": score_type,
            "samples": N,
            "reference": M,
            "sampling_seconds": elapsed,
            "total_correction_time": float(d[:, 0].sum()),
            "step_capped_fraction": float(d[:, 1].mean()),
            "weight_clipped_fraction": float(d[:, 2].mean()),
            "maximum_mobility": float(d[:, 3].max()),
            "maximum_mobility_gradient": float(d[:, 4].max()),
        }
    )
    return metrics, out, x[:M].cpu(), trace


# ---------- Stochastic image corrector and ordinary DDPM predictor ----------
def image_score(model, x, t, abar, chunk=256):
    parts = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for q in x.split(chunk):
            parts.append(
                -model(q.contiguous(memory_format=torch.channels_last), t).float()
                / (1 - abar[t]).sqrt()
            )
    return torch.cat(parts)


def image_predictor(model, x, t, beta, abar, g, chunk=256, variance="beta"):
    """The supplied epsilon-prediction DDPM update, without x0 clipping."""
    if variance not in {"beta", "posterior"}:
        raise ValueError("variance must be beta or posterior")
    score = image_score(model, x, t, abar, chunk)
    mean = (x + beta[t] * score) / (1 - beta[t]).sqrt()
    if t:
        noise_variance = beta[t]
        if variance == "posterior":
            noise_variance = beta[t] * (1 - abar[t - 1]) / (1 - abar[t])
        mean += noise_variance.sqrt() * torch.randn(
            x.shape, device=x.device, generator=g
        )
    return mean


def image_ratio_gradient(discriminator, x, t, chunk=256):
    logs = []
    grads = []
    discriminator.eval().requires_grad_(False)
    for q in x.split(chunk):
        with torch.enable_grad():
            q = q.detach().requires_grad_(True)
            h = discriminator(q, t)
            dh = torch.autograd.grad(h.sum(), q)[0]
        logs.append(h.detach())
        grads.append(dh.detach())
    return torch.cat(logs), torch.cat(grads)


def image_corrector(
    model, discriminator, x, t, beta, abar, k, alpha, g, M, adapt, chunk=256
):
    dim = x[0].numel()
    initial = image_score(model, x, t, abar, chunk)
    nominal = (
        2
        * (1 - beta[t])
        * (
            0.08
            * math.sqrt(dim)
            / initial[:M].flatten(1).norm(dim=1).mean().clamp_min(1e-8)
        )
        ** 2
        / k
    )
    bridge = torch.randn((6, *x.shape), device=x.device, generator=g)
    diagnostics = []
    for j in range(k):
        fitting = adapt(discriminator, x[:M].detach(), t, j) if alpha != 1 else {}
        if alpha == 1:
            a = torch.ones(len(x), device=x.device)
            da = torch.zeros_like(x)
            clipped = 0.0
        else:
            h, dh = image_ratio_gradient(discriminator, x, t, chunk)
            a, da, clipped = mobility(h, dh, alpha, M)
        sc = initial if j == 0 else image_score(model, x, t, abar, chunk)
        drift = a[:, None, None, None] * sc + da
        localvar = (1 - abar[t]).clamp_min(1e-4)
        eta = torch.minimum(nominal, 0.05 * localvar / a.max())
        eta = torch.minimum(
            eta,
            0.1
            * localvar.sqrt()
            / (drift.flatten(1).square().mean(1).sqrt().max().clamp_min(1e-8)),
        )
        z = bridge[j * (6 // k) : (j + 1) * (6 // k)].sum(0) / math.sqrt(6 // k)
        x = x + eta * drift + (2 * eta * a[:, None, None, None]).sqrt() * z
        diagnostics.append(
            {
                "t": t,
                "substep": j,
                "eta": eta.item(),
                "nominal_eta": nominal.item(),
                "step_capped": float(eta < nominal * 0.999999),
                "clipped_fraction": clipped,
                "max_mobility": a.max().item(),
                "max_mobility_gradient": da.flatten(1).norm(dim=1).max().item(),
                **fitting,
            }
        )
    return x.detach(), diagnostics


def gaussian_kde_log_prob_score(
    x, reference, bandwidth, chunk_size=1024, leave_reference_out=0
):
    """Isotropic Gaussian KDE, with an analytic spatial gradient.

    If the first M queries are the reference points, leave_reference_out=M
    removes the diagonal kernel for those queries. Reference centers are fixed.
    """
    m, d = reference.shape
    h2 = bandwidth**2
    logs, scores = [], []
    ref_sq = reference.square().sum(-1)[None, :]
    for start in range(0, len(x), chunk_size):
        q = x[start : start + chunk_size]
        dist2 = (q.square().sum(-1)[:, None] + ref_sq - 2 * q @ reference.T).clamp_min(
            0
        )
        logits = -dist2 / (2 * h2)
        counts = torch.full((len(q),), m, dtype=q.dtype, device=q.device)
        limit = min(start + len(q), leave_reference_out)
        if limit > start:
            local = torch.arange(limit - start, device=q.device)
            logits[local, torch.arange(start, limit, device=q.device)] = -torch.inf
            counts[local] = m - 1
        lp = (
            torch.logsumexp(logits, 1)
            - counts.log()
            - d * math.log(bandwidth * math.sqrt(2 * math.pi))
        )
        probs = torch.softmax(logits, dim=1)
        score = (probs @ reference - q) / h2
        logs.append(lp)
        scores.append(score)
    return torch.cat(logs), torch.cat(scores)


def gaussian_mobility_from_log_ratio(
    log_ratio,
    ratio_gradient,
    order,
    reference_count,
    clip=4.0,
    prefactor="unit",
    frozen_log_normalizer=None,
    frozen_clip_normalizer=None,
):
    """Clip normalized positive mobility, then renormalize over references.

    Returns the exact spatial derivative of this modified mobility, treating
    both normalizers as constants. Hard clipping has zero derivative outside.
    """
    if order <= 0:
        raise ValueError("Renyi order must be positive")
    z = (order - 1) * log_ratio
    log_g = (
        torch.logsumexp(z[:reference_count], 0) - math.log(reference_count)
        if frozen_log_normalizer is None
        else frozen_log_normalizer
    )
    log_a = z - log_g
    if clip is None:
        b = log_a.exp()
        active = torch.ones_like(log_a, dtype=torch.bool)
    else:
        if clip < 1:
            raise ValueError("clip must be >= 1")
        bound = math.log(clip)
        active = (log_a > -bound) & (log_a < bound)
        b = log_a.clamp(-bound, bound).exp()
    norm = (
        b[:reference_count].mean()
        if frozen_clip_normalizer is None
        else frozen_clip_normalizer
    )
    a = b / norm
    grad_a = a[:, None] * (order - 1) * ratio_gradient * active[:, None]
    if prefactor == "renyi":
        a, grad_a = order * a, order * grad_a
    elif prefactor != "unit":
        raise ValueError(prefactor)
    stats = {
        "clipped_fraction": (~active).float().mean(),
        "mobility_mean": a[:reference_count].mean(),
        "mobility_max": a.max(),
        "mobility_min": a.min(),
        "grad_mobility_max": grad_a.norm(dim=1).max(),
        "log_normalizer": log_g,
        "clip_normalizer": norm,
    }
    return a, grad_a, stats


def gaussian_kde_mobility(
    x,
    reference_count,
    mixture,
    abar,
    order,
    bandwidth,
    clip=4.0,
    prefactor="unit",
    chunk_size=1024,
):
    if order == 1:
        a = torch.ones(len(x), device=x.device, dtype=x.dtype)
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        return (
            a,
            torch.zeros_like(x),
            {
                "clipped_fraction": zero,
                "mobility_mean": zero + 1,
                "mobility_max": zero + 1,
                "mobility_min": zero + 1,
                "grad_mobility_max": zero,
            },
        )
    lp, sr = gaussian_kde_log_prob_score(
        x, x[:reference_count], bandwidth, chunk_size, reference_count
    )
    lpi, spi = mixture.log_prob_score(x, abar)
    return gaussian_mobility_from_log_ratio(
        lp - lpi, sr - spi, order, reference_count, clip, prefactor
    )


def gaussian_ancestral_predictor(x, score, beta, noise, final=False):
    mean = (x + beta * score) / torch.sqrt(1 - beta)
    return mean if final else mean + beta.sqrt() * noise


def gaussian_stable_step_size(
    nominal, x, score, a, grad_a, local_variance, noise_cap=0.10, drift_cap=0.25
):
    """One shared scalar step, chosen before fresh increment realization.

    Caps eta*max(A) and maximum drift displacement. These are numerical
    safeguards, not a general stability theorem or a manifold projection.
    """
    drift = a[:, None] * score + grad_a
    eta_noise = noise_cap * local_variance / a.max().clamp_min(1e-12)
    eta_drift = (
        drift_cap
        * torch.sqrt(local_variance)
        / drift.norm(dim=1).max().clamp_min(1e-12)
    )
    eta = torch.minimum(torch.minimum(nominal, eta_noise), eta_drift)
    return eta, drift


# Paper-faithful uncapped corrector. Existing safeguarded samplers stay unchanged.
class PaperNumericalFailure(FloatingPointError):
    def __init__(self, reason, t, substep, state, diagnostics):
        super().__init__(reason)
        self.t = t
        self.substep = substep
        self.state = state.detach().cpu()
        self.diagnostics = diagnostics


def paper_mobility(log_ratio, ratio_gradient, order, reference_count):
    """Unclipped mean-one mobility; reference normalization is spatially frozen."""
    if order == 1.0:
        return torch.ones_like(log_ratio), torch.zeros_like(ratio_gradient)
    z = (order - 1.0) * log_ratio.double()
    log_g = torch.logsumexp(z[:reference_count], 0) - math.log(reference_count)
    a = (z - log_g).exp()
    shape = (-1,) + (1,) * (ratio_gradient.ndim - 1)
    da = a.reshape(shape) * (order - 1.0) * ratio_gradient.double()
    return a, da


def paper_nominal_step(score, reference_count, retention, k, snr):
    d = score[0].numel()
    expected_norm = math.sqrt(2.0) * math.exp(
        math.lgamma((d + 1) / 2) - math.lgamma(d / 2)
    )
    norm = score[:reference_count].double().flatten(1).norm(dim=1).mean()
    return (
        2 * retention.double() / k * (snr * expected_norm / norm.clamp_min(1e-8)) ** 2
    )


def paper_em_update(x, score, a, da, eta, noise):
    shape = (-1,) + (1,) * (x.ndim - 1)
    value = x.double() + eta * (a.reshape(shape) * score.double() + da)
    value = value + (2 * eta * a.reshape(shape)).sqrt() * noise.double()
    return value.to(x.dtype)


@torch.no_grad()
def paper_transport_sample(
    score_fn,
    ratio_fn,
    beta,
    abar,
    shape,
    k,
    order,
    seed,
    reference_count,
    snr=0.1,
    levels=None,
):
    """Fixed nominal step per level, no mobility or drift/noise safeguards."""
    if k not in (0, 1, 2, 3):
        raise ValueError("Configured experiment supports K=0,1,2,3")
    device = beta.device
    gi = gen(seed, device)
    gc = gen(1000000 + seed, device)
    gp = gen(2000000 + seed, device)
    x = torch.randn(shape, device=device, generator=gi)
    levels = set(range(len(beta))) if levels is None else set(levels)
    diagnostics = []
    for t in reversed(range(len(beta))):
        if k and t in levels:
            initial = score_fn(x, t)
            eta = paper_nominal_step(initial, reference_count, 1 - beta[t], k, snr)
            if not torch.isfinite(eta) or eta <= 0:
                raise PaperNumericalFailure(
                    "nonfinite or nonpositive nominal step", t, -1, x, diagnostics
                )
            bridge = torch.randn((6, *shape), device=device, generator=gc)
            block = 6 // k
            for j in range(k):
                sc = initial if j == 0 else score_fn(x, t)
                extra = {}
                if order == 1:
                    a = torch.ones(len(x), device=device, dtype=torch.float64)
                    da = torch.zeros_like(x, dtype=torch.float64)
                else:
                    h, dh, extra = ratio_fn(x, t, j)
                    a, da = paper_mobility(h, dh, order, reference_count)
                if (
                    not torch.isfinite(sc).all()
                    or not torch.isfinite(a).all()
                    or not torch.isfinite(da).all()
                    or not (a > 0).all()
                ):
                    raise PaperNumericalFailure(
                        "nonfinite score/mobility/gradient or mobility underflow",
                        t,
                        j,
                        x,
                        diagnostics,
                    )
                noise = bridge[j * block : (j + 1) * block].sum(0) / math.sqrt(block)
                nxt = paper_em_update(x, sc, a, da, eta, noise)
                diagnostics.append(
                    dict(
                        t=t,
                        substep=j,
                        eta=float(eta),
                        mobility_mean=float(a[:reference_count].mean()),
                        mobility_min=float(a.min()),
                        mobility_max=float(a.max()),
                        gradient_max=float(da.flatten(1).norm(dim=1).max()),
                        displacement_rms=float(
                            (nxt.double() - x).square().mean().sqrt()
                        ),
                        **extra
                    )
                )
                if not torch.isfinite(nxt).all():
                    raise PaperNumericalFailure(
                        "nonfinite Euler-Maruyama update", t, j, nxt, diagnostics
                    )
                x = nxt
        sc = score_fn(x, t)
        x = (x + beta[t] * sc) / (1 - beta[t]).sqrt()
        if t:
            x = x + beta[t].sqrt() * torch.randn(x.shape, device=device, generator=gp)
        if not torch.isfinite(x).all():
            raise PaperNumericalFailure(
                "nonfinite ancestral predictor", t, k, x, diagnostics
            )
    return x[reference_count:].cpu(), x[:reference_count].cpu(), diagnostics


class VectorRatioDiscriminator(torch.nn.Module):
    """Smooth time-conditioned classifier with no cross-example normalization."""

    def __init__(self, dim, timesteps, width=128):
        super().__init__()
        self.timesteps = timesteps
        self.register_buffer(
            "frequencies", torch.exp(torch.linspace(0, math.log(1000), 8))
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 16, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, width),
            torch.nn.SiLU(),
            torch.nn.Linear(width, 1),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        ti = torch.as_tensor(t, device=x.device, dtype=x.dtype).expand(len(x))
        phase = ti[:, None] / max(self.timesteps - 1, 1) * self.frequencies
        return self.net(torch.cat([x, phase.sin(), phase.cos()], 1)).squeeze(1)


class CurrentVectorRatio:
    """Refit at every inner step using current reference particles and fresh pi_t."""

    def __init__(
        self,
        target,
        abar,
        reference_count,
        seed=20260912,
        initial_steps=100,
        refresh_steps=20,
        batch=256,
    ):
        device = abar.device
        with torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ):
            torch.manual_seed(seed)
            self.model = VectorRatioDiscriminator(target.dim, len(abar)).to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.target, self.abar = target, abar
        self.reference_count = reference_count
        self.train_count = reference_count * 3 // 4
        self.generator = gen(seed + 71, device)
        self.validation_generator = gen(seed + 72, device)
        self.initial_steps, self.refresh_steps, self.batch = (
            initial_steps,
            refresh_steps,
            batch,
        )
        self.fitted = False
        self.updates = 0

    def noisy_real(self, n, t, generator):
        clean = self.target.sample(n, generator)
        if isinstance(clean, tuple):
            clean = clean[0]
        return self.abar[t].sqrt() * clean + (1 - self.abar[t]).sqrt() * torch.randn(
            clean.shape, device=clean.device, generator=generator
        )

    def __call__(self, x, t, substep):
        import torch.nn.functional as F

        self.model.train().requires_grad_(True)
        fake_pool = x[: self.train_count].detach()
        steps = self.initial_steps if not self.fitted else self.refresh_steps
        with torch.enable_grad():
            for _ in range(steps):
                ix = torch.randint(
                    len(fake_pool),
                    (self.batch,),
                    device=x.device,
                    generator=self.generator,
                )
                fake = fake_pool[ix]
                real = self.noisy_real(self.batch, t, self.generator)
                loss = (
                    F.softplus(-self.model(fake, t)).mean()
                    + F.softplus(self.model(real, t)).mean()
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite discriminator training loss")
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                ):
                    raise FloatingPointError(
                        "nonfinite discriminator parameter gradient"
                    )
                self.opt.step()
        self.fitted = True
        self.updates += steps
        self.model.eval().requires_grad_(False)
        h, dh = [], []
        for q in x.split(1024):
            with torch.enable_grad():
                q = q.detach().requires_grad_(True)
                value = self.model(q, t)
                gradient = torch.autograd.grad(value.sum(), q)[0]
            h.append(value.detach())
            dh.append(gradient.detach())
        with torch.no_grad():
            fake = x[self.train_count : self.reference_count]
            real = self.noisy_real(len(fake), t, self.validation_generator)
            hf, hr = self.model(fake, t), self.model(real, t)
            val = F.softplus(-hf).mean() + F.softplus(hr).mean()
        return (
            torch.cat(h),
            torch.cat(dh),
            dict(
                discriminator_train_bce=float(loss),
                discriminator_validation_bce=float(val),
                discriminator_updates=self.updates,
            ),
        )
