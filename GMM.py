"""Two- and higher-dimensional imbalanced Gaussian-mixture experiments."""

import argparse
import math
import numpy as np
import torch
from GMM_utils import (
    ROOT,
    GaussianMixtureBase,
    gen,
    run_vector_experiment,
)


def maximin_directions(k, d, seed=0, steps=400, lr=0.02):
    x = torch.randn(k, d, generator=gen(seed, "cpu"))
    x /= x.norm(dim=1, keepdim=True)
    for _ in range(steps):
        diff = x[:, None] - x[None]
        dist = diff.norm(dim=2) + torch.eye(k) * 1e9
        x += lr * (diff / dist[:, :, None] ** 3).sum(1)
        x /= x.norm(dim=1, keepdim=True)
    return x


class ImbalancedGMM(GaussianMixtureBase):
    """Eight separated Gaussian modes with a one-percent rare component."""

    def __init__(self, dim=16, k=8, separation=5.6569, std=0.3, rare=0.01, seed=0):
        self.kind = "gmm"
        self.dim = dim
        self.k = k
        self.std = std
        self.rare = rare
        self.device = "cpu"
        directions = maximin_directions(k, dim, seed)
        distances = torch.cdist(directions, directions) + torch.eye(k) * 1e9
        self.centers = directions * (separation / distances.min())
        self.weights = torch.tensor([rare] + [(1 - rare) / (k - 1)] * (k - 1))

    def metrics(self, x):
        x = x.detach().cpu()
        centers = self.centers.cpu()
        ds = []
        ids = []
        for q in x.split(1024):
            d = torch.cdist(q, centers)
            val, idx = d.min(1)
            ds.append(val)
            ids.append(idx)
        dist = torch.cat(ds)
        nearest = torch.cat(ids)
        inter = torch.cdist(centers, centers)
        inter.fill_diagonal_(float("inf"))
        cutoff = 0.5 * inter.min()
        on = dist <= cutoff
        counts = torch.bincount(nearest[on], minlength=self.k)
        freq = counts.float() / len(x)
        leak = (~on).float().mean()
        truth = self.weights.cpu()
        allfreq = torch.bincount(nearest, minlength=self.k).float() / len(x)
        from scipy.stats import chi

        natural_cutoff = self.std * float(chi.ppf(0.99, self.dim))
        out = {
            "rare_recall": (counts[0] / max(on.sum().item(), 1)).item(),
            "rare_mass": freq[0].item(),
            "rare_expected": self.rare,
            "unassigned_frac": leak.item(),
            "tv_distance": (0.5 * ((freq - truth).abs().sum() + leak)).item(),
            "nearest_mode_tv": (0.5 * (allfreq - truth).abs().sum()).item(),
            "natural_99pct_tail": (dist > natural_cutoff).float().mean().item(),
            "mode_coverage": (counts > 0).float().mean().item(),
            "mode_counts": counts.tolist(),
        }
        chi_mean = (
            self.std
            * math.sqrt(2)
            * math.exp(math.lgamma((self.dim + 1) / 2) - math.lgamma(self.dim / 2))
        )
        logs = []
        for k in range(self.k):
            ratio = (freq[k] / truth[k]).item()
            out[f"r_mode{k}"] = ratio
            sel = on & (nearest == k)
            out[f"rad_mode{k}"] = (
                (dist[sel].mean() / chi_mean).item() if sel.any() else None
            )
            if ratio > 0:
                logs.append(math.log(ratio))
        out["logr_mode_sd"] = float(np.std(logs)) if logs else 0.0
        out["logr_mode_absmax"] = max(map(abs, logs)) if logs else 0.0
        out["nonfinite_count"] = int((~torch.isfinite(x)).sum())
        out["max_norm"] = x.norm(dim=1).max().item()
        return out


def validate_gmm():
    for dim in [2, 8, 16, 32, 64]:
        target = ImbalancedGMM(dim=dim)
        distances = torch.cdist(target.centers, target.centers)
        distances.fill_diagonal_(float("inf"))
        assert abs(distances.min().item() - 5.6569) < 1e-4
    target = ImbalancedGMM(dim=8)
    x = torch.randn(12, 8, requires_grad=True)
    lp, score = target.log_score(x, 0.4)
    torch.testing.assert_close(
        torch.autograd.grad(lp.sum(), x)[0], score, atol=3e-6, rtol=3e-5
    )
    print("GMM validation passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 8, 16, 32, 64])
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        validate_gmm()
        return
    for dim in args.dims:
        run_vector_experiment(
            ImbalancedGMM(dim=dim),
            ROOT / "outputs/GMM" / f"d{dim}",
            15000,
            f"gmm{dim}",
            args.train_only,
        )


def paper_main():
    from GMM_utils import paper_vector_sweep, VectorDDPM, schedule, sha

    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", choices=["kde", "discriminator"], required=True)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 8, 16, 32, 64])
    args = parser.parse_args(__import__("sys").argv[2:])
    for dim in args.dims:
        folder = ROOT / "outputs/GMM" / f"d{dim}"
        checkpoint, dataset = folder / "model.pt", folder / "dataset.pt"
        saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
        assert saved["dataset_sha256"] == sha(dataset)
        target = ImbalancedGMM(dim=dim).to("cuda")
        original = torch.load(
            folder / "target.pt", map_location="cuda", weights_only=False
        )
        assert torch.equal(target.centers, original["centers"]) and torch.equal(
            target.weights, original["weights"]
        )
        net = VectorDDPM(dim, T=saved["T"]).cuda().eval().requires_grad_(False)
        net.load_state_dict(saved["ema"])
        beta, abar = schedule(saved["T"])

        def score(q, t):
            return -net(q, t) / (1 - abar[t]).sqrt()

        paper_vector_sweep(
            target,
            score,
            beta,
            abar,
            folder / ("paper_" + args.estimator),
            checkpoint,
            dataset,
            target.metrics,
            bandwidth_floor=target.std * 0.45,
            estimator=args.estimator,
        )


if __name__ == "__main__":
    if len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "paper":
        paper_main()
    else:
        main()
