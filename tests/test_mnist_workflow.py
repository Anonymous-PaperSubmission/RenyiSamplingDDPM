"""Exercise CLI dispatch and sampling orchestration without a GPU or dataset."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import MNIST
import MNIST_utils as U


class WorkflowTests(unittest.TestCase):
    def test_default_sampling_dispatch(self):
        with patch.object(U, "configure_experiment"), patch.object(
            MNIST, "activate_workspace"
        ), patch.object(MNIST, "run_sampling") as sample, patch.object(
            sys, "argv", ["MNIST.py", "sample"]
        ):
            MNIST.main()
        sample.assert_called_once_with(samples=10000, seeds=MNIST.SEEDS, levels=64)
        anchors = U.correction_anchors()
        self.assertEqual(len(set(anchors)), 64)
        self.assertEqual((anchors[0], anchors[-1]), (0, 999))
        for levels in (0, 1001):
            with self.assertRaises(ValueError):
                U.correction_anchors(levels)

    def test_removed_commands_rejected(self):
        parser = MNIST.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            for command in ("report", "capacity-report", "more-levels", "bns"):
                with self.assertRaises(SystemExit):
                    parser.parse_args([command])

    def test_fresh_all_dispatch(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(U, "configure_experiment"))
            stack.enter_context(patch.object(MNIST, "activate_workspace"))
            stack.enter_context(patch.object(sys, "argv", ["MNIST.py", "all"]))
            calls = []
            stack.enter_context(
                patch.object(
                    MNIST,
                    "train_models",
                    side_effect=lambda args: calls.append("train"),
                )
            )
            stack.enter_context(
                patch.object(
                    MNIST,
                    "pretrain_discriminator",
                    side_effect=lambda **kw: calls.append(
                        ("discriminator", kw["levels"])
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    MNIST,
                    "run_sampling",
                    side_effect=lambda **kw: calls.append(("sample", kw["levels"])),
                )
            )
            MNIST.main()
        self.assertEqual(calls, ["train", ("discriminator", 64), ("sample", 64)])

    def test_sampling_outputs_without_reports(self):
        torch.set_num_threads(2)
        random = torch.randn

        def cpu_random(*args, **kwargs):
            kwargs["device"] = "cpu"
            return random(*args, **kwargs)

        corrections = []

        def corrector(model, critic, x, t, beta, abar, k, *args, **kwargs):
            corrections.append((k, t))
            return x, [{"eta": 0.1, "step_capped": False, "clipped_fraction": 0.0}]

        def evaluate(x, classifier):
            return {}, torch.zeros(len(x), dtype=torch.long), torch.ones(len(x), 3)

        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            torch.save(
                {
                    "state": {},
                    "ddpm_sha256": "hash",
                    "dataset_sha256": "hash",
                    "anchors": [0, 2, 5, 7],
                },
                root / "discriminator.pt",
            )
            for name, value in [("MNIST_ROOT", root), ("T", 8)]:
                stack.enter_context(patch.object(U, name, value))
            stack.enter_context(patch.object(MNIST, "ORDERS", [1.0]))
            stack.enter_context(patch.object(MNIST, "activate_workspace"))
            stack.enter_context(patch.object(MNIST, "seed_all"))
            for name in (
                "load_ddpm",
                "evaluator",
                "StackedData",
                "online_adapter",
            ):
                stack.enter_context(patch.object(U, name))
            critic = Mock()
            stack.enter_context(patch.object(U, "RatioCNN", return_value=critic))
            stack.enter_context(patch.object(MNIST, "sha", return_value="hash"))
            stack.enter_context(
                patch.object(
                    MNIST,
                    "gen",
                    side_effect=lambda seed: torch.Generator().manual_seed(seed),
                )
            )
            stack.enter_context(
                patch.object(U, "mnist_schedule", return_value=(None, None))
            )
            stack.enter_context(
                patch.object(MNIST, "image_corrector", side_effect=corrector)
            )
            stack.enter_context(
                patch.object(
                    MNIST,
                    "image_predictor",
                    side_effect=lambda model, x, *args, **kwargs: x,
                )
            )
            stack.enter_context(patch.object(U, "evaluate_modes", side_effect=evaluate))
            stack.enter_context(patch.object(torch, "randn", side_effect=cpu_random))
            stack.enter_context(patch.object(torch.cuda, "synchronize"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            MNIST.run_sampling(samples=2, seeds=[42], levels=4)
            expected = [
                (k, t) for k in (1, 2, 3) for t in reversed(U.correction_anchors(4))
            ]
            self.assertEqual(corrections, expected)
            folder = root / "correction_levels4"
            self.assertEqual(len(list(folder.glob("runs/*/result.pt"))), 4)
            self.assertFalse(list(root.rglob("*.md")))
            self.assertFalse(list(root.rglob("*.csv")))
            run = folder / "runs/K0_a1_s42"
            saved = torch.load(run / "samples.pt", weights_only=False)
            self.assertEqual(saved["samples"].shape, (2, 3, 28, 28))
            self.assertEqual(saved["config"]["levels"], [0, 2, 5, 7])
            with Image.open(run / "generated_samples/000.png") as preview:
                self.assertEqual(preview.size, (84, 28))
            self.assertEqual(len(list((run / "generated_samples").glob("*.png"))), 2)
            self.assertEqual(
                torch.load(folder / "completion.pt", weights_only=False)["succeeded"], 4
            )
            MNIST.run_sampling(samples=2, seeds=[42], levels=4)
            self.assertEqual(corrections, expected)

    def test_prior_option_removed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            MNIST.build_parser().parse_args(["sample", "--rare-probability", "0.02"])

    def test_discriminator_bank_uses_64_anchors(self):
        random = torch.randn

        def cpu_random(*args, **kwargs):
            kwargs["device"] = "cpu"
            return random(*args, **kwargs)

        with patch.object(torch, "randn", side_effect=cpu_random), patch.object(
            U, "gen", side_effect=lambda seed: torch.Generator().manual_seed(seed)
        ), patch.object(U, "mnist_schedule", return_value=(None, None)), patch.object(
            U, "image_predictor", side_effect=lambda model, x, *args, **kwargs: x
        ):
            bank, _ = U.baseline_bank(None, n=2)
        self.assertEqual(sorted(bank), U.correction_anchors())
        self.assertEqual(len(bank), 64)

    def test_discriminator_rejects_mismatched_anchors(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            U, "MNIST_ROOT", Path(tmp)
        ), patch.object(MNIST, "activate_workspace"), patch.object(
            U, "load_ddpm"
        ) as load:
            torch.save({"anchors": [0, 999]}, Path(tmp) / "discriminator.pt")
            with self.assertRaisesRegex(ValueError, "anchors"):
                MNIST.pretrain_discriminator()
            load.assert_not_called()

    def test_preview_pixels(self):
        x = torch.linspace(-1, 1, 3 * 28 * 28).reshape(1, 3, 28, 28)
        with tempfile.TemporaryDirectory() as tmp:
            U.save_sample_images(x, tmp)
            expected = torch.cat(
                list(((x[0] + 1) * 127.5).round().byte()), dim=1
            ).numpy()
            with Image.open(Path(tmp) / "generated_samples/000.png") as preview:
                np.testing.assert_array_equal(np.asarray(preview), expected)


if __name__ == "__main__":
    unittest.main()
