"""Tes src/metrics.py tanpa data asli dan tanpa checkpoint asli."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metrics import (
    collect_val_paths,
    compute_image_metrics,
    evaluate_metrics,
    main,
)


class ShiftGenerator(nn.Module):
    """Generator palsu: keluaran = masukan + 0.2 (selisih 0.1 di skala [0, 1])."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.2


def test_identical_images_give_perfect_scores():
    torch.manual_seed(0)
    x = torch.rand(3, 64, 64) * 2 - 1  # [3, 64, 64] di [-1, 1]
    with np.errstate(divide="ignore"):  # PSNR gambar identik = inf
        psnr, ssim = compute_image_metrics(x, x)
    assert psnr > 100
    assert ssim == pytest.approx(1.0, abs=1e-6)


def test_constant_shift_gives_known_psnr():
    real = torch.zeros(3, 64, 64)
    fake = real + 0.2  # di skala [0, 1] selisihnya 0.1, MSE 0.01, PSNR 20 dB
    psnr, ssim = compute_image_metrics(fake, real)
    assert psnr == pytest.approx(20.0, abs=1e-4)
    assert 0.9 < ssim < 1.0


def test_noisy_image_lowers_both_scores():
    torch.manual_seed(0)
    real = torch.rand(3, 64, 64) * 2 - 1
    noisy = (real + 0.3 * torch.randn_like(real)).clamp(-1, 1)
    psnr, ssim = compute_image_metrics(noisy, real)
    assert psnr < 25
    assert ssim < 0.95


def test_evaluate_metrics_is_per_image():
    torch.manual_seed(0)
    sketches = torch.rand(10, 3, 64, 64) - 0.5  # [10, 3, 64, 64] di [-0.5, 0.5]
    loader = DataLoader(TensorDataset(sketches, sketches.clone()), batch_size=4, shuffle=False)
    result = evaluate_metrics(ShiftGenerator(), loader, torch.device("cpu"))
    assert set(result) == {"psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "n"}
    assert result["n"] == 10  # 10 gambar dalam 3 batch (4 + 4 + 2)
    assert result["psnr_mean"] == pytest.approx(20.0, abs=1e-3)
    assert result["psnr_std"] == pytest.approx(0.0, abs=1e-3)
    assert 0.9 < result["ssim_mean"] < 1.0


def test_evaluate_metrics_rejects_eval_mode():
    loader = DataLoader(TensorDataset(torch.zeros(2, 3, 64, 64), torch.zeros(2, 3, 64, 64)))
    with pytest.raises(RuntimeError):
        evaluate_metrics(ShiftGenerator().eval(), loader, torch.device("cpu"))


def test_collect_val_paths_sorted_and_filtered(tmp_path):
    for name in ["b_AB.jpg", "a_AB.jpg", "catatan.txt"]:
        (tmp_path / name).write_bytes(b"")
    paths = collect_val_paths(tmp_path)
    assert [p.name for p in paths] == ["a_AB.jpg", "b_AB.jpg"]


def test_collect_val_paths_empty_folder_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_val_paths(tmp_path)


def test_cli_fails_clearly_when_val_dir_missing(tmp_path, capsys):
    code = main(["--val_dir", str(tmp_path / "tidak_ada"), "--ckpt", str(tmp_path / "x.pt")])
    assert code == 1
    assert "tidak ditemukan" in capsys.readouterr().err