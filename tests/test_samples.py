import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from src.train import train, make_sample_saver, save_curves


class TinyG(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.seen = []

    def forward(self, x):
        self.seen.append((self.training, torch.is_grad_enabled()))
        return torch.tanh(self.conv(x))


class TinyD(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 1, 4, stride=2, padding=1)

    def forward(self, sketch, photo):
        return self.conv(torch.cat([sketch, photo], dim=1))


def _batch(n=2, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, size, size, generator=g) * 2 - 1,
            torch.rand(n, 3, size, size, generator=g) * 2 - 1)


def test_sample_saver_creates_file_and_restores_mode(tmp_path):
    G = TinyG()
    sketch, photo = _batch()
    saver = make_sample_saver(sketch, photo, tmp_path / "samples", device="cpu")

    G.train()
    saver(1, G, TinyD(), {})

    out = tmp_path / "samples" / "epoch_001.png"
    assert out.exists() and out.stat().st_size > 0
    assert G.seen[-1] == (True, False), "sampel harus dibuat dengan mode train() + no_grad()"
    assert G.training, "G harus kembali ke mode semula (train)"


def test_hook_called_each_epoch_and_samples_use_fixed_batch(tmp_path):
    torch.manual_seed(0)
    sketch, photo = _batch(n=2, size=16, seed=1)
    saver = make_sample_saver(sketch, photo, tmp_path / "samples", device="cpu")
    calls = []

    def hook(epoch, G, D, history):
        calls.append((epoch, len(history["loss_D"])))
        saver(epoch, G, D, history)

    loader = [_batch(2, 16, seed=2), _batch(1, 16, seed=3)]
    train(TinyG(), TinyD(), loader, loader, num_epochs=2,
          ckpt_dir=tmp_path / "ckpt", device="cpu", on_epoch_end=hook)

    assert calls == [(1, 1), (2, 2)]
    p1 = tmp_path / "samples" / "epoch_001.png"
    p2 = tmp_path / "samples" / "epoch_002.png"
    assert p1.exists() and p2.exists()

    a = np.array(Image.open(p1))
    b = np.array(Image.open(p2))
    assert a.shape == b.shape
    # panel: padding 2, lebar 16 -> sketsa x=2:18, hasil G x=18:34, target x=34:50
    assert np.array_equal(a[:, 2:18], b[:, 2:18]), "sketsa harus sama tiap epoch"
    assert np.array_equal(a[:, 34:50], b[:, 34:50]), "target harus sama tiap epoch"


def test_save_curves_creates_file(tmp_path):
    pytest.importorskip("matplotlib")
    history = {"loss_D": [0.7, 0.6], "loss_G": [30.0, 25.0], "val_l1": [0.3, 0.25]}
    save_curves(history, tmp_path / "results" / "loss_curves.png")
    assert (tmp_path / "results" / "loss_curves.png").exists()\


def test_sample_saver_restores_eval_mode(tmp_path):
    G = TinyG()
    sketch, photo = _batch()
    saver = make_sample_saver(sketch, photo, tmp_path / "samples", device="cpu")
    G.eval()
    saver(1, G, TinyD(), {})
    assert not G.training, "mode semula (eval) harus dipulihkan"
    assert G.seen[-1] == (True, False)