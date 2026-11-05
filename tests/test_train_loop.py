import math

import torch
from torch import nn
from torch.optim import Adam

from src.train import train, load_checkpoint


class TinyG(nn.Module):
    """Generator mini: [B,3,H,W] -> [B,3,H,W], mencatat (training, grad_enabled)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.seen = []

    def forward(self, x):
        self.seen.append((self.training, torch.is_grad_enabled()))
        return torch.tanh(self.conv(x))


class TinyD(nn.Module):
    """Discriminator mini: D(sketch, photo) -> logits tanpa Sigmoid."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 1, 4, stride=2, padding=1)

    def forward(self, sketch, photo):
        return self.conv(torch.cat([sketch, photo], dim=1))  # [B, 1, H/2, W/2]


def _loader():
    g = torch.Generator().manual_seed(0)
    # batch kedua berukuran 1 untuk menguji rata-rata berbobot
    return [
        (torch.rand(2, 3, 16, 16, generator=g) * 2 - 1,
         torch.rand(2, 3, 16, 16, generator=g) * 2 - 1),
        (torch.rand(1, 3, 16, 16, generator=g) * 2 - 1,
         torch.rand(1, 3, 16, 16, generator=g) * 2 - 1),
    ]


def test_history_finite_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    hist = train(TinyG(), TinyD(), _loader(), _loader(), num_epochs=2,
                 ckpt_dir=tmp_path, device="cpu", keep_all=True)
    for key in ("loss_D", "loss_G", "val_l1"):
        assert len(hist[key]) == 2
        assert all(math.isfinite(v) for v in hist[key]), key
    for name in ("latest.pt", "epoch_001.pt", "epoch_002.pt"):
        assert (tmp_path / name).exists(), name


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    G, D = TinyG(), TinyD()
    hist = train(G, D, _loader(), _loader(), num_epochs=1,
                 ckpt_dir=tmp_path, device="cpu")

    torch.manual_seed(123)  # inisialisasi berbeda
    G2, D2 = TinyG(), TinyD()
    opt_G2 = Adam(G2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D2 = Adam(D2.parameters(), lr=2e-4, betas=(0.5, 0.999))
    epoch, hist2 = load_checkpoint(tmp_path / "latest.pt", G2, D2, opt_G2, opt_D2, "cpu")

    assert epoch == 1
    assert hist2 == hist
    for a, b in zip(G.state_dict().values(), G2.state_dict().values()):
        assert torch.equal(a, b)
    for a, b in zip(D.state_dict().values(), D2.state_dict().values()):
        assert torch.equal(a, b)
    assert len(opt_G2.state_dict()["state"]) > 0, "state optimizer G tidak dipulihkan"


def test_resume_continues_from_saved_epoch(tmp_path):
    torch.manual_seed(0)
    hist1 = train(TinyG(), TinyD(), _loader(), _loader(), num_epochs=2,
                  ckpt_dir=tmp_path, device="cpu")

    torch.manual_seed(7)
    hist2 = train(TinyG(), TinyD(), _loader(), _loader(), num_epochs=3,
                  ckpt_dir=tmp_path, device="cpu",
                  resume_from=tmp_path / "latest.pt")

    assert len(hist2["loss_D"]) == 3
    assert hist2["loss_D"][:2] == hist1["loss_D"]
    assert torch.load(tmp_path / "latest.pt")["epoch"] == 3


def test_validation_uses_eval_no_grad_then_back_to_train(tmp_path):
    torch.manual_seed(0)
    G, D = TinyG(), TinyD()
    train(G, D, _loader(), _loader(), num_epochs=1, ckpt_dir=tmp_path, device="cpu")

    assert (True, True) in G.seen, "training harus memakai mode train dengan grad"
    assert G.seen[-1] == (False, False), "validasi harus eval() + no_grad()"
    assert G.training and D.training, "G dan D harus kembali ke mode train"