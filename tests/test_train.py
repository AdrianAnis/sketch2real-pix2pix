import math

import torch
from torch import nn

from src.model import Generator, Discriminator, init_weights
from src.train import train_step


def _snapshot(module: nn.Module):
    return [p.detach().clone() for p in module.parameters()]


def _any_changed(before, module: nn.Module) -> bool:
    return any(not torch.equal(b, p.detach()) for b, p in zip(before, module.parameters()))


def test_train_step_dummy():
    torch.manual_seed(0)

    G = Generator(in_channels=3, out_channels=3, base=64)
    D = Discriminator(in_channels=6, base_ch=64)
    G.apply(init_weights)
    D.apply(init_weights)

    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    # Batch 2 (bukan 1): bottleneck 1x1 + BatchNorm butuh >1 sampel di mode train
    sketch = torch.rand(2, 3, 256, 256) * 2 - 1
    photo = torch.rand(2, 3, 256, 256) * 2 - 1

    g_before = _snapshot(G)
    d_before = _snapshot(D)

    loss_D, loss_G = train_step(G, D, opt_G, opt_D, sketch, photo, bce, l1, lambda_l1=100)

    assert isinstance(loss_D, float) and isinstance(loss_G, float)
    assert math.isfinite(loss_D), f"loss_D tidak finite: {loss_D}"
    assert math.isfinite(loss_G), f"loss_G tidak finite: {loss_G}"

    assert _any_changed(g_before, G), "bobot G tidak berubah setelah satu langkah"
    assert _any_changed(d_before, D), "bobot D tidak berubah setelah satu langkah"