import pytest
import torch
import torch.nn as nn

from src.model import Discriminator, Generator, init_weights

STD = 0.02
MODELS = [Generator, Discriminator]


def _fresh(model_cls):
    torch.manual_seed(0)
    model = model_cls()
    model.apply(init_weights)
    return model


@pytest.mark.parametrize("model_cls", MODELS)
def test_conv_weights_are_normal_0_02(model_cls):
    convs = [m for m in _fresh(model_cls).modules()
             if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    assert convs, "tidak ada layer conv yang ditemukan"
    for m in convs:
        w = m.weight.detach()
        assert w.std().item() == pytest.approx(STD, rel=0.1)
        assert abs(w.mean().item()) < 5 * STD / w.numel() ** 0.5
        if m.bias is not None:
            assert torch.count_nonzero(m.bias).item() == 0


@pytest.mark.parametrize("model_cls", MODELS)
def test_batchnorm_init(model_cls):
    bns = [m for m in _fresh(model_cls).modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns, "tidak ada BatchNorm2d yang ditemukan"

    # Gamma dipool lintas layer: satu layer (64 channel) terlalu sedikit untuk estimasi std yang stabil
    gamma = torch.cat([m.weight.detach() for m in bns])
    assert abs(gamma.mean().item() - 1.0) < 5 * STD / gamma.numel() ** 0.5
    assert gamma.std().item() == pytest.approx(STD, rel=0.15)

    for m in bns:
        assert torch.count_nonzero(m.bias).item() == 0
        # running stats tidak boleh tersentuh
        assert torch.count_nonzero(m.running_mean).item() == 0
        assert torch.all(m.running_var == 1)


# ---- Pengukuran saturasi Tanh (hanya melaporkan, tanpa assert arah) ----

def saturation_pct(img: torch.Tensor) -> float:
    """Persentase piksel dengan |nilai| > 0.99."""
    return (img.abs() > 0.99).float().mean().item() * 100


@torch.no_grad()
def _measure(g: nn.Module, sketch: torch.Tensor) -> tuple[float, float, float]:
    g.train()  # BatchNorm pakai statistik batch, sama seperti saat training
    out = g(sketch)  # [B, 3, 256, 256]
    return saturation_pct(out), out.min().item(), out.max().item()


def test_report_tanh_saturation_before_after():
    torch.manual_seed(123)
    sketch = torch.randn(8, 3, 256, 256)  # ganti dengan batch val asli (lihat catatan)

    print("\nseed | default init: sat% (min, max) -> N(0,0.02): sat% (min, max)")
    before_all, after_all = [], []
    for seed in range(5):
        torch.manual_seed(seed)
        g = Generator()                       # init default PyTorch
        b = _measure(g, sketch)
        g.apply(init_weights)                 # model yang sama, bobot diganti
        a = _measure(g, sketch)
        before_all.append(b[0])
        after_all.append(a[0])
        print(f"{seed:>4} | {b[0]:6.2f}% ({b[1]:+.4f}, {b[2]:+.4f}) -> "
              f"{a[0]:6.2f}% ({a[1]:+.4f}, {a[2]:+.4f})")
    print(f"rata-rata: {sum(before_all)/5:.2f}% -> {sum(after_all)/5:.2f}%")