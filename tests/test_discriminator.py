import torch
import torch.nn as nn

from src.model import Discriminator


def test_output_shape_and_no_sigmoid() -> None:
    d = Discriminator()
    sketch = torch.randn(2, 3, 256, 256)
    photo = torch.randn(2, 3, 256, 256)
    logits = d(sketch, photo)

    # Ukuran sesuai hitungan: 256 -> 128 -> 64 -> 32 -> 31 -> 30
    assert logits.shape == (2, 1, 30, 30), logits.shape

    # Tanpa Sigmoid: tidak ada modul Sigmoid, dan nilai tidak terkunci di [0, 1]
    assert not any(isinstance(m, nn.Sigmoid) for m in d.modules())
    assert logits.min() < 0 or logits.max() > 1


def test_receptive_field_is_70x70() -> None:
    d = Discriminator().eval()  # eval: BatchNorm tidak mencampur statistik antar posisi
    sketch = torch.randn(1, 3, 256, 256)
    photo = torch.randn(1, 3, 256, 256, requires_grad=True)

    logits = d(sketch, photo)
    logits[0, 0, 15, 15].backward()  # satu sel output di tengah

    mask = photo.grad[0].abs().sum(dim=0) > 0  # [256, 256]
    rows = mask.any(dim=1).nonzero().flatten()
    cols = mask.any(dim=0).nonzero().flatten()
    assert (rows.max() - rows.min() + 1).item() == 70
    assert (cols.max() - cols.min() + 1).item() == 70