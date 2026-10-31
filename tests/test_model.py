import torch
from src.model import DownBlock, UpBlock


def test_single_block() -> None:
    x = torch.randn(2, 3, 256, 256)                  # [B, 3, 256, 256]
    y = DownBlock(3, 64, use_norm=False)(x)
    assert y.shape == (2, 64, 128, 128), y.shape     # [B, 64, 128, 128]


def test_stacked_blocks() -> None:
    channels = [3, 64, 128, 256, 512, 512, 512, 512, 512]
    h = torch.randn(2, 3, 256, 256)
    for i in range(8):
        use_norm = i not in (0, 7)                   # blok pertama & bottleneck tanpa norm
        h = DownBlock(channels[i], channels[i + 1], use_norm)(h)
        print(f"blok {i + 1}: {tuple(h.shape)}")
    assert h.shape == (2, 512, 1, 1), h.shape


def test_up_single_block() -> None:
    x = torch.randn(2, 512, 1, 1)                    # [B, 512, 1, 1] (ukuran bottleneck)
    y = UpBlock(512, 512, use_dropout=True)(x)
    assert y.shape == (2, 512, 2, 2), y.shape


def test_up_stacked_blocks() -> None:
    # Tanpa concat: in_channels = out blok sebelumnya (concat ditambahkan di Generator)
    channels = [512, 512, 512, 512, 512, 256, 128, 64]
    h = torch.randn(2, 512, 1, 1)
    for i in range(7):
        h_in = h.shape[2]
        h = UpBlock(channels[i], channels[i + 1], use_dropout=i < 3)(h)
        assert h.shape[2] == h.shape[3] == 2 * h_in, f"blok {i + 1}: {tuple(h.shape)}"
        print(f"up {i + 1}: {tuple(h.shape)}")
    assert h.shape == (2, 64, 128, 128), h.shape