import torch
from src.model import DownBlock


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


if __name__ == "__main__":
    test_single_block()
    test_stacked_blocks()
    print("OK")