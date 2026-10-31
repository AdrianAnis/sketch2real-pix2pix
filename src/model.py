"""
model.py — Generator (U-Net) & Discriminator (PatchGAN 70x70)
Penanggung jawab: Rian

Kontrak:
  Generator(sketch) -> fake_photo          [B, 3, 256, 256] -> [B, 3, 256, 256]
  Discriminator(sketch, photo) -> logits   input digabung jadi 6 channel

Kode dibangun lewat vibe coding; catat setiap sesi LLM di logs/adrian/.
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    """Encoder block U-Net: Conv(k4,s2,p1) -> [BatchNorm] -> LeakyReLU(0.2).

    Mengecilkan H dan W menjadi setengah, mengubah channel in -> out.
    use_norm=False untuk blok pertama dan blok bottleneck (1x1).
    """

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
                bias=not use_norm,  # bias redundan jika diikuti BatchNorm
            )
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        assert x.dim() == 4, f"Harus 4D [B,C,H,W], dapat {tuple(x.shape)}"
        assert x.shape[1] == self.in_channels, f"Channel {x.shape[1]} != {self.in_channels}"
        assert x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0, "H dan W harus genap"
        return self.block(x)  # [B, C_out, H/2, W/2]