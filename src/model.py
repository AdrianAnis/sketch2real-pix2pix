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

class UpBlock(nn.Module):
    """Decoder block U-Net: ConvT(k4,s2,p1) -> BatchNorm -> [Dropout(0.5)] -> ReLU.

    Menggandakan H dan W, mengubah channel in -> out.
    Concat skip connection TIDAK dilakukan di sini (tanggung jawab Generator).
    """

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1,
                bias=False,  # redundan karena diikuti BatchNorm
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        assert x.dim() == 4, f"Harus 4D [B,C,H,W], dapat {tuple(x.shape)}"
        assert x.shape[1] == self.in_channels, f"Channel {x.shape[1]} != {self.in_channels}"
        return self.block(x)  # [B, C_out, 2H, 2W]

class Generator(nn.Module):
    """U-Net generator: sketch [B, 3, 256, 256] -> fake photo [B, 3, 256, 256] dalam [-1, 1]."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base: int = 64) -> None:
        super().__init__()
        # Encoder: 8 blok, 256 -> 1. Channel berhenti naik di base*8 (512).
        self.downs = nn.ModuleList([
            DownBlock(in_channels, base, use_norm=False),  # d1 [B, 64, 128, 128]
            DownBlock(base, base * 2),                     # d2 [B, 128, 64, 64]
            DownBlock(base * 2, base * 4),                 # d3 [B, 256, 32, 32]
            DownBlock(base * 4, base * 8),                 # d4 [B, 512, 16, 16]
            DownBlock(base * 8, base * 8),                 # d5 [B, 512, 8, 8]
            DownBlock(base * 8, base * 8),                 # d6 [B, 512, 4, 4]
            DownBlock(base * 8, base * 8),                 # d7 [B, 512, 2, 2]
            DownBlock(base * 8, base * 8, use_norm=False), # d8 [B, 512, 1, 1] bottleneck
        ])
        # Decoder: input = 2x output blok sebelumnya (karena concat), kecuali up1.
        self.ups = nn.ModuleList([
            UpBlock(base * 8, base * 8, use_dropout=True),   # up1 512  -> [B, 512, 2, 2]
            UpBlock(base * 16, base * 8, use_dropout=True),  # up2 1024 -> [B, 512, 4, 4]
            UpBlock(base * 16, base * 8, use_dropout=True),  # up3 1024 -> [B, 512, 8, 8]
            UpBlock(base * 16, base * 8),                    # up4 1024 -> [B, 512, 16, 16]
            UpBlock(base * 16, base * 4),                    # up5 1024 -> [B, 256, 32, 32]
            UpBlock(base * 8, base * 2),                     # up6 512  -> [B, 128, 64, 64]
            UpBlock(base * 4, base),                         # up7 256  -> [B, 64, 128, 128]
        ])
        # Output: 128 -> [B, 3, 256, 256], Tanh agar rentang cocok dengan data [-1, 1].
        self.out = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sambil menyimpan skip, lalu decode dengan concat skip di dim channel."""
        assert x.dim() == 4 and x.shape[2] % 256 == 0 and x.shape[3] % 256 == 0, \
            f"H dan W harus kelipatan 256 (8x downsample ke 1x1), dapat {tuple(x.shape)}"

        skips: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)

        skips.pop()  # d8 = bottleneck = x saat ini, bukan skip
        for up in self.ups:
            x = up(x)
            skip = skips.pop()  # d7, d6, ..., d1
            assert x.shape[2:] == skip.shape[2:], \
                f"Spatial mismatch: up {tuple(x.shape)} vs skip {tuple(skip.shape)}"
            x = torch.cat([x, skip], dim=1)  # [B, C_up + C_skip, H, W]

        return self.out(x)  # [B, 3, 256, 256]