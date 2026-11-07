"""Evaluasi PSNR dan SSIM Generator Pix2Pix pada gambar validasi edges2shoes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader

from src.data import Edges2ShoesDataset
from src.infer import load_generator

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def to_unit_hwc(img: torch.Tensor) -> np.ndarray:
    """Ubah satu gambar [3, H, W] di [-1, 1] menjadi [H, W, 3] float64 di [0, 1]."""
    assert img.ndim == 3 and img.shape[0] == 3, f"bentuk tidak valid: {tuple(img.shape)}"
    img = img.detach().cpu().clamp(-1.0, 1.0)  # [3, H, W]
    unit = (img + 1.0) / 2.0  # [3, H, W], rentang [0, 1]
    return unit.permute(1, 2, 0).numpy().astype(np.float64)  # [H, W, 3]


def compute_image_metrics(fake: torch.Tensor, real: torch.Tensor) -> tuple[float, float]:
    """Hitung (PSNR, SSIM) untuk satu pasang gambar [3, H, W] di rentang [-1, 1]."""
    assert fake.shape == real.shape, f"{tuple(fake.shape)} != {tuple(real.shape)}"
    fake_np = to_unit_hwc(fake)  # [H, W, 3]
    real_np = to_unit_hwc(real)  # [H, W, 3]
    psnr = peak_signal_noise_ratio(real_np, fake_np, data_range=1.0)
    ssim = structural_similarity(real_np, fake_np, data_range=1.0, channel_axis=-1)
    return float(psnr), float(ssim)


def collect_val_paths(val_dir: str | Path) -> list[Path]:
    """Kumpulkan path gambar di val_dir, terurut. Gagal dengan pesan jelas bila kosong."""
    val_dir = Path(val_dir)
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Folder val tidak ditemukan: {val_dir}")
    paths = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"Tidak ada gambar (.jpg/.jpeg/.png) di folder: {val_dir}")
    return paths


def evaluate_metrics(G: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Hitung PSNR dan SSIM per gambar, lalu rata-rata dan simpangan bakunya.

    Generator harus dalam mode train() (keputusan D-017). Tidak memanggil eval().
    """
    if not G.training:
        raise RuntimeError("Generator harus dalam mode train() (keputusan D-017), bukan eval().")

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    with torch.no_grad():
        for sketch, photo in loader:  # keduanya [B, 3, 256, 256]
            fake = G(sketch.to(device))  # [B, 3, 256, 256]
            assert fake.shape == photo.shape, f"{tuple(fake.shape)} != {tuple(photo.shape)}"
            for i in range(fake.shape[0]):
                psnr, ssim = compute_image_metrics(fake[i], photo[i])
                psnr_values.append(psnr)
                ssim_values.append(ssim)

    n = len(psnr_values)
    if n == 0:
        raise ValueError("Loader tidak menghasilkan gambar apa pun.")
    ddof = 1 if n > 1 else 0  # simpangan baku sampel
    return {
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_std": float(np.std(psnr_values, ddof=ddof)),
        "ssim_mean": float(np.mean(ssim_values)),
        "ssim_std": float(np.std(ssim_values, ddof=ddof)),
        "n": n,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hitung PSNR dan SSIM Generator pada data val.")
    parser.add_argument("--ckpt", default="G_only.pt", help="file bobot Generator")
    parser.add_argument("--val_dir", default="data/edges2shoes/val", help="folder gambar val")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                        help="default: cuda jika tersedia, selain itu cpu")
    parser.add_argument("--output", default="results/metrics.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        paths = collect_val_paths(args.val_dir)
    except FileNotFoundError as err:
        print(f"Eror: {err}", file=sys.stderr)
        return 1

    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        print(f"Eror: file checkpoint tidak ditemukan: {ckpt}", file=sys.stderr)
        return 1

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("Eror: --device cuda diminta, tetapi CUDA tidak tersedia.", file=sys.stderr)
        return 1
    device = torch.device(device_name)

    G = load_generator(ckpt, device)
    dataset = Edges2ShoesDataset([str(p) for p in paths], mode="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)  # Dropout aktif di mode train(), jadi seed perlu dicatat
    results = evaluate_metrics(G, loader, device)

    report = {
        "ckpt": str(ckpt),
        "val_dir": str(args.val_dir),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": device_name,
        **results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Jumlah gambar : {results['n']}")
    print(f"PSNR          : {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"SSIM          : {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
    print(f"Disimpan ke   : {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())