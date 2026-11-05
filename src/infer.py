"""
infer.py — Demo sketsa -> foto (Pix2Pix)
Penanggung jawab: Rian

Memuat Generator dari checkpoint lalu mengubah satu sketsa menjadi foto.
Per D-017, Generator dijalankan dalam mode train() di bawah no_grad()
(BatchNorm memakai statistik batch, Dropout aktif), bukan eval().

Pemakaian:
  python -m src.infer --ckpt checkpoints/latest.pt --input sketsa.png --output hasil.png --seed 0

Kode dibangun lewat vibe coding; catat setiap sesi LLM di logs/adrian/.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

from src.data import load_pair, pil_to_tensor, tensor_to_image
from src.model import Generator

PathLike = Union[str, Path]

def _extract_g_state(ckpt: Any) -> dict[str, torch.Tensor]:
    """Ambil state_dict Generator dari checkpoint penuh atau file state_dict murni."""
    state = ckpt["G"] if isinstance(ckpt, dict) and "G" in ckpt else ckpt
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint tidak berisi state_dict Generator yang valid.")
    if not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("State_dict Generator harus berisi tensor saja.")
    return state


def load_generator(ckpt_path: PathLike, device: Union[str, torch.device] = "cpu") -> Generator:
    """Muat Generator untuk inferensi gaya Pix2Pix (mode train(), BN momentum 0).

    Mendukung checkpoint penuh (kunci "G") maupun file state_dict Generator saja.
    """
    device = torch.device(device)
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=True)

    G = Generator(in_channels=3, out_channels=3, base=64)
    G.load_state_dict(_extract_g_state(ckpt))  # strict: salah arsitektur langsung error
    G.to(device)

    G.train()  # keputusan D-017: BN pakai statistik batch, Dropout aktif
    for m in G.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0  # running stats tidak berubah saat inferensi

    return G

IMG_SIZE = 256
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC  # Pillow lama & baru


def _flatten_to_white_rgb(img: Image.Image) -> Image.Image:
    """Putuskan orientasi EXIF dan komposit transparansi ke latar putih -> RGB."""
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return img.convert("RGB")


def _is_dark_background(img: Image.Image) -> bool:
    """Median luminansi < 128 dianggap latar gelap (garis tipis tidak memengaruhi median)."""
    lum = np.asarray(img.convert("L"))
    return float(np.median(lum)) < 128.0


def _letterbox_white(img: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    """Skala sisi terpanjang ke `size`, jaga rasio, pad putih ke size x size."""
    w, h = img.size
    if (w, h) == (size, size):
        return img
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = img.resize((nw, nh), _BICUBIC)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas

def preprocess_sketch(sketch_image: Image.Image, invert: Optional[bool] = None) -> Image.Image:
    """Sketsa apa pun -> RGB 256x256, garis gelap di latar putih.

    invert: None = deteksi otomatis, True = paksa invert, False = jangan invert.
    """
    img = _flatten_to_white_rgb(sketch_image)
    if invert is None:
        invert = _is_dark_background(img)
    if invert:
        img = ImageOps.invert(img)
    return _letterbox_white(img)

def sketch_to_photo(
    G: Generator,
    sketch_image: Image.Image,
    seed: Optional[int] = None,
    invert: Optional[bool] = None,
    context: Optional[list[Image.Image]] = None,
) -> Image.Image:
    """Sketsa (PIL) -> foto hasil Generator (PIL RGB 256x256).

    Mode train() dipaksa (D-017). `context`: sketsa pendamping (sudah 256x256,
    garis gelap di latar putih) agar statistik BatchNorm dihitung atas batch
    [1+N], bukan batch 1. Hanya output sketsa pengguna (indeks 0) yang dikembalikan.
    Dengan `seed`, mask Dropout deterministik (untuk context yang sama).
    """
    device = next(G.parameters()).device
    G.train()

    sketch = preprocess_sketch(sketch_image, invert)
    ctx = list(context) if context else []
    for i, c in enumerate(ctx):
        if c.size != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"context[{i}] harus {IMG_SIZE}x{IMG_SIZE}, dapat {c.size}.")

    n = 1 + len(ctx)
    x = torch.stack([pil_to_tensor(s.convert("RGB")) for s in [sketch, *ctx]]).to(device)
    assert x.shape == (n, 3, IMG_SIZE, IMG_SIZE), x.shape  # [1+N, 3, 256, 256]

    if seed is None:
        rng_ctx = contextlib.nullcontext()
    else:
        cuda_ids = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        rng_ctx = torch.random.fork_rng(devices=cuda_ids)

    with rng_ctx:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            fake = G(x)  # [1+N, 3, 256, 256], rentang [-1, 1]
    assert fake.shape == x.shape, fake.shape

    arr = np.asarray(tensor_to_image(fake[0].cpu()))  # hanya sketsa pengguna: [256, 256, 3]
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_val_images(val_dir: PathLike) -> list[Path]:
    """Daftar gambar val terurut berdasarkan nama (urutan indeks --val_index)."""
    d = Path(val_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"Folder validasi tidak ditemukan: {d}")
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _IMG_EXTS)


DEFAULT_CONTEXT_DIR = "assets/context_sketches"


def load_context_sketches(context_dir: PathLike = DEFAULT_CONTEXT_DIR) -> list[Image.Image]:
    """Baca semua PNG di folder (urut nama), praproses tiap sketsa -> list PIL 256x256.

    Folder tidak ada atau kosong -> list kosong + peringatan (inferensi jatuh ke batch 1).
    """
    d = Path(context_dir)
    if not d.is_dir():
        warnings.warn(
            f"Folder context '{d}' tidak ada: inferensi memakai batch 1 "
            "(warna bisa lebih jenuh/bernoda).",
            stacklevel=2,
        )
        return []
    files = sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png")
    if not files:
        warnings.warn(
            f"Tidak ada PNG di '{d}': inferensi memakai batch 1.", stacklevel=2
        )
        return []
    out: list[Image.Image] = []
    for p in files:
        with Image.open(p) as im:
            out.append(preprocess_sketch(im.copy()))  # auto-invert + letterbox
    return out


def compose_side_by_side(panels: list[Image.Image]) -> Image.Image:
    """Susun panel 256x256 secara horizontal -> (256*n) x 256."""
    if not panels:
        raise ValueError("Tidak ada panel untuk disusun.")
    for p in panels:
        if p.size != (IMG_SIZE, IMG_SIZE):
            raise ValueError(f"Panel harus {IMG_SIZE}x{IMG_SIZE}, dapat {p.size}.")
    canvas = Image.new("RGB", (IMG_SIZE * len(panels), IMG_SIZE), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(p.convert("RGB"), (i * IMG_SIZE, 0))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.infer",
        description="Inferensi Pix2Pix: sketsa -> foto (gambar berdampingan).",
    )
    p.add_argument("--ckpt", required=True, help="Path checkpoint (latest.pt atau state_dict G).")
    p.add_argument("--output", required=True, help="Path gambar hasil (mis. hasil.png).")
    p.add_argument("--input", default=None, help="Path gambar sketsa pengguna.")
    p.add_argument("--val_index", type=int, default=None,
                   help="Indeks file di --val_dir (urut nama), alternatif dari --input.")
    p.add_argument("--val_dir", default=None,
                   help="Folder val edges2shoes (file 512x256). Wajib bila --val_index dipakai.")
    p.add_argument("--seed", type=int, default=None, help="Seed Dropout agar hasil bisa diulang.")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None,
                   help="Default: cuda jika tersedia, selain itu cpu.")
    p.add_argument("--invert", action="store_true", default=None,
                   help="Paksa invert warna sketsa (default: deteksi otomatis).")
    p.add_argument("--context_dir", default=DEFAULT_CONTEXT_DIR,
                   help="Folder PNG sketsa pendamping untuk statistik BatchNorm.")
    p.add_argument("--no_context", action="store_true",
                   help="Batch 1 saja, tanpa sketsa pendamping (untuk perbandingan).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (args.input is None) == (args.val_index is None):
        parser.error("Isi tepat salah satu: --input atau --val_index.")
    if args.val_index is not None and args.val_dir is None:
        parser.error("--val_index wajib dipakai bersama --val_dir.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda diminta, tetapi CUDA tidak tersedia.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        real: Optional[Image.Image] = None
        if args.val_index is not None:
            files = list_val_images(args.val_dir)
            n = len(files)
            if n == 0:
                raise ValueError(f"Tidak ada gambar (.jpg/.jpeg/.png) di {args.val_dir}.")
            if not 0 <= args.val_index < n:
                raise ValueError(
                    f"--val_index {args.val_index} di luar rentang [0, {n - 1}] "
                    f"({n} file di {args.val_dir})."
                )
            sketch_raw, real = load_pair(files[args.val_index])
        else:
            path = Path(args.input)
            if not path.is_file():
                raise FileNotFoundError(f"Sketsa tidak ditemukan: {path}")
            with Image.open(path) as im:
                sketch_raw = im.copy()

        sketch = preprocess_sketch(sketch_raw, args.invert)  # [256x256, putih-hitam]
        context = [] if args.no_context else load_context_sketches(args.context_dir)
        G = load_generator(args.ckpt, device)
        fake = sketch_to_photo(G, sketch, seed=args.seed, invert=False, context=context)

        panels = [sketch, fake] + ([real] if real is not None else [])
        result = compose_side_by_side(panels)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(out_path)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(
        f"Tersimpan: {out_path} ({len(panels)} panel, device={device}, "
        f"seed={args.seed}, batch={1 + len(context)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
