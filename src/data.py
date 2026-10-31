from __future__ import annotations

import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Union

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def select_edges2shoes_subset(
    data_root: Union[str, Path],
    n_train: int = 10_000,
    seed: int = 42,
    train_subdir: str = "train",
    val_subdir: str = "val",
) -> Tuple[List[Path], List[Path]]:
    """Pilih subset train (acak, seed tetap) dan seluruh val dari edges2shoes.

    Args:
        data_root: Folder yang berisi subfolder train/ dan val/.
        n_train: Jumlah gambar train yang diambil.
        seed: Seed untuk pengambilan subset (reproducible).
        train_subdir, val_subdir: Nama subfolder split.

    Returns:
        (train_paths, val_paths): dua list Path terurut. Tiap file berisi
        sketch (setengah kiri) dan foto (setengah kanan), 512x256.
    """
    root = Path(data_root)
    train_dir, val_dir = root / train_subdir, root / val_subdir

    for d in (train_dir, val_dir):
        if not d.is_dir():
            isi = sorted(p.name for p in root.iterdir()) if root.is_dir() else "root tidak ada"
            raise FileNotFoundError(f"Folder tidak ditemukan: {d}. Isi {root}: {isi}")

    def _list_images(folder: Path) -> List[Path]:
        # sorted -> urutan deterministik, tidak bergantung urutan file system
        return sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )

    train_all = _list_images(train_dir)
    val_all = _list_images(val_dir)

    if len(train_all) < n_train:
        raise ValueError(f"Train hanya {len(train_all)} gambar, diminta {n_train}.")

    rng = random.Random(seed)  # RNG lokal, tidak mengganggu RNG global
    train_paths = sorted(rng.sample(train_all, n_train))  # tanpa pengembalian

    return train_paths, val_all

from PIL import Image

IMG_SIZE = 256  # sisi tiap gambar (sketch dan photo), sesuai kontrak


def load_pair(path: Union[str, Path]) -> Tuple[Image.Image, Image.Image]:
    """Baca satu file edges2shoes (512x256) lalu potong jadi (sketch, photo).

    Setengah kiri = sketch (input), setengah kanan = photo (target).

    Returns:
        (sketch, photo): dua PIL Image RGB, masing-masing 256x256.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")  # paksa 3 channel; hasilnya salinan yang sudah terbaca

    w, h = im.size  # PIL: (lebar, tinggi)
    assert (w, h) == (2 * IMG_SIZE, IMG_SIZE), f"Ukuran tak terduga {im.size} pada {path}"

    half = w // 2  # 256
    sketch = im.crop((0, 0, half, h))   # kolom 0..255
    photo = im.crop((half, 0, w, h))    # kolom 256..511

    assert sketch.size == photo.size == (IMG_SIZE, IMG_SIZE)
    return sketch, photo

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> tensor [3, 256, 256], float32, rentang [-1, 1].

    Pemetaan linear: 0 -> -1.0, 127.5 -> 0.0, 255 -> +1.0.
    """
    arr = np.array(img.convert("RGB"), dtype=np.uint8)    # [H, W, 3], 0..255 (salinan)
    x = torch.from_numpy(arr).permute(2, 0, 1)            # [3, H, W]
    x = x.to(torch.float32) / 127.5 - 1.0                 # [-1, 1]
    x = x.contiguous()
    assert x.shape == (3, IMG_SIZE, IMG_SIZE), f"Shape tak terduga: {tuple(x.shape)}"
    return x


def tensor_to_image(t: torch.Tensor) -> np.ndarray:
    """Tensor [3, H, W] di [-1, 1] -> array [H, W, 3] float32 di [0, 1] (siap imshow)."""
    assert t.ndim == 3 and t.shape[0] == 3, f"Butuh [3, H, W], dapat {tuple(t.shape)}"
    x = (t.detach().cpu().to(torch.float32) + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    x = x.clamp(0.0, 1.0)
    return x.permute(1, 2, 0).numpy()                     # [H, W, 3]