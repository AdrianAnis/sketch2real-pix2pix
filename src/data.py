"""
data.py — Dataset & DataLoader edges2shoes
Penanggung jawab: Naura

Kontrak output: pasangan (sketch, photo)
  shape [B, 3, 256, 256], float32, rentang [-1, 1]

Kode dibangun lewat vibe coding; catat setiap sesi LLM di logs/naura/.
"""

from __future__ import annotations
from PIL import Image 

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Union, Optional, Literal, Sequence
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 256  
LOAD_SIZE: int = 286   
CROP_SIZE: int = 256   

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

def random_jitter_pair(
    sketch: Image.Image,
    photo: Image.Image,
    rng: Optional[random.Random] = None,
    load_size: int = LOAD_SIZE,
    crop_size: int = CROP_SIZE,
    flip_prob: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """Random jitter + flip horizontal dengan parameter yang SAMA untuk
    sketsa dan foto. Input/output: PIL RGB 256x256."""
    rng = rng if rng is not None else random.Random()
    assert sketch.size == photo.size == (crop_size, crop_size), \
        f"Ukuran harus {crop_size}x{crop_size}, dapat {sketch.size} dan {photo.size}"

    # 1) Resize 256 -> 286 (kedua gambar, interpolasi sama)
    resample = Image.Resampling.BICUBIC
    sketch = sketch.resize((load_size, load_size), resample)
    photo = photo.resize((load_size, load_size), resample)

    # 2) Parameter crop diambil SEKALI, dipakai untuk keduanya
    max_offset = load_size - crop_size                 # 30
    left = rng.randint(0, max_offset)                  # inklusif: 0..30
    top = rng.randint(0, max_offset)
    box = (left, top, left + crop_size, top + crop_size)
    sketch = sketch.crop(box)                          # 256x256
    photo = photo.crop(box)

    # 3) Keputusan flip diambil SEKALI, dipakai untuk keduanya
    if rng.random() < flip_prob:
        sketch = sketch.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        photo = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    assert sketch.size == photo.size == (crop_size, crop_size)
    return sketch, photo

class Edges2ShoesDataset(Dataset):
    """Dataset pasangan (sketch, photo) edges2shoes.

    mode="train": load_pair -> random_jitter_pair -> pil_to_tensor
    mode="val"  : load_pair -> pil_to_tensor (tanpa augmentasi)
    """

    def __init__(self, paths: Sequence[str], mode: Literal["train", "val"]) -> None:
        if mode not in ("train", "val"):
            raise ValueError(f'mode harus "train" atau "val", dapat {mode!r}')
        assert len(paths) > 0, "Daftar path kosong"
        self.paths: List[str] = list(paths)
        self.mode: str = mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sketch, photo = load_pair(self.paths[idx])          # PIL RGB 256x256
        if self.mode == "train":
            sketch, photo = random_jitter_pair(sketch, photo)  # PIL RGB 256x256
        sketch_t = pil_to_tensor(sketch)                    # [3, 256, 256]
        photo_t = pil_to_tensor(photo)                      # [3, 256, 256]
        return sketch_t, photo_t

def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    """Buat DataLoader train dan val dari lokasi dataset.

    Setiap batch: (sketch, photo), masing-masing [B, 3, 256, 256],
    float32, rentang [-1, 1].
    Return: (train_loader, val_loader)
    """
    assert batch_size > 0, "batch_size harus positif"

    train_paths, val_paths = select_edges2shoes_subset(data_root)
    train_ds = Edges2ShoesDataset(train_paths, mode="train")
    val_ds = Edges2ShoesDataset(val_paths, mode="val")

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # acak urutan tiap epoch
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,        # buang batch terakhir yang tidak penuh
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # urutan tetap, hasil mudah dibandingkan
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,       # semua data val ikut terukur
    )
    return train_loader, val_loader

def show_batch(
    loader: DataLoader,
    n_pairs: int = 6,
    save_path: Optional[str] = None,
) -> None:
    """Tampilkan satu batch dari loader sebagai grid 2 baris.

    Baris atas berisi sketsa, baris bawah berisi foto pasangannya
    (kolom yang sama = pasangan yang sama).

    Args:
        loader: DataLoader yang menghasilkan (sketch, photo).
        n_pairs: Jumlah pasangan yang ditampilkan (dibatasi batch size).
        save_path: Jika diisi, grid juga disimpan ke path ini.
    """
    if n_pairs < 1:
        raise ValueError("n_pairs harus >= 1")

    sketch, photo = next(iter(loader))  # masing-masing [B, 3, 256, 256]
    assert sketch.ndim == 4 and sketch.shape[1:] == (3, 256, 256), (
        f"Shape sketsa tidak sesuai kontrak: {tuple(sketch.shape)}"
    )
    assert sketch.shape == photo.shape, (
        f"Shape sketsa {tuple(sketch.shape)} != foto {tuple(photo.shape)}"
    )

    n = min(n_pairs, sketch.size(0))
    fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 5.2), squeeze=False)

    for i in range(n):
        # tensor_to_image: [3, 256, 256] di [-1, 1] -> [256, 256, 3] di [0, 1]
        axes[0, i].imshow(tensor_to_image(sketch[i]))
        axes[1, i].imshow(tensor_to_image(photo[i]))
        axes[0, i].set_title(f"#{i}", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Sketsa")
    axes[1, 0].set_ylabel("Foto")

    fig.suptitle(f"Satu batch ({n} dari {sketch.size(0)} pasangan)")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()