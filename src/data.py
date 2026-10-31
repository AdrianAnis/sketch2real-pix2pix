from __future__ import annotations

import random
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