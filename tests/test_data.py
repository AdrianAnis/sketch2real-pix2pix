"""Tes untuk src/data.py. Semua tes memakai gambar palsu di folder sementara."""
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import Edges2ShoesDataset

N_IMAGES = 4


@pytest.fixture
def fake_paths(tmp_path: Path) -> List[str]:
    """Buat N_IMAGES gambar palsu 512x256 (kiri = sketsa, kanan = foto)."""
    rng = np.random.default_rng(0)  # seed tetap -> isi gambar selalu sama
    paths = []
    for i in range(N_IMAGES):
        # [256, 512, 3] uint8; noise supaya crop/flip benar-benar mengubah piksel
        arr = rng.integers(0, 256, size=(256, 512, 3), dtype=np.uint8)
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(arr).save(path)
        paths.append(str(path))
    return paths


def test_len(fake_paths):
    assert len(Edges2ShoesDataset(fake_paths, mode="val")) == N_IMAGES


@pytest.mark.parametrize("mode", ["train", "val"])
def test_item_shape_dtype_range(fake_paths, mode):
    ds = Edges2ShoesDataset(fake_paths, mode=mode)
    sketch, photo = ds[0]
    for t in (sketch, photo):
        assert t.shape == (3, 256, 256)
        assert t.dtype == torch.float32
        assert t.min() >= -1.0 and t.max() <= 1.0


def test_val_is_deterministic(fake_paths):
    ds = Edges2ShoesDataset(fake_paths, mode="val")
    sketch_1, photo_1 = ds[0]
    sketch_2, photo_2 = ds[0]
    assert torch.equal(sketch_1, sketch_2)
    assert torch.equal(photo_1, photo_2)


def test_train_differs_for_same_index(fake_paths):
    ds = Edges2ShoesDataset(fake_paths, mode="train")
    first_sketch, first_photo = ds[0]
    differs = False
    for _ in range(20):
        sketch, photo = ds[0]
        if not torch.equal(sketch, first_sketch) or not torch.equal(photo, first_photo):
            differs = True
            break
    assert differs, "Mode train menghasilkan keluaran identik 20 kali: jitter tidak aktif?"