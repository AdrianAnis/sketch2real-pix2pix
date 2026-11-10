# sketch2real-pix2pix

Proyek 1 — Vibe Coding Generative AI from Scratch (CAK4KBB3) — **Kelompok 6**
Model **Pix2Pix (Conditional GAN)** untuk mengubah sketsa (edges) menjadi foto sepatu realistis (dataset edges2shoes, 256×256).

## Anggota & Pembagian Tugas

| Nama | Peran | File |
|---|---|---|
| Rian (Adrian Anis Pratama) | Ketua — Prompt Engineer & LLM Navigator, Lead Documentarian & Error Handler | `src/model.py`, `src/infer.py`, `logs/`, `DECISIONS.md`, `WORKFLOW.md` |
| Naura | Data Pipeline & Preprocessing Engineer | `src/data.py`, `src/metrics.py` |
| Meyra | Training Loop & Hyperparameter Tuning Engineer | `src/train.py` |

Kelompok terdiri dari 3 orang, sehingga Rian merangkap peran Anggota 1 dan 4 (lihat D-009 di `DECISIONS.md`).

## Hasil Ringkas

| Hal | Hasil |
|---|---|
| Training | 12 epoch, batch 16, lr 2e-4, GPU Kaggle T4 (±6 menit per epoch) |
| `val_L1` (mode `train()`, epoch 12) | 0,1619 |
| PSNR / SSIM (200 gambar val) | **16,98 ± 3,92 dB** / **0,7128 ± 0,0946** |
| Tuning batch size 8 vs 16 (PSNR / SSIM, batch validasi sama) | Batch 16 sedikit lebih baik: 16,98 vs 16,39 dB dan 0,7128 vs 0,6983; satu run per konfigurasi (D-018) |
| Perpanjangan training 12 ke 24 epoch (PSNR / SSIM) | Tidak ada perbaikan berarti: 16,87 vs 16,98 dB dan 0,7163 vs 0,7128; hasil utama tetap 12 epoch (D-021) |

Grafik loss dan sampel gambar ada di `results/` (`loss_curves.png`, `samples/epoch_004.png`, `epoch_008.png`, `epoch_012.png`); angka metrik di `results/metrics.json`; hasil eksperimen 24 epoch ada di `results/e24/` dan `results/metrics_e24.json`. Alasan setiap keputusan desain ada di `DECISIONS.md`.

Catatan: pada epoch 1-3 `val_L1` dihitung dengan kode lama (mode `eval()`), sehingga nilainya tidak sebanding dengan epoch 4-12 (lihat D-017).

## Kontrak Antar Modul

Semua anggota wajib mengikuti kontrak ini supaya modul bisa langsung digabung.

- **Dataset:** edges2shoes (subset 10.000 pasangan train, 200 val), resolusi 256×256
- **`src/data.py`:** `build_dataloaders(data_root, batch_size, num_workers=2) -> (train_loader, val_loader)`
  - setiap batch `(sketch, photo)`, shape `[B, 3, 256, 256]`, dtype `float32`, rentang `[-1, 1]`
  - `data_root` berisi subfolder `train/` dan `val/` (tiap file 512×256: sketsa kiri, foto kanan)
- **`src/model.py`:** `Generator(in_channels=3, out_channels=3, base=64)` (U-Net), `Discriminator(in_channels=6, base_ch=64)` (PatchGAN 70×70), dan `init_weights(m)`
  - `Generator(sketch) -> fake_photo`, shape sama dengan input
  - `Discriminator(sketch, photo) -> patch_logits` `[B, 1, 30, 30]`, input digabung di dimensi channel (6 channel), tanpa Sigmoid
- **`src/train.py`:** loss `BCEWithLogits + 100 × L1` (loss D dikali 0,5), Adam (lr 2e-4, β 0,5/0,999); validasi dan sampel dalam mode `train()` (D-017)
- **`src/infer.py`:** demo sketsa → foto; **`src/metrics.py`:** evaluasi PSNR dan SSIM

## Struktur Repo

```
sketch2real-pix2pix/
├── assets/
│   └── context_sketches/  # 7 sketsa pendamping untuk inferensi (D-019)
├── logs/
│   ├── adrian/            # Log sesi LLM Rian
│   ├── meyra/             # Log sesi LLM Meyra
│   ├── naura/             # Log sesi LLM Naura
│   ├── README.md          # Indeks semua sesi (urut tanggal)
│   ├── SYSTEM_PROMPT.md   # System prompt tim untuk setiap sesi LLM
│   └── TEMPLATE.md        # Template log per sesi
├── notebooks/             # Notebook Kaggle untuk menjalankan training
├── results/               # Grafik loss, sampel gambar, metrik (metrics.json)
├── src/
│   ├── __init__.py
│   ├── data.py            # Dataset & DataLoader
│   ├── model.py           # Generator & Discriminator
│   ├── train.py           # Training loop, loss, checkpoint, sampel per epoch
│   ├── infer.py           # Demo sketsa -> foto (CLI)
│   └── metrics.py         # Evaluasi PSNR dan SSIM (CLI)
├── tests/                 # Tes per modul (data, model, train, infer, metrics)
├── .gitignore
├── DECISIONS.md           # Catatan keputusan desain & alasannya
├── README.md
├── requirements.txt
└── WORKFLOW.md            # Timeline, checklist per fase, siklus kerja
```

File besar tidak disimpan di repo (`.gitignore`): `checkpoints/`, `*.pt`, `*.zip`, dan `data/`.

## Instalasi & Tes

```bash
git clone https://github.com/AdrianAnis/sketch2real-pix2pix.git
cd sketch2real-pix2pix
python -m pip install -r requirements.txt
python -m pytest tests -q
```

## Training di Kaggle

Aktifkan **GPU** dan **Internet** di Kaggle Notebook, tambahkan dataset `balraj98/edges2shoes-dataset` sebagai Input, lalu:

```python
!git clone https://github.com/AdrianAnis/sketch2real-pix2pix.git
%cd sketch2real-pix2pix
!pip install -q -r requirements.txt
!python -m pytest tests -q
!python -m src.train --data_root /kaggle/input/datasets/balraj98/edges2shoes-dataset --epochs 12
```

Opsi lain: `--batch_size` (default 16), `--seed` (default 42), `--ckpt_dir`, `--results_dir`, `--resume checkpoints/latest.pt` (melanjutkan; `--epochs` adalah target total), `--keep_all`. Keluaran training: `checkpoints/latest.pt`, `results/loss_curves.png`, dan `results/samples/epoch_XXX.png`. File di `/kaggle/working/` hilang saat sesi ditutup, jadi unduh hasilnya sebelum itu.

## Demo: sketsa → foto

Butuh bobot Generator (`G_only.pt` atau `checkpoints/latest.pt`) yang tidak disimpan di repo (minta ke tim).

```bash
python -m src.infer --ckpt G_only.pt --input sketsa.png --output hasil.png --seed 0
```

Keluarannya gambar berdampingan (sketsa | hasil). Sketsa sebaiknya garis hitam di latar putih (latar gelap dideteksi otomatis, atau paksa dengan `--invert`). Secara bawaan sketsa dijalankan bersama 7 sketsa pendamping dari `assets/context_sketches/` (D-019); `--no_context` menjalankan sketsa sendirian (warna bisa lebih jenuh dan bernoda). Untuk membandingkan dengan foto asli: `--val_dir <folder val> --val_index N`.

## Evaluasi: PSNR dan SSIM

Butuh 200 gambar validasi edges2shoes di `data/edges2shoes/val/` (ambil dari dataset Kaggle) dan bobot Generator:

```bash
python -m src.metrics --ckpt G_only.pt --val_dir data/edges2shoes/val --batch_size 16 --seed 42 --device cpu
```

Hasil disimpan ke `results/metrics.json`. Generator dijalankan dalam mode `train()` (D-017) dengan batch 16, sehingga hasilnya sedikit bergantung pada ukuran batch dan seed.

## Aturan Kerja

1. Semua kode ditulis lewat vibe coding dengan LLM, **tanpa copy-paste dari tutorial/GitHub orang lain**.
2. Setiap sesi LLM diawali dengan isi `logs/SYSTEM_PROMPT.md`.
3. Setiap sesi dicatat di `logs/<nama>/` memakai `logs/TEMPLATE.md`, lengkap dengan share link.
4. Commit memakai akun GitHub masing-masing.

## Pola Menulis Prompt

Setiap prompt sebaiknya memuat 4 unsur:

1. **Konteks**: siapa kamu dan file apa yang dikerjakan
2. **Target spesifik**: satu komponen kecil, bukan satu file utuh
3. **Minta penjelasan**: kenapa pendekatan itu dipilih (bahan presentasi)
4. **Minta bukti jalan**: tes shape atau contoh output

Contoh:
> "Aku [nama], mengerjakan src/[file].py. Sekarang buat [komponen kecil].
> Jelaskan dulu [konsep yang belum dipahami], lalu tulis kodenya
> beserta tes sederhana untuk memastikan [shape/output] benar."

Saat eror:
> "Kode [komponen] menghasilkan eror berikut: [tempel eror utuh].
> Jelaskan penyebabnya, lalu beri perbaikan minimal."
