# Indeks Log Interaksi LLM

Daftar semua sesi LLM kelompok, diurutkan berdasarkan tanggal.
Kolom **Commit** berisi hash commit pertama yang menambahkan file log tersebut (`git log --diff-filter=A --format=%h -- <file>`).

| No | Tanggal | Anggota | Modul | Tahap | File | Commit |
|---|---|---|---|---|---|---|
| 0 | 31/10/2026 | Rian | Perencanaan arsitektur, pembagian tugas, struktur repo | Inisiasi | [adrian/00-perencanaan-arsitektur.md](adrian/00-perencanaan-arsitektur.md) | `3defc19` |
| 1 | 31/10/2026 | Naura | `data.py`: load dataset, subset, crop sketsa/foto, normalisasi | Inisiasi, Debugging | [naura/01-load-dataset.md](naura/01-load-dataset.md) | `d582b9b` |
| 2 | 01/11/2026 | Rian | `model.py`: blok encoder dan decoder | Inisiasi, Debugging | [adrian/01-encoder-decoder.md](adrian/01-encoder-decoder.md) | `335bb76` |
| 3 | 01/11/2026 | Rian | `model.py`: Generator U-Net | Inisiasi | [adrian/02-generator-unet.md](adrian/02-generator-unet.md) | `ce60644` |
| 4 | 01/11/2026 | Rian | `model.py`: Discriminator PatchGAN dan inisialisasi bobot | Inisiasi | [adrian/03-discriminator-init.md](adrian/03-discriminator-init.md) | `6a036dd` |
| 5 | 01/11/2026 | Naura | `data.py`: augmentasi, Dataset, DataLoader | Inisiasi, Debugging | [naura/02-augmentasi-dataloader.md](naura/02-augmentasi-dataloader.md) | `5e9a252` |
| 6 | 01/11/2026 | Naura | `data.py`: cek visual dan tes Dataset | Inisiasi, Debugging | [naura/03-cek-visual-dan-test.md](naura/03-cek-visual-dan-test.md) | `ef35b40` |
| 7 | 05/11/2026 | Meyra | `train.py`: langkah training, loop epoch, checkpoint, sampel | Inisiasi, Debugging | [meyra/01-training-loop.md](meyra/01-training-loop.md) | `1ab8aaf` |
| 8 | 05/11/2026 | Meyra | Integrasi dan uji jalan di Kaggle (GPU) | Inisiasi, Debugging | [meyra/02-integrasi-kaggle.md](meyra/02-integrasi-kaggle.md) | `d519681` |
| 9 | 05/11-06/11/2026 | Meyra | `train.py`: validasi dan sampel mode `train()` (D-017) | Inisiasi | [meyra/03-perbaikan-mode-eval.md](meyra/03-perbaikan-mode-eval.md) | `4a59215` |
| 10 | 06/11/2026 | Rian | `infer.py`: demo sketsa ke foto | Inisiasi | [adrian/04-infer.md](adrian/04-infer.md) | `631bf16` |
| 11 | 07/11/2026 | Naura | `metrics.py`: evaluasi PSNR dan SSIM | Inisiasi, Debugging | [naura/04-metrik-psnr-ssim.md](naura/04-metrik-psnr-ssim.md) | `3d987e2` |
| 12 | 07/11/2026 | Meyra | Tuning batch size 8 vs 16, pengamanan bobot | Inisiasi, Debugging | [meyra/04-tuning-batchsize.md](meyra/04-tuning-batchsize.md) | `c8cedd8` |
| 13 | 10/11/2026 | Meyra | Lanjut training 12 ke 24 epoch (`--resume`), pembacaan riwayat val_L1 | Inisiasi | [meyra/05-lanjut-training-24-epoch.md](meyra/05-lanjut-training-24-epoch.md) | belum di-commit |
