# Alur Kerja & Timeline

## Ringkasan Fase

| Fase | Fokus                       | Target Selesai | Status |
| ---- | --------------------------- | -------------- | ------ |
| 0    | Perencanaan & struktur repo | 31/10/2026     | ✅     |
| 1    | Pengerjaan modul (paralel)  | 05/11/2026     | ✅     |
| 2    | Integrasi di Kaggle         | 05/11/2026     | ✅     |
| 3    | Training & tuning           | 07/11/2026     | ⬜     |
| 4    | Demo, laporan, presentasi   | 08/11/2026     | ⬜     |

## Checklist per Fase

### Fase 0 — Perencanaan

- [x] Struktur repo, kontrak antar modul, system prompt — **Rian**

### Fase 1 — Pengerjaan Modul (paralel)

**`src/data.py` — Naura**

- [x] Download edges2shoes, ambil subset ±10.000 gambar
- [x] Baca satu gambar, potong jadi sketsa (kiri) dan foto (kanan)
- [x] Resize ke 256×256, normalisasi ke [-1, 1]
- [x] Augmentasi (random jitter + flip horizontal)
- [x] Class Dataset + DataLoader
- [x] Tampilkan satu pasangan (sketsa & foto) untuk cek visual
- [x] `tests/test_data.py`: tes shape `[B, 3, 256, 256]`, dtype `float32`, rentang `[-1, 1]`

**`src/model.py` — Rian**

- [x] Blok downsampling (encoder)
- [x] Blok upsampling (decoder) + dropout
- [x] Generator U-Net utuh dengan skip connection
- [x] Discriminator PatchGAN 70×70
- [x] Inisialisasi bobot (normal, std 0.02)
- [x] Tes shape model: `tests/test_model.py` (blok), `test_generator.py`, `test_discriminator.py`, `test_init.py`

**`src/train.py` — Meyra**

- [v] Setup notebook Kaggle (GPU aktif)
- [v] Satu langkah training dengan tensor dummy
- [v] Loss: BCEWithLogits + 100 × L1
- [v] Update Discriminator lalu Generator secara bergantian
- [v] Loop epoch + simpan checkpoint
- [v] `tests/test_train.py`: satu langkah training dengan tensor dummy, loss bernilai finite dan bobot model berubah

### Fase 2 — Integrasi

- [v] Ganti tensor dummy di `train.py` dengan DataLoader dari `data.py` — **Meyra**
- [v] Jalankan ketiga modul bersama di Kaggle, perbaiki eror yang muncul — **Semua**

### Fase 3 — Training & Tuning

- [v] Training penuh — **Meyra**
- [v] Simpan grafik loss dan grid sampel ke `results/` — **Meyra**
- [ ] Eksperimen tuning (lr / batch size), catat hasilnya dan pilihan akhirnya di `DECISIONS.md` — **Meyra**
- [v] Hitung PSNR/SSIM di data validasi — **Naura**

### Fase 4 — Demo, Laporan, Presentasi

- [v] `src/infer.py`: demo sketsa → foto — **Rian**
- [ ] Laporan akhir — **Rian**
- [ ] Latihan presentasi — **Semua**

## Siklus Kerja Setiap Sesi

1. `git pull`
2. Buka chat baru dengan AI, tempel isi `logs/SYSTEM_PROMPT.md`, lalu salin `logs/TEMPLATE.md` ke `logs/<nama>/XX-<topik>.md`
   (kalau melanjutkan topik kemarin, buka lagi chat dan file log yang sama; ganti hari tidak harus ganti chat)
3. Ambil item checklist pertama yang belum dicentang, minta ke AI
4. Tempel kode dari AI ke file milikmu, jalankan atau tes
5. Eror → tempel pesan eror ke chat yang sama, minta penjelasan dan perbaikan
6. Berhasil → centang item (`[x]`), tulis entrinya di file log (1a, 1b, 2a, ...), lalu `git add`, `git commit`, `git push`
7. Ulangi langkah 3–6 untuk item berikutnya di chat yang sama
8. Chat selesai (topik beres atau chat sudah terlalu panjang) → Share chat, tempel link di log, lengkapi tanggal sesi
   (share link hanya memuat isi chat sampai saat link dibuat, jadi share setelah chat selesai atau perbarui link-nya)
9. Kabari Rian untuk dimasukkan ke `logs/README.md`

## Aturan

- Hanya edit file milik sendiri. Kalau perlu mengubah file orang lain, bilang dulu di grup.
- Selalu `git pull` sebelum mulai kerja.
- Commit kecil-kecil, setiap satu item checklist selesai.
- Sebelum push, jalankan `python -m pytest tests/` dan pastikan semua tes lulus.
- Ada keputusan desain baru (mengubah kontrak, arsitektur, atau hyperparameter final)? Tambahkan baris di `DECISIONS.md` dan kabari grup.
- Kalau push ditolak: `git pull --rebase` lalu `git push`.
