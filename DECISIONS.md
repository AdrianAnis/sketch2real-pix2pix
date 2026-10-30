# Catatan Keputusan Desain

Daftar keputusan penting proyek beserta alasannya, sebagai bahan menjawab "kenapa memilih ini?" saat presentasi.
Tambahkan baris baru setiap ada keputusan yang mengubah kontrak, arsitektur, atau hyperparameter final, lalu kabari grup.

| ID | Keputusan | Alasan | Sumber |
|---|---|---|---|
| D-001 | Model Pix2Pix (Conditional GAN) | Termasuk pilihan di panduan proyek; dataset berpasangan tersedia dan hasilnya mudah dievaluasi secara visual | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-002 | Dataset edges2shoes, subset ±10.000 pasangan, 256×256 | Sudah berpasangan (sketsa–foto) dan cukup ringan untuk GPU gratis. Batasan: model hanya bisa menghasilkan sepatu | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-003 | GitHub untuk kode, Kaggle Notebook (GPU T4/P100) untuk training | Gratis, GPU cukup untuk 256×256, dan repo bisa langsung di-clone di notebook | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-004 | Generator U-Net 8 lapis dengan skip connection | Skip connection membawa detail garis sketsa langsung ke decoder sehingga struktur gambar tetap terjaga | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-005 | Discriminator PatchGAN 70×70 | Menilai keaslian per patch lokal (tekstur) dengan parameter lebih sedikit daripada menilai satu gambar utuh | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-006 | Loss BCEWithLogits (adversarial) + 100 × L1 | L1 menjaga hasil tetap mirip foto asli, adversarial membuat hasil tajam; bobot 100 mengikuti paper Pix2Pix | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-007 | Adam, lr 2e-4, betas (0.5, 0.999) | Pengaturan umum untuk GAN; β1 0.5 membuat training lebih stabil | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-008 | Kontrak data `(sketch, photo)`, `[B, 3, 256, 256]`, float32, rentang `[-1, 1]` | Supaya ketiga modul bisa dikerjakan paralel; rentang `[-1, 1]` cocok dengan output Tanh di Generator | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-009 | Kelompok 3 orang: Rian merangkap peran Anggota 1 dan 4 | Panduan mengasumsikan 4 anggota, sedangkan kelompok kami 3 orang | [log 00](logs/adrian/00-perencanaan-arsitektur.md) |
| D-010 | Log interaksi: 1 file = 1 sesi chat, setiap komponen dan bug menjadi entri tersendiri | Template dosen memakai kolom Tanggal Sesi dan Tautan Sesi Chat, jadi satuan log adalah sesi chat | [logs/TEMPLATE.md](logs/TEMPLATE.md) |
