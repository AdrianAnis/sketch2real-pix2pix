# LOG INTERAKSI LLM — KELOMPOK 6

- **Tanggal Sesi:** 31/10/2026
- **Anggota:** Rian (Adrian Anis Pratama)
- **Target Modul:** Perencanaan arsitektur, pembagian tugas, dan struktur repo
- **LLM:** Claude (claude.ai)
- **Tautan Sesi Chat:** https://claude.ai/share/3eda9f1b-3076-4258-a5dc-9aa7be161366 

> Catatan: prompt pada log ini disunting untuk keterbacaan tanpa mengubah isi. Teks asli dapat dilihat pada tautan sesi.

## Kronologi Prompt & Respons

### 1. Tahap Inisiasi (Perencanaan)

- **Prompt Kelompok:**
  > "Berdasarkan panduan proyek terlampir, kelompok kami ingin membangun model generatif yang mengubah gambar sketsa menjadi gambar realistis menggunakan Pix2Pix. Apakah ini layak dikerjakan, dan bagaimana rencana pengerjaannya?"
- **Alasan Khusus:** Sebelum menulis kode, kelompok perlu memastikan Pix2Pix layak dikerjakan dengan hardware gratis dan menentukan dataset, platform, serta arsitektur yang sesuai panduan dosen.
- **Output Model:**
  - Dataset: edges2shoes (subset ±10.000 pasangan, 256×256)
  - Platform: GitHub untuk kode, Kaggle Notebook (GPU T4/P100) untuk training
  - Arsitektur: Generator U-Net 8 lapis dengan skip connection, Discriminator PatchGAN 70×70
  - Loss: BCE adversarial + 100 × L1; Adam lr 2e-4, β1 0.5
  - Saran pengerjaan bertahap per modul agar log interaksi memiliki kronologi yang jelas

### 2. Tahap Penyesuaian Pembagian Tugas

- **Prompt Kelompok:**
  > "Kelompok kami terdiri dari 3 orang, bukan 4. Saya sebagai ketua sekaligus penanggung jawab dokumentasi, Naura di peran kedua, dan Meyra di peran ketiga (training). Bagaimana pemetaan job desk yang tepat?"
- **Alasan Khusus:** Panduan mengasumsikan 4 anggota, sedangkan kelompok kami 3 orang, sehingga job desk perlu dipetakan ulang.
- **Output Model:**
  - Rian: peran Anggota 1 dan 4 (prompt engineer, arsitektur `src/model.py`, dokumentasi)
  - Naura: data pipeline (`src/data.py`)
  - Meyra: training loop dan tuning (`src/train.py`)
  - Kontrak antar modul: data berbentuk `(sketch, photo)` dengan shape `[B, 3, 256, 256]` dan rentang `[-1, 1]`

### 3. Tahap Revisi Struktur Repo

- **Prompt Kelompok:**
  > "Struktur repo terlihat kurang rapi karena semua file berada di root. Selain itu, log akan bertambah banyak; apakah sebaiknya dipisah per anggota? Jelaskan juga fungsi setiap folder."
- **Alasan Khusus:** Struktur awal menaruh semua file `.py` di root, dan log belum dipisah per anggota.
- **Hasil:**
  - Kode dipindah ke package `src/` dengan `__init__.py`; training dijalankan via `python -m src.train`
  - Log dipisah per anggota (`logs/adrian`, `logs/naura`, `logs/meyra`) dengan indeks kronologis di `logs/README.md`
  - Struktur pembanding dari kelompok lain dievaluasi; yang diadopsi hanya `tests/` (dan `configs/` nanti), karena folder lain tumpang tindih atau seharusnya tidak di-push (`.venv`, `.pytest_cache`, `.vscode`)
  - Dibuat `logs/SYSTEM_PROMPT.md` sebagai system prompt bersama agar gaya kode ketiga anggota konsisten

### 4. Tahap Penyusunan Alur Kerja Tim

- **Prompt Kelompok:**
  > "Bagaimana anggota lain menyusun prompt mereka sendiri? Apakah perlu template?"
  > "Jelaskan alur kerja proyek ini agar bisa saya sampaikan ke anggota kelompok. Apakah perlu dokumen perencanaan, dan di mana sebaiknya disimpan?"
- **Alasan Khusus:** Ketiga anggota bekerja paralel di modul berbeda, sehingga dibutuhkan pola prompt dan alur kerja yang seragam agar hasil kode dan log konsisten tanpa menyeragamkan isi prompt.
- **Hasil:**
  - Ditambahkan bagian "Pola Menulis Prompt" di `README.md` (konteks, target spesifik, minta penjelasan, minta bukti jalan), bukan prompt jadi, agar variasi prompt tiap anggota tetap terjaga
  - Ditambahkan kolom **LLM** di `logs/TEMPLATE.md`; lampiran riwayat chat berupa share link atau ekspor PDF/Markdown sesuai panduan
  - Dibuat `WORKFLOW.md` berisi 4 fase (pengerjaan modul paralel → integrasi di Kaggle → training & tuning → demo & laporan), siklus kerja per sesi, aturan Git, dan checklist detail per modul
  - Ditegaskan bahwa output sistem berupa gambar RGB 256×256, dan model hanya mampu menghasilkan objek sepatu sesuai dataset edges2shoes (dicatat sebagai batasan)

## Catatan

Sesi ini belum menghasilkan kode model sehingga tidak ada eror kompilasi. Implementasi dimulai pada sesi `01-generator-unet.md`.