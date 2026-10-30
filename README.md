# sketch2real-pix2pix

Proyek 1 — Vibe Coding Generative AI from Scratch (CAK4KBB3)
Model **Pix2Pix (Conditional GAN)** untuk mengubah sketsa (edges) menjadi gambar realistis.

## Anggota & Pembagian Tugas

| Nama | Peran | File |
|---|---|---|
| Rian (Adrian Anis Pratama) | Ketua — Prompt Engineer & LLM Navigator, Lead Documentarian & Error Handler | `src/model.py`, `logs/` |
| Naura | Data Pipeline & Preprocessing Engineer | `src/data.py` |
| Meyra | Training Loop & Hyperparameter Tuning Engineer | `src/train.py` |

## Kontrak Antar Modul

Semua anggota wajib mengikuti kontrak ini supaya modul bisa langsung digabung.

- **Dataset:** edges2shoes (subset ±10.000 pasangan), resolusi 256×256
- **Output `src/data.py`:** DataLoader yang menghasilkan pasangan `(sketch, photo)`
  - shape `[B, 3, 256, 256]`, dtype `float32`, rentang nilai `[-1, 1]`
- **Output `src/model.py`:** class `Generator()` (U-Net) dan `Discriminator()` (PatchGAN 70×70)
  - `Generator(sketch) -> fake_photo`, shape sama dengan input
  - `Discriminator(sketch, photo) -> patch_logits`, input digabung di dimensi channel (6 channel)
- **`src/train.py`:** import dari `src.data` dan `src.model`, loss `BCE + 100 × L1`, Adam (lr 2e-4, β1 0.5)

## Struktur Repo

```
sketch2real-pix2pix/
├── logs/
│   ├── adrian/          # Log sesi LLM Rian
│   ├── meyra/           # Log sesi LLM Meyra
│   ├── naura/           # Log sesi LLM Naura
│   ├── README.md        # Indeks semua sesi (urut tanggal)
│   ├── SYSTEM_PROMPT.md # System prompt tim untuk setiap sesi LLM
│   └── TEMPLATE.md      # Template log per sesi
├── notebooks/           # Notebook Kaggle untuk menjalankan training
├── results/             # Grafik loss, sampel gambar, metrik
├── src/
│   ├── __init__.py
│   ├── data.py          # Dataset & DataLoader
│   ├── model.py         # Generator & Discriminator
│   └── train.py         # Training loop, loss, metrik, checkpoint
├── tests/               # Tes per modul (test_data.py, test_model.py, test_train.py)
├── .gitignore
├── DECISIONS.md         # Catatan keputusan desain & alasannya
├── README.md
├── requirements.txt
└── WORKFLOW.md          # Timeline, checklist per fase, siklus kerja
```

## Menjalankan di Kaggle

Aktifkan GPU di Kaggle Notebook, lalu:

```python
!git clone https://github.com/<username>/sketch2real-pix2pix.git
%cd sketch2real-pix2pix
!pip install -q -r requirements.txt
!python -m pytest tests/
!python -m src.train
```

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