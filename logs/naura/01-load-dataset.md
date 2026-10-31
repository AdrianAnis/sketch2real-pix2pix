- **Tanggal Sesi:** DD/MM/YYYY (atau rentang, misal 01/11–02/11/2026)
- **Anggota:** [Nama]
- **Target Modul:** [contoh: src/model.py — Generator U-Net]
- **LLM:** [contoh: Claude (claude.ai) / ChatGPT / Gemini]
- **Tautan Sesi Chat:** [share link, atau nama file ekspor PDF/Markdown]

## Kronologi Prompt & Respons

### 1. Tahap Inisiasi (Meminta Kode Awal)

#### 1a. [Nama komponen, misal: Blok Encoder]

- **Prompt Kelompok:**
  > [tempel prompt yang kamu kirim ke AI]
- **Alasan Khusus:** [kenapa memakai pendekatan/instruksi ini]
- **Output Model:** [ringkasan kode yang diberikan AI, atau cuplikan bagian pentingnya]

#### 1b. [Komponen berikutnya]

- **Prompt Kelompok:**
  > [tempel prompt]
- **Alasan Khusus:**
- **Output Model:**

### 2. Tahap Penanganan Eror / Debugging

#### 2a. [Nama eror singkat + komponennya, misal: Shape mismatch di skip connection (komponen 1c)]

- **Pesan Eror dari Notebook:**
  ```
  [tempel pesan eror utuh]
  ```
- **Prompt Perbaikan ke LLM:**
  > [tempel prompt yang kamu kirim untuk memperbaiki eror]
- **Analisis & Solusi Kelompok:** [penyebab erornya apa, bagaimana AI membantu, dan apa pelajarannya]

### 3. Tahap Optimasi (opsional)

#### 3a. [Nama optimasi]

- **Prompt:**
  > [tempel prompt]
- **Hasil:** [apa yang membaik, misalnya loss turun atau kode lebih rapi]
