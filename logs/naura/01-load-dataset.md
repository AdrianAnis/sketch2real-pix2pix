# LOG INTERAKSI LLM — KELOMPOK 4

- **Tanggal Sesi:** 31/10/2026
- **Anggota:** Naura
- **Target Modul:** src/data.py — Pemilihan subset dataset edges2shoes dan pemotongan sketsa/foto
- **LLM:** Claude (claude.ai)
- **Tautan Sesi Chat:** https://claude.ai/share/18627179-f7da-419f-8b02-64f05fe5039a

## Kronologi Prompt & Respons

### 1. Tahap Inisiasi (Meminta Kode Awal)

#### 1a. Pemilihan subset dataset edges2shoes

- **Prompt Kelompok:**
  > Aku Naura, mengerjakan src/data.py (data pipeline) untuk proyek ini. Training nanti dijalankan di Kaggle Notebook, jadi kodenya harus bisa jalan di sana.
  >
  > Aku mulai dari item pertama: mengambil dataset edges2shoes dan memilih subset ±10.000 gambar untuk train, ditambah gambar val-nya.
  >
  > Jelaskan dulu:
  >
  > 1. Seperti apa struktur folder dan isi satu file gambar di dataset edges2shoes.
  > 2. Cara mendapatkan dataset ini di Kaggle, dan mana yang lebih praktis.
  > 3. Kenapa cukup memakai subset, dan kenapa pengambilan subset perlu seed tetap.
  >
  > Setelah itu tulis kodenya sebagai satu fungsi. Lokasi dataset harus jadi parameter fungsi, jangan ditulis mati di kode. Terakhir, beri cara mengecek bahwa jumlah gambar train = 10.000, val = 200, dan semua filenya ada.
  >
  > (Prompt ini dikirim setelah isi `logs/SYSTEM_PROMPT.md` ditempel di pesan yang sama.)
- **Alasan Khusus:** Data harus siap lebih dulu sebelum bisa dipotong, diaugmentasi, dan dimuat ke model. Lokasi dataset dijadikan parameter karena path di Kaggle berbeda dengan di laptop. Penjelasan diminta sebelum kode agar kami paham alasan desainnya untuk presentasi.
- **Output Model:**
  - AI mencari struktur dataset lewat web (sumber yang disebut: repo DeepImage Pix2Pix dan BicycleGAN). Struktur: folder `train/` (±49.8 ribu file) dan `val/` (200 file), tiap file `.jpg` berukuran 512×256 berisi sketsa (kiri) dan foto (kanan) bersebelahan.
  - Dua cara mendapatkan dataset di Kaggle: Add Input atau kagglehub/API. AI merekomendasikan Add Input (tanpa download, tanpa internet, semua anggota melihat data yang sama; folder input read-only).
  - Alasan subset: domain sempit (sepatu), batas waktu dan GPU Kaggle, dan trade-off variasi data yang lebih sedikit. Alasan seed tetap: reproducibility dan konsistensi saat resume training; daftar file di-`sorted` dulu dan memakai `random.Random(seed)` lokal agar RNG global tidak terganggu.
  - Kode fungsi `select_edges2shoes_subset(data_root, n_train=10_000, seed=42, ...)` yang mengembalikan `(train_paths, val_paths)`, berupa daftar path terurut. Fungsi hanya memilih file; pemotongan sketsa/foto dikerjakan di komponen berikutnya.
  - Blok verifikasi: jumlah train = 10.000 dan val = 200, tanpa duplikat, semua file ada, hasil identik untuk seed yang sama, dan ukuran file 512×256.
  - AI belum bisa memastikan nama dataset di Kaggle dan menyarankan kami mengeceknya sendiri lewat `!ls`.
- **Hasil di Kaggle:** `OK: train = 10000 | val = 200`. Dataset yang dipakai: "Edges2shoes Dataset" (balraj98).
- **Keputusan:** Dataset nanti menerima list Path ini langsung, tanpa menyalin subset ke `/kaggle/working`.

#### 1b. Membaca satu gambar dan memotongnya jadi sketsa dan foto

- **Prompt Kelompok:**
  > Subset sudah jalan. Item berikutnya: membaca satu gambar lalu memotongnya jadi sketsa dan foto. Jelaskan dulu kenapa pasangan gambar disimpan dalam satu file dan bagaimana memotongnya tepat di tengah. Lalu tulis kodenya dan beri cara menampilkan sketsa dan foto berdampingan untuk memastikan keduanya tidak tertukar dan masing-masing berukuran 256x256.
- **Alasan Khusus:** Setelah file dipilih, langkah berikutnya memisahkan sketsa (input) dan foto (target) dari satu file 512×256. Pemisahan harus tepat dan tidak tertukar karena menentukan arah pembelajaran model.
- **Output Model:**
  - Alasan format satu file (aligned pair): pasangan terkunci secara fisik sehingga tidak mungkin tertukar, hanya satu kali baca disk per sampel, dan augmentasi nanti bisa diterapkan sekaligus ke keduanya.
  - Pemotongan di titik tengah memakai `w // 2` (sketsa = kolom 0–255, foto = kolom 256–511), tanpa resize karena tiap setengah sudah 256×256, dan dikonversi ke RGB agar selalu 3 channel.
  - Fungsi `load_pair(path)` yang mengembalikan dua PIL Image (belum tensor), plus konstanta `IMG_SIZE = 256` dan assert ukuran.
  - Fungsi `show_pairs` untuk menampilkan beberapa pasangan berdampingan, dan cek angka `color_spread` (rata-rata atas 200 gambar) untuk memastikan sketsa lebih "abu-abu" daripada foto.
  - Pertanyaan lanjutan: kode tampilan ditaruh di cell baru di bawah cell subset, dan AI mengingatkan tiga syarat (`train_paths` harus ada, `load_pair` harus ada di `src/data.py`, dan modul lama tersimpan di memori).
- **Hasil di Kaggle:** Panel menampilkan `sketch (256, 256)` di kiri dan `photo (256, 256)` di kanan. Sketsa dan foto tidak tertukar, dan bentuk sepatunya cocok. Setelah ditanya "apakah hasilnya sudah berhasil?", AI menilai hasilnya benar untuk bagian yang terlihat dan menyarankan mengecek pasangan ketiga.

### 2. Tahap Penanganan Eror / Debugging

> Catatan: eror 2a dan 2c ditangani lewat sesi Claude Code terpisah, tidak dikirim ke chat pada tautan di atas. Eror 2b dan 2d ditanyakan ke chat ini.

#### 2a. Path dataset tidak ditemukan (komponen 1a)

- **Pesan Eror dari Notebook:**
  ```
  datasets
  ls: cannot access '/kaggle/input/edges2shoes-dataset': No such file or directory
  ```
- **Prompt Perbaikan ke LLM:**
  > Tidak dikirim ke chat ini. Eror ditangani lewat sesi Claude Code terpisah.
- **Analisis & Solusi Kelompok:** Path yang kami tebak (`/kaggle/input/edges2shoes-dataset`) salah. Struktur input Kaggle ternyata `/kaggle/input/datasets/<pemilik>/<nama-dataset>`. Solusinya mengecek struktur lewat `!find /kaggle/input -maxdepth 4`, sehingga `DATA_ROOT` yang benar adalah `/kaggle/input/datasets/balraj98/edges2shoes-dataset`. Pelajaran: jangan menebak path, dan itu alasan lokasi dataset dijadikan parameter fungsi.

#### 2b. Notebook tidak bisa mengakses GitHub (komponen 1a)

- **Pesan Eror dari Notebook:**
  ```
  Cloning into 'sketch2real-pix2pix'...
  fatal: unable to access 'https://github.com/AdrianAnis/sketch2real-pix2pix.git/': Could not resolve host: github.com
  [Errno 2] No such file or directory: 'sketch2real-pix2pix'
  ```
- **Prompt Perbaikan ke LLM:**
  > Waktu aku mencoba mengambil kode repo kelompok ke Kaggle Notebook (git clone, lalu pindah ke foldernya), muncul eror ini:
  >
  > [pesan eror di atas]
  >
  > Jelaskan penyebabnya, lalu beri perbaikan minimal. Aku juga ingin tahu kenapa muncul dua eror sekaligus.
- **Analisis & Solusi Kelompok:**
  - **Penyebab:** `Could not resolve host: github.com` berarti notebook tidak bisa menjangkau situs luar karena akses Internet di notebook Kaggle sedang OFF (default Kaggle). URL repo tidak bermasalah.
  - **Dua eror sekaligus:** keduanya satu rantai sebab-akibat. `git clone` gagal sehingga folder `sketch2real-pix2pix` tidak pernah dibuat, lalu `%cd` mencoba masuk ke folder yang tidak ada dan memunculkan `[Errno 2]`. Posisi kerja tetap di `/kaggle/working`.
  - **Solusi:** menyalakan toggle Internet di Session options. Toggle tidak bisa dinyalakan sebelum akun verifikasi nomor HP, jadi kami verifikasi dulu, lalu menjalankan ulang cell yang sama. Hasilnya `/kaggle/working/sketch2real-pix2pix`.
  - **Catatan dari AI:** kalau repo private, setelah internet aktif eror berubah jadi masalah autentikasi (perlu Personal Access Token; jangan taruh token di cell notebook, simpan lewat Kaggle Secrets). Alternatif tanpa internet: upload zip repo sebagai Kaggle Dataset lalu salin ke `/kaggle/working`, tapi repot karena tiap perubahan kode harus di-upload ulang.
  - **Pelajaran:** Internet dan GPU di Kaggle butuh akun terverifikasi, dan eror kedua sering hanya dampak dari eror pertama, jadi baca eror dari yang paling atas.

#### 2c. Fungsi tidak dikenali saat verifikasi (komponen 1a)

- **Pesan Eror dari Notebook:**
  ```
  NameError: name 'select_edges2shoes_subset' is not defined
  ```
- **Prompt Perbaikan ke LLM:**
  > Tidak dikirim ke chat ini. Eror ditangani lewat sesi Claude Code terpisah.
- **Analisis & Solusi Kelompok:** Baris `from src.data import select_edges2shoes_subset` tidak ikut ke dalam cell, sehingga Python tidak tahu asal fungsinya. Solusinya menambahkan baris import di awal cell, ditambah `%cd` ke folder repo agar modul `src` bisa ditemukan. Pelajaran: fungsi dari file lain harus di-import dulu sebelum dipakai.

#### 2d. Fungsi load_pair tidak bisa di-import (komponen 1b)

- **Pesan Eror dari Notebook:**
  ```
  ImportError: cannot import name 'load_pair' from 'src.data' (/kaggle/working/sketch2real-pix2pix/src/data.py)
  ```
- **Prompt Perbaikan ke LLM:**

  > kode yang kamu berikan menghasilkan error berikut. jelaskan penyebabnya dan berikan perbaikan minimal.
  >
  > [pesan eror di atas]

  Pertanyaan lanjutan setelah AI meminta menjalankan `!grep -n "def load_pair" ...`:

  > tidak terdapat output pada cell tersebut

  > aku sudah push kode yang berisi function load_pair, apakah aku harus pull ke kaggle nya?

- **Analisis & Solusi Kelompok:**
  - **Penyebab:** Python menemukan `src/data.py` (path benar), tetapi di dalamnya tidak ada `load_pair`. AI menyebut tiga kemungkinan (fungsi belum ada di file, modul lama masih di memori, atau indentasi salah) dan meminta kami membedakannya lewat `grep`.
  - **Hasil `grep` kosong:** artinya `load_pair` belum ada di salinan repo di Kaggle. Penyebabnya, folder di Kaggle adalah hasil `git clone` sebelum fungsi itu ditambahkan, dan tidak otomatis mengikuti perubahan di GitHub.
  - **Solusi:** AI menjelaskan bahwa push hanya mengirim kode ke GitHub, jadi di notebook harus menjalankan `%cd` ke folder repo lalu `!git pull`, kemudian memuat ulang modul dengan `importlib.reload(src.data)` karena versi lama masih tersimpan di memori. AI juga menawarkan jalur cepat `%%writefile -a`, tetapi kami memilih push dan pull agar kode di Kaggle sama dengan repo.
  - **Hasil:** `git pull` memperbarui `src/data.py`, `grep` menunjukkan `def load_pair` di baris 60, dan setelah reload cell berjalan normal.
  - **Pelajaran:** salinan repo di Kaggle tidak otomatis ikut berubah. Setiap ada kode baru: push dari laptop, `git pull` di notebook, lalu reload modul atau restart session.

## Catatan
