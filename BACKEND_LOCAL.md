# Menjalankan backend lokal

Jika ingin backend berjalan di terminal yang sedang dipakai dan dihentikan
dengan Ctrl+C, jalankan:

```powershell
python -m src.backend.app
```

Perintah ini tidak memakai pemuat ulang otomatis. Setelah mengubah kode,
hentikan lalu jalankan kembali. Jika backend terpisah masih aktif, hentikan
dengan `.\start_backend.ps1 -Action stop` sebelum memakai perintah ini.

Untuk menjalankan backend terpisah dari terminal, gunakan peluncur berikut.

Dari PowerShell di folder proyek:

```powershell
.\start_backend.ps1
.\start_backend.ps1 -Action status
```

Peluncur menjalankan backend di proses terpisah dengan log di
`outputs/backend-runtime/`. Backend siap digunakan setelah log menampilkan
`Application startup complete`. Alamat API adalah `http://127.0.0.1:8000`.

Setelah mengubah kode backend, muat versi terbaru dengan:

```powershell
.\start_backend.ps1 -Action restart
```

Untuk berhenti:

```powershell
.\start_backend.ps1 -Action stop
```

Peluncur ini tidak menggunakan pemuat ulang otomatis. Penutupan terminal VS Code
tidak menghentikan backend; gunakan perintah `stop`. Saat backend dihentikan,
perhitungan aktif menerima permintaan pembatalan sebelum server ditutup.
Perintah `stop` menunggu maksimal 10 detik untuk penutupan normal. Jika proses
tetap tersangkut, peluncur memeriksa ulang identitas dan waktu mulai proses
sebelum menghentikan backend proyek ini. Jadwal rebalance yang sudah selesai
tetap tersimpan. Jangan memakai Ctrl+C di terminal lain untuk menghentikan
backend yang diluncurkan secara terpisah.
Jika port 8000 masih dipakai server yang dijalankan dengan cara lain, hentikan
server tersebut terlebih dahulu.

Panel progres menyediakan tombol **Hentikan analisis**. Tombol ini menutup
permintaan analisis aktif, termasuk penantian layanan narasi. Setelah pembatalan,
periksa riwayat sebelum menjalankan kembali karena jadwal yang telah selesai
atau hasil yang sudah tersimpan tetap tersedia.

## Narasi portofolio melalui OpenAI

Integrasi menggunakan OpenAI Responses API dengan model `gpt-6-luna`.
Pasang dependensi proyek melalui `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
Isi `OPENAI_API_KEY` pada `.env` lokal. Contoh pengaturan tersedia pada `.env.example`;
jangan menyalin ulang seluruh berkas contoh jika `.env` sudah berisi konfigurasi lain.
Pengaturan Groq lama tidak lagi digunakan oleh pembangkitan narasi.

Model bawaan adalah `OPENAI_MODEL=gpt-6-luna`, dengan
`OPENAI_REASONING_EFFORT=medium` dan `OPENAI_MAX_OUTPUT_TOKENS=4096`.
Batas token ini mencakup penalaran dan teks jawaban. Tidak ada pergantian model
otomatis secara bawaan. Respons yang belum selesai tidak diterima sebagai narasi.
`OPENAI_TIMEOUT_MS` menjadi batas waktu keseluruhan tahap narasi, termasuk
percobaan ulang dan koreksi draf. Penantian jaringan dan jeda percobaan ulang
dapat dibatalkan tanpa menunggu timeout layanan habis.
Narasi tetap diperiksa berdasarkan fakta numerik dan sumber berita; draf yang
gagal pemeriksaan dapat diperbaiki satu kali sebelum memakai narasi cadangan.

Jika kunci belum diisi atau layanan gagal, konfigurasi bawaan memakai narasi
cadangan. `OPENAI_ENABLED=false` menonaktifkan pemanggilan API.
Log mencatat jumlah token input, output, dan penalaran untuk memantau penggunaan.
Setelah mengisi atau mengubah konfigurasi, muat ulang backend dengan
`.\start_backend.ps1 -Action restart`.

Progres generasi dikirim paling sering sekali per detik. Angka kecepatan
perhitungan menggunakan waktu backend, sedangkan pembaruan panel dapat melompati
beberapa generasi agar tampilan tetap ringan. Keduanya tidak mengurangi jumlah
generasi yang dihitung.

Pada Windows yang mendukungnya, tugas evolusi GA memakai HighQoS selama
perhitungan. Ini memberi tahu Windows bahwa tugas tersebut membutuhkan
respons cepat walaupun jendela editor tertutup oleh browser. Kebijakan thread
sebelumnya dipulihkan setelah GA selesai atau dibatalkan. Prioritas proses dan
pengaturan daya Windows tidak diubah.
