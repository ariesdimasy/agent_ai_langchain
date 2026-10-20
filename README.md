# Rangkumin

Asisten buku berbasis AI untuk membuat rangkuman dan menjawab pertanyaan dari
PDF. Untuk tanya jawab, model dapat memilih tool pencarian dokumen; jawaban
menyertakan nomor halaman sumber. Backend menggunakan FastAPI dan LangChain
dengan API DeepSeek, sedangkan frontend berupa React SPA yang dibangun dengan
Vite.

Riwayat tanya jawab disimpan sementara di state browser dan dikirim bersama
permintaan berikutnya (maksimal 12 pesan terakhir). PDF dikirim ke backend
untuk setiap pertanyaan dan tidak disimpan oleh aplikasi.

## Prasyarat

- Python 3.10 atau lebih baru
- Node.js 18 atau lebih baru
- API key DeepSeek
- Untuk PDF hasil pindai: Tesseract OCR dengan data bahasa Indonesia dan Inggris

## Menjalankan backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Isi `DEEPSEEK_API_KEY` di `backend/.env`, lalu jalankan:

```powershell
uvicorn main:app --reload
```

API tersedia di `http://localhost:8000`; dokumentasi interaktif tersedia di
`http://localhost:8000/docs`.

Endpoint utama:

- `POST /api/summarize` — membuat rangkuman buku.
- `POST /api/summarize/stream` — membuat rangkuman sambil mengirim progres
  pembacaan halaman dan bagian yang dirangkum melalui Server-Sent Events.
- `POST /api/chat` — menerima `file` (PDF), `question`, dan `history` sebagai
  form data; mengembalikan `answer` dan daftar nomor halaman `sources`.

## Menjalankan frontend

Buka terminal lain:

```powershell
cd frontend
npm install
npm run dev
```

Buka alamat yang ditampilkan Vite (biasanya `http://localhost:5173`). Jika API
berjalan di alamat lain, buat `frontend/.env` dengan `VITE_API_URL`, misalnya:

```dotenv
VITE_API_URL=http://localhost:8000
```

## Menjalankan backend dan frontend sekaligus (Windows)

Setelah virtual environment, dependencies, dan `backend/.env` disiapkan,
klik dua kali [`Jalankan Rangkumin.bat`](./Jalankan%20Rangkumin.bat) dari folder
proyek. Launcher membuka backend dan frontend di dua jendela terminal terpisah,
lalu membuka aplikasi di `http://localhost:5173`. Tutup kedua jendela terminal
untuk menghentikan server.

## Pengujian

Dari folder `backend`, jalankan:

```powershell
pip install pytest
pytest
```

PDF harus memiliki lapisan teks; PDF hasil pemindaian yang hanya berisi gambar
akan diproses dengan OCR lokal melalui Tesseract. Di Windows, instal Tesseract
dari [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) dan pastikan
bahasa Indonesia (`ind`) serta Inggris (`eng`) ikut terpasang. Setelah
mengubah dependency, jalankan ulang `pip install -r requirements.txt`.

Konfigurasi OCR opsional di `backend/.env`:

```dotenv
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANG=ind+eng
OCR_TESSDATA_DIR=C:\Users\<username>\AppData\Local\Tesseract-OCR\tessdata
```

Atur `TESSERACT_CMD` ke lokasi `tesseract.exe` jika executable tidak ditemukan
melalui `PATH`. `OCR_TESSDATA_DIR` hanya diperlukan jika data bahasa Tesseract
berada di luar lokasi instalasi standar. Ukuran unggahan default dibatasi 20 MB
dan dapat disesuaikan dengan `MAX_FILE_SIZE` (dalam byte).
