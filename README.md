# Smart Insect Identifier

Aplikasi berbasis web untuk mengidentifikasi spesies serangga secara instan menggunakan AI (Kecerdasan Buatan), serta memberikan informasi habitat, perilaku, dan ekologinya secara otomatis (Insights by Gemini).

## Struktur Proyek

- **`/frontend`**: Aplikasi Next.js (React) yang menyajikan antarmuka pengguna responsif bergaya glassmorphism dan tata letak grid modern.
- **`/backend`**: Aplikasi FastAPI (Python) yang melayani model Machine Learning PyTorch (`RegNetY-32GF`) dan berintegrasi dengan Google Gemini API.

## Cara Menjalankan

### Backend
1. Masuk ke direktori `backend/`
2. Buat _virtual environment_: `python3 -m venv .venv` dan aktifkan: `source .venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Pastikan file model `model.pth` ada di dalam folder `backend/artifacts/` (tidak disertakan dalam Git karena ukuran 1.4GB)
5. Jalankan server: `python main.py` (berjalan di port 8000)

### Frontend
1. Masuk ke direktori `frontend/`
2. Install dependencies: `npm install`
3. Jalankan server: `npm run dev` (berjalan di port 3000)

## Fitur Utama
- **Deteksi Serangga Real-Time**: Upload atau gunakan kamera perangkat untuk deteksi.
- **Dukungan Multi-Bahasa**: Bahasa Indonesia dan Inggris.
- **Riwayat (*History*)**: Menampilkan riwayat hasil deteksi serangga sebelumnya.
- **UI Layar Penuh Modern**: Transisi mulus (*smooth*) dengan latar belakang yang berkelas.

## Catatan
File besar (seperti model `.pth` dan *cache*) tidak dimasukkan ke dalam Git untuk mempermudah dan meringankan beban *upload/push* ke GitHub.
