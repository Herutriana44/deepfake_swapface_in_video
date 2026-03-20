# Deepfake Face Swap - Video

Aplikasi web Flask untuk mengganti wajah di video dengan wajah dari gambar menggunakan **InsightFace**.

## Fitur

- Audio video asli tetap tersimpan di output
- Pilih CPU atau GPU untuk processing
- Load model sebelum processing
- Upload 1 video + 1 gambar wajah
- Progress real-time di web
- Download hasil video

## Instalasi Lokal

```bash
# Clone repository
git clone https://github.com/USERNAME/deepfake_swapface_in_video.git
cd deepfake_swapface_in_video

# Buat virtual environment (opsional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Untuk GPU (CUDA): ganti onnxruntime dengan onnxruntime-gpu
pip uninstall onnxruntime
pip install onnxruntime-gpu

# Jalankan
python app.py
```

Buka http://localhost:5000 di browser.

## Menjalankan di Google Colab / Kaggle

### Menggunakan Notebook

1. Buka `run_colab_kaggle.ipynb` di Colab atau Kaggle
2. Edit `REPO_URL` dengan URL repository Anda
3. Jalankan cell secara berurutan
4. Untuk Colab: gunakan cell dengan ngrok untuk mendapat URL publik

### Menggunakan Script

```python
# Cell 1: Clone & Install
!git clone https://github.com/USERNAME/deepfake_swapface_in_video.git
%cd deepfake_swapface_in_video
!pip install -r requirements.txt
!pip install pyngrok

# Cell 2: Run (dengan ngrok untuk akses dari browser)
!python run_colab.py --ngrok
```

## Dependencies

- **ffmpeg**: Diperlukan untuk menyimpan audio. Biasanya sudah terpasang di Colab/Kaggle. Lokal: `sudo apt install ffmpeg` (Linux) atau download dari ffmpeg.org.

## Model

- **Face Analysis**: buffalo_l (didownload otomatis oleh InsightFace)
- **Face Swapper**: inswapper_128.onnx (didownload otomatis dari HuggingFace saat pertama load)

Jika download otomatis gagal, download manual dari:
- https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx

Simpan di folder `models/` dalam direktori proyek.

## Format File

- **Video**: MP4, AVI, MOV, MKV, WebM
- **Gambar**: JPG, PNG, BMP, WebP

## Lisensi

Model InsightFace/inswapper untuk penggunaan non-komersial. Gunakan secara bertanggung jawab.
