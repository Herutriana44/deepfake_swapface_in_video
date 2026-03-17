#!/usr/bin/env python3
"""
Script untuk menjalankan Flask Face Swap di Google Colab / Kaggle.
Jalankan setelah: git clone <repo> && cd <repo>

Usage di Colab:
  !git clone https://github.com/USERNAME/deepfake_swapface_in_video.git
  %cd deepfake_swapface_in_video
  !pip install -r requirements.txt
  !pip install pyngrok  # untuk akses dari browser
  !python run_colab.py
"""

import os
import sys
import threading
import time

# Pastikan kita di direktori proyek
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_with_ngrok():
    """Jalankan Flask + ngrok untuk akses publik (Colab)."""
    try:
        from pyngrok import ngrok
    except ImportError:
        print("Install pyngrok: pip install pyngrok")
        return False

    def run_flask():
        import app
        app.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

    thread = threading.Thread(target=run_flask, daemon=True)
    thread.start()
    time.sleep(5)

    public_url = ngrok.connect(5000)
    print("=" * 60)
    print("Buka URL berikut di browser Anda:")
    print(public_url)
    print("=" * 60)
    print("Tekan Ctrl+C untuk menghentikan.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass
    return True


def run_direct():
    """Jalankan Flask langsung (tanpa ngrok)."""
    import app
    app.app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    use_ngrok = "--ngrok" in sys.argv or "-n" in sys.argv
    if use_ngrok:
        run_with_ngrok()
    else:
        run_direct()
