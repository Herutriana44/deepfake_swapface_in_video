"""
Flask Web App untuk Deepfake Face Swap
Mengganti wajah di video dengan wajah dari gambar menggunakan InsightFace
"""

import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from face_swap import load_models, process_video, is_model_loaded, set_log_handler

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.path.dirname(__file__), "outputs")

ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp"}

# Progress state
progress_state = {}
progress_lock = threading.Lock()


def allowed_file(filename, allowed):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/load-model", methods=["POST"])
def api_load_model():
    """Load model dengan device (CPU/GPU) yang dipilih."""
    data = request.get_json() or {}
    use_gpu = data.get("use_gpu", False)

    def progress(msg):
        pass  # Bisa ditambahkan callback untuk loading

    success, message = load_models(use_gpu=use_gpu, progress_callback=progress)
    return jsonify({"success": success, "message": message})


@app.route("/api/check-model", methods=["GET"])
def api_check_model():
    """Cek apakah model sudah dimuat."""
    return jsonify({"loaded": is_model_loaded()})


def _detect_gpu_system() -> dict:
    """
    Deteksi GPU dari berbagai sumber (Kaggle/Colab bisa punya GPU
    meski onnxruntime CPU terpasang). Return dict dengan info deteksi.
    """
    result = {
        "has_gpu": False,
        "has_onnx_cuda": False,
        "providers": [],
        "gpu_name": None,
        "detection_method": None,
    }
    
    # 1. Cek via PyTorch (paling reliable di Kaggle/Colab)
    try:
        import torch
        if torch.cuda.is_available():
            result["has_gpu"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            result["detection_method"] = "pytorch"
    except ImportError:
        pass
    
    # 2. Cek via nvidia-smi (jika PyTorch tidak terpasang)
    if not result["has_gpu"]:
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if out.returncode == 0 and out.stdout.strip():
                result["has_gpu"] = True
                result["gpu_name"] = out.stdout.strip().split("\n")[0]
                result["detection_method"] = "nvidia-smi"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
    
    # 3. Cek env var (Kaggle/Colab set ini saat GPU aktif)
    if not result["has_gpu"]:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible and cuda_visible != "-1" and cuda_visible != "":
            result["has_gpu"] = True
            result["detection_method"] = "env"
    
    # 4. Cek ONNX Runtime providers
    try:
        import onnxruntime as ort
        result["providers"] = ort.get_available_providers()
        result["has_onnx_cuda"] = "CUDAExecutionProvider" in result["providers"]
        # Jika sistem punya GPU tapi onnx belum, tetap tampilkan has_gpu=True
        # (user perlu install onnxruntime-gpu)
    except Exception:
        pass
    
    return result


@app.route("/api/check-gpu", methods=["GET"])
def api_check_gpu():
    """Cek ketersediaan GPU (multi-source: PyTorch, nvidia-smi, env, onnxruntime)."""
    try:
        info = _detect_gpu_system()
        return jsonify({
            "has_gpu": info["has_gpu"],
            "has_onnx_cuda": info["has_onnx_cuda"],
            "providers": info["providers"],
            "gpu_name": info["gpu_name"],
            "detection_method": info["detection_method"],
        })
    except Exception as e:
        return jsonify({"has_gpu": False, "has_onnx_cuda": False, "error": str(e)})


@app.route("/api/process", methods=["POST"])
def api_process():
    """Upload file dan mulai processing."""
    if "video" not in request.files or "face_image" not in request.files:
        return jsonify({"success": False, "message": "Video dan gambar wajah wajib diupload"}), 400

    video_file = request.files["video"]
    face_file = request.files["face_image"]

    if video_file.filename == "" or face_file.filename == "":
        return jsonify({"success": False, "message": "Pilih file video dan gambar wajah"}), 400

    if not allowed_file(video_file.filename, ALLOWED_VIDEO):
        return jsonify({"success": False, "message": "Format video tidak didukung"}), 400

    if not allowed_file(face_file.filename, ALLOWED_IMAGE):
        return jsonify({"success": False, "message": "Format gambar tidak didukung"}), 400

    # Generate job ID
    job_id = str(uuid.uuid4())
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    # Save uploaded files
    video_ext = video_file.filename.rsplit(".", 1)[1].lower()
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_video.{video_ext}")
    face_ext = face_file.filename.rsplit(".", 1)[1].lower()
    face_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}_face.{face_ext}")
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_result.mp4")

    video_file.save(video_path)
    face_file.save(face_path)

    # Initialize progress + logs
    with progress_lock:
        progress_state[job_id] = {
            "current": 0, "total": 1, "message": "Memulai...", "done": False,
            "logs": []
        }

    def log_cb(log_line: str):
        with progress_lock:
            if job_id in progress_state:
                progress_state[job_id].setdefault("logs", []).append(log_line)

    def progress_cb(current, total, msg):
        with progress_lock:
            if job_id in progress_state:
                progress_state[job_id]["current"] = current
                progress_state[job_id]["total"] = total
                progress_state[job_id]["message"] = msg
                progress_state[job_id]["done"] = current >= total and total > 0

    def run_process():
        set_log_handler(log_cb)
        try:
            success, message = process_video(
                video_path, face_path, output_path, progress_callback=progress_cb
            )
        finally:
            set_log_handler(None)
        with progress_lock:
            if job_id in progress_state:
                progress_state[job_id]["success"] = success
                progress_state[job_id]["message"] = message
                progress_state[job_id]["done"] = True

        # Cleanup uploads
        try:
            os.remove(video_path)
            os.remove(face_path)
        except OSError:
            pass

    thread = threading.Thread(target=run_process)
    thread.start()

    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/progress/<job_id>")
def api_progress(job_id):
    """Ambil progress processing + logs."""
    with progress_lock:
        if job_id not in progress_state:
            return jsonify({"error": "Job tidak ditemukan"}), 404
        state = progress_state[job_id].copy()

    percent = 0
    if state["total"] > 0:
        percent = min(100, int(100 * state["current"] / state["total"]))

    return jsonify({
        "current": state["current"],
        "total": state["total"],
        "percent": percent,
        "message": state["message"],
        "done": state.get("done", False),
        "success": state.get("success", None),
        "logs": state.get("logs", []),
    })


@app.route("/api/download/<job_id>")
def api_download(job_id):
    """Download hasil video."""
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_result.mp4")
    if not os.path.exists(output_path):
        return jsonify({"error": "File tidak ditemukan"}), 404
    return send_file(
        output_path,
        as_attachment=True,
        download_name="face_swap_result.mp4",
        mimetype="video/mp4"
    )


@app.route("/api/cleanup/<job_id>", methods=["POST"])
def api_cleanup(job_id):
    """Hapus file output setelah download."""
    output_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_result.mp4")
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
        with progress_lock:
            if job_id in progress_state:
                del progress_state[job_id]
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
