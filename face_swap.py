"""
Face swap module using InsightFace.
Swaps faces in video frames with a face from a source image.
Preserves original video audio in output.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import cv2
import numpy as np
from typing import Optional, Callable, Tuple
import threading

# Logger - bisa di-set dari luar untuk mengumpulkan log
_log_handler: Optional[Callable[[str], None]] = None


def set_log_handler(handler: Optional[Callable[[str], None]]):
    """Set handler untuk mengumpulkan log (dipanggil dari app)."""
    global _log_handler
    _log_handler = handler


def _log(msg: str, level: str = "INFO"):
    """Tulis log ke handler dan logging."""
    logging.getLogger("face_swap").log(
        getattr(logging, level.upper(), logging.INFO), msg
    )
    if _log_handler:
        try:
            _log_handler(f"[{level}] {msg}")
        except Exception:
            pass


# Global state for models (loaded per device)
_face_analyser = None
_face_swapper = None
_current_providers = None
_models_lock = threading.Lock()


def is_model_loaded() -> bool:
    """Check if models are loaded."""
    return _face_analyser is not None and _face_swapper is not None


def get_providers(use_gpu: bool) -> tuple:
    """
    Get ONNX runtime providers based on device selection.
    Returns (providers_list, error_message or None).
    """
    import onnxruntime as ort
    available = ort.get_available_providers()
    
    if use_gpu:
        if "CUDAExecutionProvider" in available:
            return (["CUDAExecutionProvider", "CPUExecutionProvider"], None)
        # GPU dipilih tapi onnxruntime tidak punya CUDA - perlu onnxruntime-gpu
        return (
            ["CPUExecutionProvider"],
            "GPU dipilih tapi ONNX Runtime tidak mendukung CUDA. "
            "Install onnxruntime-gpu: pip uninstall onnxruntime -y && pip install onnxruntime-gpu"
        )
    return (["CPUExecutionProvider"], None)


def load_models(use_gpu: bool, progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
    """
    Load InsightFace models (FaceAnalysis + FaceSwapper).
    Returns (success, message).
    """
    global _face_analyser, _face_swapper, _current_providers
    
    with _models_lock:
        providers, provider_error = get_providers(use_gpu)
        if provider_error:
            return False, provider_error
        
        # Skip if already loaded with same providers
        if _face_analyser is not None and _current_providers == tuple(providers):
            return True, "Model sudah dimuat"
        
        try:
            if progress_callback:
                progress_callback("Memuat model Face Analysis...")
            
            import insightface
            from insightface.app import FaceAnalysis
            
            # Create models directory
            model_root = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(model_root, exist_ok=True)
            
            # Download buffalo_l if needed (FaceAnalysis)
            _face_analyser = FaceAnalysis(
                name="buffalo_l",
                root=model_root,
                providers=providers
            )
            _face_analyser.prepare(ctx_id=0, det_size=(320, 320))
            
            if progress_callback:
                progress_callback("Memuat model Face Swapper...")
            
            # Load inswapper model
            swapper_path = os.path.join(model_root, "inswapper_128.onnx")
            if not os.path.exists(swapper_path):
                # Try to download from HuggingFace
                try:
                    from huggingface_hub import hf_hub_download
                    if progress_callback:
                        progress_callback("Mengunduh model inswapper (554MB)...")
                    downloaded = hf_hub_download(
                        repo_id="ezioruan/inswapper_128.onnx",
                        filename="inswapper_128.onnx",
                        local_dir=model_root,
                        local_dir_use_symlinks=False
                    )
                    swapper_path = downloaded if os.path.isfile(downloaded) else swapper_path
                except Exception as e:
                    # Fallback: try direct URL download
                    try:
                        if progress_callback:
                            progress_callback("Mengunduh model inswapper (554MB)...")
                        import urllib.request
                        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
                        urllib.request.urlretrieve(url, swapper_path)
                    except Exception as e2:
                        return False, (
                            f"Model inswapper_128.onnx tidak ditemukan. "
                            f"Download manual dari https://huggingface.co/ezioruan/inswapper_128.onnx "
                            f"dan simpan di {model_root}. Error: {str(e2)}"
                        )
            
            _face_swapper = insightface.model_zoo.get_model(
                swapper_path,
                download=False,
                download_zip=False
            )
            
            _current_providers = tuple(providers)
            device = "GPU" if "CUDAExecutionProvider" in providers else "CPU"
            return True, f"Model berhasil dimuat ({device})"
            
        except Exception as e:
            _face_analyser = None
            _face_swapper = None
            _current_providers = None
            return False, str(e)


def _has_audio_stream(video_path: str) -> bool:
    """Cek apakah video memiliki stream audio."""
    _log(f"ffprobe: cek audio di {os.path.basename(video_path)}")
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                video_path
            ],
            capture_output=True, text=True, timeout=10
        )
        _log(f"ffprobe returncode={result.returncode}, stdout='{result.stdout or ''}', stderr='{result.stderr or ''}'")
        has_audio = result.returncode == 0 and result.stdout and "audio" in result.stdout
        _log(f"Audio stream: {'ada' if has_audio else 'tidak ada'}")
        return has_audio
    except FileNotFoundError:
        _log("ffprobe tidak ditemukan - asumsikan tidak ada audio", "WARNING")
        return False
    except Exception as e:
        _log(f"Error cek audio: {e}", "WARNING")
        return False


def _merge_audio(video_no_audio: str, original_video: str, output_path: str) -> Tuple[bool, str]:
    """
    Gabungkan video (tanpa audio) dengan audio dari video asli.
    Returns (success, error_message).
    """
    _log("=== Mulai penggabungan audio ===")
    _log(f"Video (tanpa audio): {video_no_audio}, exists={os.path.exists(video_no_audio)}, size={os.path.getsize(video_no_audio) if os.path.exists(video_no_audio) else 0} bytes")
    _log(f"Original (dengan audio): {original_video}, exists={os.path.exists(original_video)}, size={os.path.getsize(original_video) if os.path.exists(original_video) else 0} bytes")
    _log(f"Output: {output_path}")

    # Cek ffmpeg tersedia
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        _log("ffmpeg tersedia")
    except FileNotFoundError:
        _log("ffmpeg tidak ditemukan di PATH", "ERROR")
        return False, "ffmpeg tidak ditemukan. Install: apt install ffmpeg"
    except Exception as e:
        _log(f"Cek ffmpeg: {e}", "WARNING")

    if not _has_audio_stream(original_video):
        _log("Video sumber tidak memiliki audio - skip merge", "WARNING")
        return False, "Video sumber tidak memiliki audio"

    try:
        for use_copy in [True, False]:
            mode = "copy" if use_copy else "re-encode"
            _log(f"Mencoba merge audio (mode={mode})...")

            if use_copy:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_no_audio,
                    "-i", original_video,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-shortest",
                    output_path
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_no_audio,
                    "-i", original_video,
                    "-map", "0:v",
                    "-map", "1:a",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-shortest",
                    output_path
                ]

            cmd_str = " ".join(f'"{x}"' if " " in x else x for x in cmd)
            _log(f"ffmpeg command: {cmd_str}")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            stderr_full = result.stderr or ""
            stdout_full = result.stdout or ""

            if result.returncode == 0:
                _log("Audio berhasil digabungkan")
                return True, ""

            _log(f"ffmpeg gagal (returncode={result.returncode})", "WARNING")
            _log(f"ffmpeg stderr: {stderr_full}", "WARNING")
            if stdout_full:
                _log(f"ffmpeg stdout: {stdout_full}", "WARNING")
            if use_copy:
                _log("Fallback ke re-encode...")
                continue
            return False, f"ffmpeg error: {stderr_full[-1000:] if len(stderr_full) > 1000 else stderr_full}"

    except FileNotFoundError:
        _log("ffmpeg tidak ditemukan", "ERROR")
        return False, "ffmpeg tidak ditemukan. Install: apt install ffmpeg"
    except subprocess.TimeoutExpired:
        _log("ffmpeg timeout (600s)", "ERROR")
        return False, "ffmpeg timeout"
    except Exception as e:
        _log(f"Error merge audio: {e}", "ERROR")
        import traceback
        _log(traceback.format_exc(), "ERROR")
        return False, str(e)


def get_face(face_analyser, frame: np.ndarray):
    """Get the first/main face from frame."""
    faces = face_analyser.get(frame)
    if not faces:
        return None
    return min(faces, key=lambda x: x.bbox[0])


def swap_face_in_frame(
    frame: np.ndarray,
    source_face,
    face_analyser,
    face_swapper,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> np.ndarray:
    """
    Swap faces in a single frame.
    Replaces all detected faces with source_face.
    """
    target_faces = face_analyser.get(frame)
    if not target_faces:
        return frame
    
    result = frame.copy()
    for target_face in target_faces:
        result = face_swapper.get(result, target_face, source_face, paste_back=True)
    
    return result


def process_video(
    video_path: str,
    source_image_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Tuple[bool, str]:
    """
    Process video: replace all faces with the face from source image.
    Returns (success, message).
    """
    global _face_analyser, _face_swapper
    frame_idx = 0

    if _face_analyser is None or _face_swapper is None:
        _log("Model belum dimuat")
        return False, "Model belum dimuat. Pilih device dan load model terlebih dahulu."

    try:
        _log(f"Memulai processing: video={video_path}, face={source_image_path}")

        # Read source image and get face
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            _log("Gagal membaca gambar wajah sumber", "ERROR")
            return False, "Gagal membaca gambar wajah sumber"

        source_face = get_face(_face_analyser, source_img)
        if source_face is None:
            _log("Tidak ada wajah terdeteksi pada gambar sumber", "ERROR")
            return False, "Tidak ada wajah terdeteksi pada gambar sumber"
        _log("Wajah sumber berhasil dideteksi")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _log("Gagal membuka video", "ERROR")
            return False, "Gagal membuka video"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            total_frames = 1  # Hindari division by zero
        _log(f"Video: {width}x{height}, {fps} fps, ~{total_frames} frame")

        # Tulis ke file sementara (video saja, tanpa audio)
        fd, temp_video_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        _log(f"Temp file: {temp_video_path}")

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                _log("Gagal membuat VideoWriter", "ERROR")
                return False, "Gagal membuat output video"

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if progress_callback:
                    progress_callback(frame_idx, total_frames, f"Memproses frame {frame_idx + 1}/{total_frames}")

                result_frame = swap_face_in_frame(
                    frame, source_face, _face_analyser, _face_swapper, progress_callback
                )
                out.write(result_frame)
                frame_idx += 1

                if frame_idx % 30 == 0:
                    _log(f"Frame {frame_idx}/{total_frames} selesai")

            cap.release()
            out.release()
            _log(f"Face swap selesai: {frame_idx} frame diproses")

            # Gabungkan dengan audio dari video asli
            _log("Memeriksa audio video sumber...")
            if _has_audio_stream(video_path):
                if progress_callback:
                    progress_callback(frame_idx, total_frames, "Menggabungkan audio...")
                _log("Memanggil _merge_audio...")
                merge_ok, merge_err = _merge_audio(temp_video_path, video_path, output_path)
                if merge_ok:
                    _log("Output dengan audio tersimpan")
                    merge_error_msg = None
                else:
                    _log(f"Merge gagal: {merge_err}", "WARNING")
                    _log("Fallback: menyimpan video tanpa audio", "WARNING")
                    shutil.copy(temp_video_path, output_path)
                    _log("Video tanpa audio berhasil disimpan")
                    merge_error_msg = merge_err
            else:
                _log("Video sumber tidak punya audio, menyimpan tanpa suara")
                shutil.copy(temp_video_path, output_path)
                merge_error_msg = None
        finally:
            if os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError as e:
                    _log(f"Gagal hapus temp: {e}", "WARNING")

        if progress_callback:
            progress_callback(total_frames, total_frames, "Selesai!")

        msg = f"Video berhasil diproses ({frame_idx} frame)"
        if merge_error_msg:
            msg += f". Tanpa audio: {merge_error_msg}"
        _log(msg)
        return True, msg

    except Exception as e:
        _log(f"Error: {e}", "ERROR")
        import traceback
        _log(traceback.format_exc(), "ERROR")
        return False, str(e)
