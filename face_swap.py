"""
Face swap module using InsightFace.
Swaps faces in video frames with a face from a source image.
"""

import os
import cv2
import numpy as np
from typing import Optional, Callable, Tuple
import threading

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
    
    if _face_analyser is None or _face_swapper is None:
        return False, "Model belum dimuat. Pilih device dan load model terlebih dahulu."
    
    try:
        # Read source image and get face
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            return False, "Gagal membaca gambar wajah sumber"
        
        source_face = get_face(_face_analyser, source_img)
        if source_face is None:
            return False, "Tidak ada wajah terdeteksi pada gambar sumber"
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Gagal membuka video"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
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
        
        cap.release()
        out.release()
        
        if progress_callback:
            progress_callback(total_frames, total_frames, "Selesai!")
        
        return True, f"Video berhasil diproses ({frame_idx} frame)"
        
    except Exception as e:
        return False, str(e)
