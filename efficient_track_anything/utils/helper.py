import numpy as np
import torch
from typing import List
import tempfile
from pathlib import Path
import os
import sys
import cv2
import json
from typing import Optional
from pathlib import Path
from typing import Union


# ----------------------------
# Configuration (defaults)
# ----------------------------
HERE = Path(__file__).resolve().parent          # .../RealtimeEfficientTAM/RealtimeEfficientTAM/src/realtime_efficient_tam
REPO = HERE.parent.parent                           # repo root (one level up from notebooks)

DEFAULT_TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_ti_512x512.pt")
DEFAULT_MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_ti_512x512.yaml")


# ----------------------------
# Utilities
# ----------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            # Enable TF32 on Ampere+ for speed
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Note: MPS support is preliminary; outputs may differ vs CUDA.")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device


def overlay_mask_bgr(
    frame_bgr: np.ndarray, mask_uint8: np.ndarray, alpha: float = 0.35, color=(0, 255, 0)
) -> np.ndarray:
    """
    frame_bgr: HxWx3 uint8 (OpenCV image)
    mask_uint8: HxW uint8, 0 or 255 (binary)
    alpha: opacity of the colored overlay
    color: BGR tuple
    """
    overlay = frame_bgr.copy()
    m = np.ascontiguousarray(mask_uint8)
    color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_layer[m > 0] = color
    blended = overlay.astype(np.float32)
    blended[m > 0] = (1 - alpha) * blended[m > 0] + alpha * color_layer[m > 0]
    return blended.astype(np.uint8)


# ----------------------------
# Helpers for CLI
# ----------------------------
def masks_to_uint8_batch(out_mask_logits: torch.Tensor) -> List[np.ndarray]:
    """Convert (N,1,H,W) logits -> list of (H,W) uint8 {0,255} masks."""
    masks: List[np.ndarray] = []
    for i in range(out_mask_logits.shape[0]):
        mask_t = (out_mask_logits[i] > 0.0).to(torch.uint8).squeeze(0)
        masks.append(np.ascontiguousarray(mask_t.cpu().numpy() * 255))
    return masks

def read_frame(path: str) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return frame


def save_imgs(
    imgs: Union[np.ndarray, list[np.ndarray]],
    out_dir: str,
    filename: str,
    ext: str = ".png"
) -> list[str]:
    """
    Save one or more images to disk with a given base filename.

    Args:
        imgs: A single np.ndarray or a list of np.ndarray images (HxW or HxWxC).
        out_dir: Directory to save images.
        filename: Base filename (without extension).
        ext: File extension (default: ".png").

    Returns:
        List of written file paths.
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize to list
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    written = []
    if len(imgs) == 1:
        outp = out_dir / f"{filename}{ext}"
        cv2.imwrite(str(outp), imgs[0])
        written.append(str(outp))
    else:
        for i, im in enumerate(imgs):
            outp = out_dir / f"{filename}_i{i:02d}{ext}"
            cv2.imwrite(str(outp), im)
            written.append(str(outp))

    return written


def _json_reply(obj: dict):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def overlay_mask_bgr(frame_bgr: np.ndarray, mask_uint8: np.ndarray, alpha: float = 0.35, color=(0, 255, 0)) -> np.ndarray:
    overlay = frame_bgr.copy()
    # ensure mask is (H,W) and contiguous
    m = np.ascontiguousarray(mask_uint8)
    # create 3-channel color layer where mask > 0
    color_layer = np.zeros_like(frame_bgr, dtype=np.uint8)
    color_layer[m > 0] = color
    # alpha blend only on mask area
    blended = overlay.astype(np.float32)
    blended[m > 0] = (1 - alpha) * blended[m > 0] + alpha * color_layer[m > 0]
    return blended.astype(np.uint8)
