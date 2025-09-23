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

def find_matching_cfg(ckpt_path: Path) -> Optional[Path]:
    """
    Given a checkpoint path, return the corresponding model config path if it exists.

    Args:
        ckpt_path (str | Path): Path to the checkpoint file.

    Returns:
        Optional[Path]: Path to the matching YAML config, or None if not found.
    """
    if not ckpt_path.is_file():
        return None

    name = ckpt_path.stem.lower()  # filename without extension
    model_config_path = (
        REPO
        / "efficient_track_anything"
        / "configs"
        / "efficienttam"
        / f"{name}.yaml"
    )

    return model_config_path if model_config_path.is_file() else None

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

def _to_np_xy(pts):
    """
    Accepts: (N,2) array-like or a single (2,) pair.
    Returns: (N,2) float32 numpy array.
    """
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr

def converter_pts_after_crop(pts, left, top):
    """
    Convert full-image XY points to crop-relative XY by subtracting the crop origin.
    pts: (N,2) or (2,) in global image coordinates
    left, top: crop's top-left corner in the global image
    """
    original_shape = pts.shape
    pts = _to_np_xy(pts).copy()
    pts[:, 0] -= float(left)
    pts[:, 1] -= float(top)
    # Return in original format
    if original_shape == (2,):  # Single point input
        return pts.squeeze()  # Return as (2,) not (1,2)
    else:
        return pts

def converter_pts_after_resize(pts, src_wh, dst_wh):
    """
    Scale crop-relative XY points to the resized image coordinates.
    src_wh: (src_w, src_h) of the crop BEFORE resize
    dst_wh: (dst_w, dst_h) of the resized local image
    """
    original_shape = pts.shape
    pts = _to_np_xy(pts).copy()
    src_w, src_h = float(src_wh[0]), float(src_wh[1])
    dst_w, dst_h = float(dst_wh[0]), float(dst_wh[1])
    sx = dst_w / src_w
    sy = dst_h / src_h
    pts[:, 0] *= sx
    pts[:, 1] *= sy
        # Return in original format
    if original_shape == (2,):  # Single point input
        return pts.squeeze()  # Return as (2,) not (1,2)
    else:
        return pts

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

def mask_to_bbox_xyxy(mask_u8: np.ndarray, img_shape=None, pad: int = 10):
    """
    Tight bbox from a uint8 mask considering ALL foreground pixels.
    Returns (left, top, right, bottom) with right/bottom EXCLUSIVE: [x1, y1, x2, y2).
    If img_shape is None, uses mask shape for clamping.
    """
    # Ensure uint8, normalize to binary {0,1}
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)
    mask_bin = (mask_u8 > 0).astype(np.uint8)

    # Empty mask -> None
    if cv2.countNonZero(mask_bin) == 0:
        return None

    # Tight bbox over all foreground pixels
    ys, xs = np.where(mask_bin > 0)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    # Apply padding
    x1 -= pad; y1 -= pad; x2 += pad; y2 += pad

    # Clamp to image bounds
    if img_shape is None:
        h, w = mask_u8.shape[:2]
    else:
        h, w = img_shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)  # (left, top, right, bottom), right/bottom are EXCLUSIVE

def lift_local_mask_to_global(mask_local_u8, bbox, global_hw):
    """
    Take a local (resized-crop) mask and paste it back into a full-frame mask.
    - mask_local_u8: (h_local, w_local) uint8 mask (0/255)
    - [left, top, right, bottom]: crop bbox in global coords (exclusive right/bottom)
    - global_hw: (H, W) of the original image
    Returns: (H, W) uint8 mask with the local mask placed into the bbox region.
    """
    left, top, right, bottom = bbox
    H, W = int(global_hw[0]), int(global_hw[1])
    out = np.zeros((H, W), dtype=mask_local_u8.dtype)

    crop_w = int(right - left)
    crop_h = int(bottom - top)
    if crop_w <= 0 or crop_h <= 0:
        return out  # empty (defensive)

    # Resize local mask back to the crop size
    local_up = cv2.resize(mask_local_u8, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    # Paste into full-frame
    out[top:bottom, left:right] = local_up
    return out