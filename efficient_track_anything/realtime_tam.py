# realtime_tam.py
import sys, json, contextlib, tempfile, os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import argparse

from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from efficient_track_anything.utils.helper import (
    get_device, DEFAULT_MODEL_CFG, DEFAULT_TAM_CHECKPOINT, _json_reply, read_frame, masks_to_uint8_batch
)




# ----------------------------
# State + Builder
# ----------------------------
@dataclass
class TAMState:
    predictor: any
    device: torch.device
    initialized: bool = False
    frame_idx: int = 0
    obj_id: int = 0  # default object id to reuse if desired

def _select_device(pref: Optional[str]) -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if pref == "mps":
        mps = getattr(torch.backends, "mps", None)
        return torch.device("mps") if (mps and mps.is_available()) else torch.device("cpu")
    return get_device()

def build_predictor(
    model_cfg: str = str(DEFAULT_MODEL_CFG),
    tam_checkpoint: str = str(DEFAULT_TAM_CHECKPOINT),
    device: Optional[torch.device] = None,
) -> TAMState:
    """
    Build the EfficientTAM camera predictor and wrap it in a TAMState.
    """
    device = device or get_device()
    predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint, device=device)
    print("[Realtime Efficient TAM] TAM model built successfully.")
    return TAMState(predictor=predictor, device=device)

# ----------------------------
# Core API
# ----------------------------
def start(
    state: TAMState,
    first_frame: np.ndarray,
    points: np.ndarray = np.array([[0, 0]], dtype=np.float32),
    obj_id: int = 0,
    labels: np.ndarray = np.array([1], dtype=np.int32),
    frame_idx: int = 0,
    get_single_connected_component: bool = False,
) -> None:
    """
    Initialize the sequence on the first frame with prompts.
    Must be called once before tracking.
    """
    print("[Realtime Efficient TAM] Starting new sequence...")
    if state.initialized:
        return
    state.predictor.load_first_frame(first_frame)
    print("[Realtime Efficient TAM] Loaded first frame...")
    state.predictor.add_new_prompts(
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
        get_single_connected_component=get_single_connected_component,
    )
    print("[Realtime Efficient TAM] Added new prompts...")
    state.initialized = True
    state.frame_idx = frame_idx
    state.obj_id = obj_id

def track(state: TAMState, frame: np.ndarray) -> Tuple[List[int], torch.Tensor]:
    print(f"[Realtime Efficient TAM] Tracking frame {state.frame_idx+1}...")
    if not state.initialized:
        raise RuntimeError("Call start(...) once before track(...).")
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if state.device.type == "cuda" else contextlib.nullcontext()
    with torch.inference_mode(), amp_ctx:
        out_obj_ids, out_mask_logits = state.predictor.track(frame)
    state.frame_idx += 1
    return out_obj_ids, out_mask_logits

"""
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EfficientTAM long-lived worker (JSON over stdin/stdout).")
    p.add_argument("--auto_build", action="store_true", help="Build predictor immediately on start.")
    p.add_argument("--model_cfg", type=str, default=str(DEFAULT_MODEL_CFG))
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_TAM_CHECKPOINT))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    return p.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print("Hello")
    # do stuff with args if you want, e.g. print(args)
    return 0

if __name__ == "__main__":
    # Run with: python -u realtime_tam.py --auto_build --device auto
    raise SystemExit(main())
"""

"""
#----------

from pathlib import Path
import numpy as np
import time


# --- Model init (unchanged) ---
# tam_real_time.py

HERE = Path(__file__).resolve().parent          # .../RealtimeEfficientTAM/RealtimeEfficientTAM/notebooks
REPO = HERE.parent                              # repo root (one level up from notebooks)

tam_checkpoint = (REPO / "checkpoints" / "efficienttam_ti_512x512.pt")
model_cfg     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_ti_512x512.yaml")

print("ckpt:", tam_checkpoint)
print("cfg :", model_cfg)

if not tam_checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {tam_checkpoint}")
if not model_cfg.is_file():
    raise FileNotFoundError(f"Config not found: {model_cfg}")

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable TF32 on Ampere+ for speed
        try:
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

device = get_device()
print(tam_checkpoint)
predictor = build_efficienttam_camera_predictor(str(model_cfg), str(tam_checkpoint), device=device)
print("TAM model built successfully.")

# Input image folder
#IMG_DIR = Path("./images/1200_900/2")
IMG_DIR = REPO / "notebooks" / "images" / "1200_900" / "2"
image_paths = sorted([p for p in IMG_DIR.glob("*.jpg")] + [p for p in IMG_DIR.glob("*.png")])
if not image_paths:
    raise RuntimeError(f"No images found in {IMG_DIR}")
# 👇 adjust point to something inside your object
points = np.array([[980, 759]], dtype=np.float32) # 1200_900/2
#points = np.array([[910, 740]], dtype=np.float32) # 1200_900/3
labels = np.array([1], dtype=np.int32)

#Output folder
out_dir = IMG_DIR / "output_tam"
out_dir.mkdir(parents=True, exist_ok=True)
video_segments = {}

if_init = False
frame_idx = 0
amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext()

with torch.inference_mode(), amp_ctx:
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Skipping unreadable file: {img_path}")
            continue

        if not if_init:
            print("Loading first frame...")
            predictor.load_first_frame(frame)

            obj_id = 0

            predictor.add_new_prompts(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            if_init = True
        else:
            frame_idx += 1
            print(frame_idx, f"Processing {img_path.name}...")

            frame_start = time.time()
            out_obj_ids, out_mask_logits = predictor.track(frame)
            frame_end = time.time()
            frame_time = frame_end - frame_start
            print(f"Frame {frame_idx} ({img_path.name}) processed in {frame_time:.3f} sec "
                f"({1/frame_time:.2f} FPS)")

            video_segments[frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

            # --- save masks & overlays for this frame ---
            for i, out_obj_id in enumerate(out_obj_ids):
                # binary mask (H,W) uint8 in {0,255}
                mask_t = (out_mask_logits[i] > 0.0).to(torch.uint8).squeeze(0)
                mask = np.ascontiguousarray(mask_t.cpu().numpy() * 255)

                # pick a color per object (stable but distinct-ish)
                palette = [(0,255,0), (0,165,255), (255,0,0), (255,0,255), (0,255,255), (255,255,0)]
                color = palette[out_obj_id % len(palette)]

                # overlay on the *current frame*
                overlay = overlay_mask_bgr(frame, mask, alpha=0.35, color=color)

                # paths
                out_ovly_path = out_dir / f"frame_{frame_idx:04d}_obj_{out_obj_id}_overlay.png"

                # save (mask optional; overlay is the main visualization)
                cv2.imwrite(str(out_ovly_path), overlay)
"""