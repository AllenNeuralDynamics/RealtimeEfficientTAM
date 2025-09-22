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
    #state.predictor.load_first_frame(first_frame)
    print("[Realtime Efficient TAM] Loaded first frame...")
    _, out_obj_ids, out_mask_logits = state.predictor.add_new_prompts(
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

    return out_obj_ids, out_mask_logits

def track(state: TAMState, frame: np.ndarray) -> Tuple[List[int], torch.Tensor]:
    print(f"[Realtime Efficient TAM] Tracking frame {state.frame_idx+1}...")
    if not state.initialized:
        raise RuntimeError("Call start(...) once before track(...).")
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if state.device.type == "cuda" else contextlib.nullcontext()
    with torch.inference_mode(), amp_ctx:
        out_obj_ids, out_mask_logits = state.predictor.track(frame)
    state.frame_idx += 1
    return out_obj_ids, out_mask_logits