# realtime_tam.py
import sys, json, contextlib, tempfile, os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import argparse

from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor
from helper import (
    get_device, DEFAULT_MODEL_CFG, DEFAULT_TAM_CHECKPOINT, _json_reply, _read_frame, masks_to_uint8_batch, _save_imgs_path
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
    print("TAM model built successfully.", file=sys.stderr)
    return TAMState(predictor=predictor, device=device)

# ----------------------------
# Core API
# ----------------------------
def start(
    state: TAMState,
    first_frame: np.ndarray,
    *,
    obj_id: int,
    points: np.ndarray,
    labels: np.ndarray,
    frame_idx: int = 0,
) -> None:
    """
    Initialize the sequence on the first frame with prompts.
    Must be called once before tracking.
    """
    print("**** Starting new sequence...", file=sys.stderr)
    if state.initialized:
        return
    state.predictor.load_first_frame(first_frame)
    print("**** Loaded first frame...", file=sys.stderr)
    state.predictor.add_new_prompts(
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )
    print("**** Add new promts...", file=sys.stderr)
    state.initialized = True
    state.frame_idx = frame_idx
    state.obj_id = obj_id

def track(state: TAMState, frame: np.ndarray) -> Tuple[List[int], torch.Tensor]:
    print(f"**** Tracking frame {state.frame_idx+1}...", file=sys.stderr)
    if not state.initialized:
        raise RuntimeError("Call start(...) once before track(...).")
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if state.device.type == "cuda" else contextlib.nullcontext()
    with torch.inference_mode(), amp_ctx:
        out_obj_ids, out_mask_logits = state.predictor.track(frame)
    state.frame_idx += 1
    return out_obj_ids, out_mask_logits


# ----------------------------
# Long-lived worker (stdin/stdout JSON)
# ----------------------------
def run_worker(
    auto_build: bool = False,
    model_cfg: str = str(DEFAULT_MODEL_CFG),
    tam_checkpoint: str = str(DEFAULT_TAM_CHECKPOINT),
    device_pref: Optional[str] = None,
):
    """
    Protocol: send one JSON object per line; get one JSON line reply.
    Supported ops:
      - {"op":"build","model_cfg":..., "checkpoint":..., "device":"auto|cuda|cpu|mps"}
      - {"op":"start","frame_path":..., "obj_id":0, "points":[[x,y],...], "labels":[1,...], "frame_idx":0}
      - {"op":"track","frame_path":..., "out_logits_path":"/path/opt.npy"}   # returns path + obj ids
      - {"op":"status"}
      - {"op":"reset"}             # forget init (keep predictor)
      - {"op":"rebuild", ...}      # destroy and rebuild predictor
      - {"op":"quit"}
    """
    print("**** EfficientTAM long-lived worker starting...", file=sys.stderr)
    state: Optional[TAMState] = None

    try:
        if auto_build:
            dev = _select_device(device_pref)
            state = build_predictor(model_cfg=model_cfg, tam_checkpoint=tam_checkpoint, device=dev)

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                op = req.get("op")
                if op == "build":
                    dev = _select_device(req.get("device"))
                    state = build_predictor(
                        model_cfg=req.get("model_cfg", model_cfg),
                        tam_checkpoint=req.get("checkpoint", tam_checkpoint),
                        device=dev,
                    )
                    _json_reply({"ok": True, "device": str(state.device)})

                elif op == "start":
                    if state is None:
                        raise RuntimeError("Predictor not built. Call op=build first.")
                    frame = _read_frame(req["frame_path"])
                    points = np.array(req["points"], dtype=np.float32)
                    labels = np.array(req["labels"], dtype=np.int32)
                    start(
                        state,
                        frame,
                        obj_id=int(req["obj_id"]),
                        points=points,
                        labels=labels,
                        frame_idx=int(req.get("frame_idx", 0)),
                    )
                    _json_reply({"ok": True})

                elif op == "track":
                    if state is None:
                        raise RuntimeError("Predictor not built. Call op=build first.")
                    frame = _read_frame(req["frame_path"])
                    out_obj_ids, out_mask_logits = track(state, frame)
                    masks = masks_to_uint8_batch(out_mask_logits)

                    img_paths = _save_imgs_path(masks, req.get("out_imgs_path"))

                    _json_reply({
                        "ok": True,
                        "out_obj_ids": out_obj_ids,
                        "out_img_paths": img_paths,
                        "frame_idx": state.frame_idx,
                    })

                elif op == "status":
                    s = None
                    if state is not None:
                        s = {
                            "device": str(state.device),
                            "initialized": state.initialized,
                            "frame_idx": state.frame_idx,
                            "obj_id": state.obj_id,
                        }
                    _json_reply({"ok": True, "state": s})

                elif op == "reset":
                    if state is None:
                        _json_reply({"ok": True})  # nothing to do
                    else:
                        state.initialized = False
                        state.frame_idx = 0
                        _json_reply({"ok": True})

                elif op == "rebuild":
                    # free old predictor (best-effort)
                    if state is not None:
                        try:
                            del state.predictor
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    dev = _select_device(req.get("device"))
                    state = build_predictor(
                        model_cfg=req.get("model_cfg", model_cfg),
                        tam_checkpoint=req.get("checkpoint", tam_checkpoint),
                        device=dev,
                    )
                    _json_reply({"ok": True, "device": str(state.device)})

                elif op == "quit":
                    _json_reply({"ok": True})
                    break

                else:
                    _json_reply({"ok": False, "error": f"unknown op '{op}'"})

            except Exception as e:
                _json_reply({"ok": False, "error": str(e)})

    finally:
        # best-effort cleanup
        try:
            if state is not None:
                del state.predictor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass

# ----------------------------
# (Optional) minimal CLI to start worker immediately
# ----------------------------

"""
python -u main.py --auto_build --device auto
Usage:

import json, subprocess, numpy as np

p = subprocess.Popen(["python","-u","main.py","--auto_build","--device","auto"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

def rpc(msg):
    p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()
    return json.loads(p.stdout.readline())

# seed
rpc({"op":"start","frame_path":"./images/1200_900/2/0000.png",
     "obj_id":0,"points":[[980,759]],"labels":[1],"frame_idx":0})

# track
res = rpc({"op":"track","frame_path":"./images/1200_900/2/0001.png",
           "out_logits_path":"./tmp_logits.npy"})
ids = res["out_obj_ids"]
logits = np.load(res["out_logits_path"])  # (N,1,H,W)


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
    print("Received arguments:", args, file=sys.stderr)  # << change here
    run_worker(
        auto_build=args.auto_build,
        model_cfg=args.model_cfg,
        tam_checkpoint=args.checkpoint,
        device_pref=args.device,
    )
    return 0

if __name__ == "__main__":
    # Run with: python -u realtime_tam.py --auto_build --device auto
    raise SystemExit(main())
