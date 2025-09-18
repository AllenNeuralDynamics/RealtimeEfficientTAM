# ---------- Demo: run on a folder of frames ----------
from pathlib import Path
import time
import contextlib
import numpy as np
import cv2
from efficient_track_anything.realtime_tam import build_predictor, start, track
from efficient_track_anything.utils.helper import (read_frame,
                                                   masks_to_uint8_batch,
                                                   save_imgs,
                                                   overlay_mask_bgr,
                                                   mask_to_bbox_xyxy,
                                                   converter_pts_after_crop,
                                                   converter_pts_after_resize,
                                                   lift_local_mask_to_global)

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "notebooks" / "images" / "1200_900" / "3"
FIRST_FRAME_PATH = IMG_DIR / "1.jpg"
SECOND_FRAME_PATH = IMG_DIR / "2.jpg"
THIRD_FRAME_PATH = IMG_DIR / "3.jpg"

points = np.array([[879, 737]], dtype=np.float32) # 1200_900/3 # first shank
points = np.array([[909, 709]], dtype=np.float32) # 1200_900/3 # 4th shank
#points = np.array([[1758, 1224]], dtype=np.float32) # 1200_900/3 # 4th shank
#points = np.array([[622, 244]], dtype=np.float32) # 1200_900/3 # 4th shank
labels = np.array([1], dtype=np.int32)
h, w = 512, 512

#TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_s_512x512.pt")
#MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_s_512x512.yaml")
TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_ti_512x512.pt")
MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_ti_512x512.yaml")


def track_local(mask_global, img, predictor_local, pt_global=None, pad=20):
    bbox = mask_to_bbox_xyxy(mask_global[0], img.shape, pad=pad)  # (x1,y1,x2,y2)
    if not bbox:
        raise RuntimeError("No foreground detected in the first frame.")
    
    left, top, right, bottom = bbox
    crop = img[top:bottom, left:right]                  # exclusive right/bottom
    crop_h, crop_w = crop.shape[:2]
    local_img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    
    
    if pt_global is not None: # Start tracking with a point prompt
        # --- Convert points: global -> crop-relative -> resized (local) ---
        pt_crop = converter_pts_after_crop(pt_global, left=left, top=top)                 # to crop coords
        pt_local = converter_pts_after_resize(pt_crop, src_wh=(crop_w, crop_h), dst_wh=(w, h))  # to local coords
        start(predictor_local, local_img, points=pt_local, get_single_connected_component=True)

    _, out_mask_logits = track(predictor_local, local_img)
    mask_local = masks_to_uint8_batch(out_mask_logits)

    # mask_local[0] matches local_img size (w,h); lift it back to full-frame
    H, W = img.shape[:2]
    mask_local_global = lift_local_mask_to_global(mask_local[0], left, top, right, bottom, (H, W))

    return mask_local_global

def demo_sequence():
    print("**** FIRST FRAME ****")
    predictor_global = build_predictor()
    predictor_local = build_predictor(model_cfg=str(MODEL_CFG), tam_checkpoint=str(TAM_CHECKPOINT))
    img = read_frame(FIRST_FRAME_PATH)                   # shape (H, W, 3)
    start(predictor_global, img, points=points)
    _, out_mask_logits = track(predictor_global, img)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    mask_local_global = track_local(mask_global, img, predictor_local, points)

    # Visualize
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local_global, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="1_output")

    print("\n**** SECOND FRAME ****")
    img = read_frame(SECOND_FRAME_PATH)
    _, out_mask_logits = track(predictor_global, img)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    mask_local_global = track_local(mask_global, img, predictor_local)
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local_global, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="2_output")

    print("\n**** THIRD FRAME ****")
    img = read_frame(THIRD_FRAME_PATH)
    _, out_mask_logits = track(predictor_global, img)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    mask_local_global = track_local(mask_global, img, predictor_local)
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local_global, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="3_output")

demo_sequence()
