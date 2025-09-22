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
                                                   lift_local_mask_to_global,
                                                   find_matching_cfg)

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "notebooks" / "images" / "1200_900" / "3"
FIRST_FRAME_PATH = IMG_DIR / "1.jpg"
SECOND_FRAME_PATH = IMG_DIR / "2.jpg"
THIRD_FRAME_PATH = IMG_DIR / "3.jpg"


points = np.array([[879, 737]], dtype=np.float32) # 1200_900/3 # first shank
#points = np.array([[909, 709]], dtype=np.float32) # 1200_900/3 # 4th shank
labels = np.array([1], dtype=np.int32)
h, w = 512, 512

TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_s_512x512.pt")
MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_s_512x512.yaml")
#TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_ti_512x512.pt")
#MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_ti_512x512.yaml")

def preprocess(img_local):
    # adaptive Thresholding
    gray = cv2.cvtColor(img_local, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(IMG_DIR / "21_local_gray.png"), gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(IMG_DIR / "21_local_thresh.png"), thresh)
    img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return img

def track_local(mask_global, img, predictor_local, pt_global=None, labels=labels, pad=20):
    bbox = mask_to_bbox_xyxy(mask_global[0], img.shape, pad=pad)  # (x1,y1,x2,y2)
    if not bbox:
        raise RuntimeError("No foreground detected in the first frame.")
    
    left, top, right, bottom = bbox
    crop = img[top:bottom, left:right]                  # exclusive right/bottom
    crop_h, crop_w = crop.shape[:2]
    local_img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    #local_img = preprocess(local_img)
    
    if pt_global is not None: # Start tracking with a point prompt
        # --- Convert points: global -> crop-relative -> resized (local) ---
        pt_crop = converter_pts_after_crop(pt_global, left=left, top=top)                 # to crop coords
        pt_local = converter_pts_after_resize(pt_crop, src_wh=(crop_w, crop_h), dst_wh=(w, h))  # to local coords
        predictor_local.predictor.load_first_frame(local_img)
        _, out_mask_logits = start(predictor_local, local_img, points=pt_local, labels=labels)
    else:
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
    predictor_global.predictor.load_first_frame(img)

    _, out_mask_logits = start(predictor_global, img, points=points, labels=labels)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    mask_local_global = track_local(mask_global, img, predictor_local, pt_global=points, labels=labels)
    # Visualize
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local_global, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="22_output")


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
