# ---------- Demo: run on a folder of frames ----------
from pathlib import Path
import time
import contextlib
import numpy as np
import cv2
from efficient_track_anything.realtime_tam import build_predictor, start, track, start_with_mask
from efficient_track_anything.utils.helper import (read_frame,
                                                   masks_to_uint8_batch,
                                                   save_imgs,
                                                   overlay_mask_bgr,
                                                   mask_to_bbox_xyxy,
                                                   converter_pts_after_crop,
                                                   converter_pts_after_resize,
                                                   lift_local_mask_to_global,
                                                )

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "notebooks" / "images" / "1200_900" / "3"
FIRST_FRAME_PATH = IMG_DIR / "1.jpg"
SECOND_FRAME_PATH = IMG_DIR / "2.jpg"
THIRD_FRAME_PATH = IMG_DIR / "3.jpg"

#points = np.array([[879, 737]], dtype=np.float32) # 1200_900/3 # first shank
points = np.array([[909, 709]], dtype=np.float32) # 1200_900/3 # 4th shank
labels = np.array([1], dtype=np.int32)
h, w = 512, 512

TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_s_512x512.pt")
MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_s_512x512.yaml")
#TAM_CHECKPOINT = (REPO / "checkpoints" / "efficienttam_ti_512x512.pt")
#MODEL_CFG     = (REPO / "efficient_track_anything" / "configs" / "efficienttam" / "efficienttam_ti_512x512.yaml")

def point_to_segment_dist(pt, a, b):
    # pt, a, b are (x, y)
    p = np.array(pt, dtype=float)
    A = np.array(a, dtype=float)
    B = np.array(b, dtype=float)
    AB = B - A
    if np.allclose(AB, 0):
        return np.linalg.norm(p - A)
    t = np.dot(p - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0.0, 1.0)
    proj = A + t * AB
    return np.linalg.norm(p - proj)

def detect_line_on_pt(img, pt, mask=None):
    out = img.copy()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adptive thresholding to get binary image
    bin_adapt = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (3, 3), 0), 50, 150)  # white edges (non-zero), black background
    img = cv2.bitwise_or(edges, bin_adapt)
    if mask is not None:
        # make mask little bit larger using dilation
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        img = cv2.bitwise_and(img, mask)

    # Use Hough Transform to detect lines
    mLength = 200
    linesP = cv2.HoughLinesP(img, 1, np.pi/180, threshold=80, minLineLength=mLength, maxLineGap=5)
    tol = 5.0
    hits = []
    if linesP is not None:
        for x1, y1, x2, y2 in linesP[:, 0]:
            # If your helper returns (dist, proj), use: dist, _ = self._point_to_segment_dist(...)
            dist = point_to_segment_dist(pt, (x1, y1), (x2, y2))
            if isinstance(dist, (tuple, list)):        # handle (dist, proj)
                dist = dist[0]
            if dist <= tol:
                hits.append((x1, y1, x2, y2, dist))
    #out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # img must be single-channel here
    if hits:
        for x1, y1, x2, y2, dist in hits:
            print(f"Line near point (d={dist:.2f}): ({x1},{y1})-({x2},{y2})")
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # draw the selected point last so it’s on top
        #cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
        save_imgs(out, out_dir=IMG_DIR, filename = "line_detected")
    else:
        print("No line detected near the point.")

    # Use detected lines to create a mask
    mask = np.zeros(img.shape, dtype=np.uint8)
    if hits:
        for x1, y1, x2, y2, dist in hits:
            cv2.line(mask, (x1, y1), (x2, y2), 255, 5)  # white line on black background
        save_imgs(mask, out_dir=IMG_DIR, filename = "1_mask")
    else:
        print("No line mask created.")

    return mask

def crop_and_resize(bbox, img, w=512, h=512):
    left, top, right, bottom = bbox
    crop = img[top:bottom, left:right]
    resized_img = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized_img

def convert_pts_after_crop_resize(pts, bbox, w=512, h=512):
    left, top, right, bottom = bbox
    crop_w = int(right - left)
    crop_h = int(bottom - top)

    pts_local = []
    if pts is not None:
        # Start tracking with a point prompt
        # --- Convert points: global -> crop-relative -> resized (local) ---
        for pt in pts:
            pt_crop = converter_pts_after_crop(pt, left=left, top=top)  # to crop coords
            pt_local = converter_pts_after_resize(pt_crop, src_wh=(crop_w, crop_h), dst_wh=(w, h))  # to local coords
            pts_local.append(pt_local)

        # Convert back to same format as input
        pts_local = np.array(pts_local, dtype=np.float32)

    return pts_local

def track_local(predictor_local, mask_global, img, points=None):
    # Local - Preprocessing
    # crop the global mask to get initial local mask
    bbox = mask_to_bbox_xyxy(mask_global, img.shape, pad=20)  # (x1,y1,x2,y2)
    if not bbox:
        raise RuntimeError("No foreground detected in the first frame.")
    img_local = crop_and_resize(bbox, img)
    mask_local = crop_and_resize(bbox, mask_global)

    if points is not None:
        print("points:", points)
        points_local = convert_pts_after_crop_resize(points, bbox)  # to crop coords
        print("points_local:", points_local)
        mask_line = detect_line_on_pt(img_local, points_local[0], mask=mask_local)  # Generate mask for line
        # Start Local
        predictor_local.predictor.load_first_frame(img_local)
        _, out_mask_logits = start_with_mask(predictor_local, mask=mask_line)
    else:
        _, out_mask_logits = track(predictor_local, img_local)

    mask_local = masks_to_uint8_batch(out_mask_logits)

    # Post processing Lift local mask to global
    # mask_local[0] matches local_img size (w,h); lift it back to full-frame
    H, W = img.shape[:2]
    mask_local_global = lift_local_mask_to_global(mask_local[0], bbox, (H, W))

    return mask_local_global


def demo_sequence():
    print("**** FIRST FRAME ****")
    predictor_global = build_predictor()
    predictor_local = build_predictor(model_cfg=str(MODEL_CFG), tam_checkpoint=str(TAM_CHECKPOINT))
    img = read_frame(FIRST_FRAME_PATH)                   # shape (H, W, 3)

    # Start Global
    predictor_global.predictor.load_first_frame(img)
    _, out_mask_logits = start(predictor_global, points=points, labels=labels)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)

    # Start Local
    mask_local = track_local(predictor_local, mask_global[0], img, points=points)
     # Visualize
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="1_output")

    print("\n**** SECOND FRAME ****")
    img = read_frame(SECOND_FRAME_PATH)
    _, out_mask_logits = track(predictor_global, img)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    mask_local_global = track_local(predictor_local, mask_global[0], img)
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local_global, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="2_output")

    print("\n**** THIRD FRAME ****")
    img = read_frame(THIRD_FRAME_PATH)
    _, out_mask_logits = track(predictor_global, img)
    mask_global = masks_to_uint8_batch(out_mask_logits)
    mask_local_global = track_local(predictor_local, mask_global[0], img)
    overlay = overlay_mask_bgr(img, mask_global[0], alpha=0.1)
    overlay = overlay_mask_bgr(overlay, mask_local_global, alpha=0.3, color=(200, 0, 200))
    save_imgs(overlay, out_dir=IMG_DIR, filename="3_output")

demo_sequence()
