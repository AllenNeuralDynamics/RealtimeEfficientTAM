# ---------- Demo: run on a folder of frames ----------
from pathlib import Path
import time
import contextlib
import numpy as np
import cv2
from efficient_track_anything.realtime_tam import build_predictor, start_with_mask, track
from efficient_track_anything.utils.helper import read_frame, masks_to_uint8_batch, save_imgs, overlay_mask_bgr

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "notebooks" / "images" / "1200_900" / "3"
FIRST_FRAME_PATH = IMG_DIR / "1.jpg"
SECOND_FRAME_PATH = IMG_DIR / "2.jpg"
THIRD_FRAME_PATH = IMG_DIR / "3.jpg"
MASK = IMG_DIR / "1_mask.png"
points = np.array([[952, 768]], dtype=np.float32) # 1200_900/3
labels = np.array([1, 0, 0], dtype=np.int32)

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
        cv2.circle(out, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
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

def demo_sequence():
    print("Building model...")
    predictor = build_predictor()
    img = read_frame(FIRST_FRAME_PATH)
    mask = read_frame(MASK)
    # convert to binary
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    predictor.predictor.load_first_frame(img)

    _, out_mask_logits = start_with_mask(predictor, mask=mask)
    mask = masks_to_uint8_batch(out_mask_logits)
    overlay = overlay_mask_bgr(img, mask[0], alpha=0.3)
    save_imgs(overlay, out_dir=IMG_DIR, filename = "1_output")
    print("Saved masks for first frame to " + str(IMG_DIR / "1_output.png"))

    img = read_frame(SECOND_FRAME_PATH)
    _, out_mask_logits = track(predictor, img)
    mask = masks_to_uint8_batch(out_mask_logits)
    overlay = overlay_mask_bgr(img, mask[0], alpha=0.4)
    save_imgs(overlay, out_dir=IMG_DIR, filename = "2_output")
    print("Saved masks for second frame to " + str(IMG_DIR / "2_output.png"))

    img = read_frame(THIRD_FRAME_PATH)
    _, out_mask_logits = track(predictor, img)
    mask = masks_to_uint8_batch(out_mask_logits)
    overlay = overlay_mask_bgr(img, mask[0], alpha=0.4)
    save_imgs(overlay, out_dir=IMG_DIR, filename = "3_output")
    print("Saved masks for third frame to " + str(IMG_DIR / "3_output.png"))

#demo_sequence()
detect_line_on_pt(read_frame(FIRST_FRAME_PATH), points[0])
demo_sequence()
