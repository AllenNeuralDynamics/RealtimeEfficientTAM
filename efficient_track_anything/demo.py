# ---------- Demo: run on a folder of frames ----------
from pathlib import Path
import time
import contextlib
import numpy as np
import cv2
from efficient_track_anything.realtime_tam import build_predictor, start, track
from efficient_track_anything.utils.helper import read_frame, masks_to_uint8_batch, save_imgs, overlay_mask_bgr

REPO = Path(__file__).resolve().parent.parent
IMG_DIR = REPO / "notebooks" / "images" / "1200_900" / "3"
FIRST_FRAME_PATH = IMG_DIR / "1.jpg"
SECOND_FRAME_PATH = IMG_DIR / "2.jpg"
THIRD_FRAME_PATH = IMG_DIR / "3.jpg"
points = np.array([[910, 740], [902,692], [775,518]], dtype=np.float32) # 1200_900/3
labels = np.array([1, 1, 0], dtype=np.int32)

def demo_sequence():
    print("Building model...")
    predictor = build_predictor()
    img = read_frame(FIRST_FRAME_PATH)
    predictor.predictor.load_first_frame(img)

    _, out_mask_logits = start(predictor, points=points, labels=labels)
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


if __name__ == "__main__":
    demo_sequence()