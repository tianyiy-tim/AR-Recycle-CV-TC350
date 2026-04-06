"""
♻️ Recyclable Detector — YOLO-World Edition
=============================================
Zero-shot detection using text prompts. No training needed.
Just tell it what to look for and it finds it.

Setup:
    pip install ultralytics --upgrade

Usage:
    python sam_clip_detector.py
"""

import time
import cv2
import numpy as np
from ultralytics import YOLOWorld

# ── Define what to detect ──────────────────────────────────────────────

CLASSES = [
    "plastic bottle",
    "cardboard box",
    "glass bottle",
    "aluminum can",
    "paper",
    "plastic bag",
    "styrofoam cup",
    "plastic container",
    "tin can",
    "glass jar",
    "water bottle",
    "soda can",
    "cereal box",
    "milk carton",
]

# Map each class to bin type and color
CLASS_INFO = {
    "plastic bottle":    ("Recycling", (255, 140, 0)),
    "cardboard box":     ("Recycling", (0, 180, 255)),
    "glass bottle":      ("Recycling", (0, 255, 100)),
    "aluminum can":      ("Recycling", (200, 200, 200)),
    "paper":             ("Recycling", (255, 255, 100)),
    "plastic bag":       ("Trash",     (0, 0, 255)),
    "styrofoam cup":     ("Trash",     (180, 100, 255)),
    "plastic container": ("Recycling", (0, 220, 220)),
    "tin can":           ("Recycling", (180, 180, 180)),
    "glass jar":         ("Recycling", (100, 255, 100)),
    "water bottle":      ("Recycling", (255, 160, 0)),
    "soda can":          ("Recycling", (210, 210, 210)),
    "cereal box":        ("Recycling", (0, 160, 255)),
    "milk carton":       ("Recycling", (200, 255, 200)),
}


def draw_frame(frame, results, fps):
    display = frame.copy()
    h, w = display.shape[:2]

    detections = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class name
                label = result.names[cls_id]
                info = CLASS_INFO.get(label, ("Unknown", (128, 128, 128)))
                bin_type, color = info

                detections.append({
                    "label": label,
                    "conf": conf,
                    "bin": bin_type,
                    "color": color,
                    "bbox": (x1, y1, x2, y2),
                })

    # Draw detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det["color"]
        conf = det["conf"]
        label = det["label"]
        bin_type = det["bin"]

        # Filled overlay
        overlay = display.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)

        # Bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cl = min(20, (x2-x1)//4, (y2-y1)//4)
        if cl > 3:
            for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                dx = cl if cx == x1 else -cl
                dy = cl if cy == y1 else -cl
                cv2.line(display, (cx,cy), (cx+dx,cy), color, 4)
                cv2.line(display, (cx,cy), (cx,cy+dy), color, 4)

        # Label
        bin_txt = "RECYCLE" if bin_type == "Recycling" else "TRASH"
        line1 = f"{label} ({conf:.0%})"
        line2 = f"-> {bin_txt}"
        (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tw = max(tw1, tw2)

        ly = y1 - 8 if y1 > th1 + th2 + 20 else y2 + 8
        lx = max(0, x1)

        cv2.rectangle(display, (lx-1, ly-2), (lx+tw+8, ly+th1+th2+14), color, -1)
        cv2.putText(display, line1, (lx+3, ly+th1+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(display, line2, (lx+3, ly+th1+th2+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # HUD
    cv2.putText(display, f"FPS: {fps:.1f} | YOLO-World | Q:Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(display, f"Detections: {len(detections)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return display


def main():
    # Load model
    print("[INFO] Loading YOLO-World model...")
    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(CLASSES)
    print(f"[INFO] Detecting: {', '.join(CLASSES)}")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No camera")
        return

    print("[INFO] Warming up camera...")
    time.sleep(2)
    for _ in range(10):
        cap.read()

    ret, test = cap.read()
    if not ret:
        print("[ERROR] No frames")
        cap.release()
        return
    print(f"[INFO] Camera: {test.shape[1]}x{test.shape[0]}")

    cv2.namedWindow("Recyclable Detector", cv2.WINDOW_AUTOSIZE)

    fps = 0.0
    prev_time = time.time()

    print("[INFO] Running. Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        # Run YOLO-World on every frame — it's fast enough
        results = model(frame, conf=0.1, verbose=False)

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 0.001))
        prev_time = now

        display = draw_frame(frame, results, fps)
        cv2.imshow("Recyclable Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()