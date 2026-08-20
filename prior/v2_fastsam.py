"""
v2: YOLO-World detection plus FastSAM segmentation.

Segments the whole frame unprompted, then keeps whichever mask best overlaps
each detection. That is the expensive way round, and replacing it with
box-prompted MobileSAM is the step to ../detector.py.

Kept for the record, not maintained. See README.md in this directory.

Usage:
    python prior/v2_fastsam.py
"""


import time
import cv2
import numpy as np
from ultralytics import YOLOWorld, FastSAM

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


def get_mask_for_bbox(frame, fastsam_model, bbox):
    """Run FastSAM and find the mask that best overlaps with the YOLO bbox."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    try:
        results = fastsam_model(frame, imgsz=640, conf=0.3, iou=0.7,
                                 retina_masks=True, verbose=False)
    except Exception:
        return None

    if not results or len(results) == 0:
        return None

    result = results[0]
    if result.masks is None:
        return None

    masks = result.masks.data.cpu().numpy()

    best_mask = None
    best_iou = 0

    for mask in masks:
        # Resize mask to frame size if needed
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5

        # Calculate overlap with YOLO bbox
        bbox_mask = np.zeros((h, w), dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True

        intersection = np.logical_and(mask, bbox_mask).sum()
        union = np.logical_or(mask, bbox_mask).sum()
        iou = intersection / max(union, 1)

        # Also check what fraction of the mask is inside the bbox
        mask_in_bbox = intersection / max(mask.sum(), 1)

        # Best mask = high IoU and mostly inside the bbox
        score = iou * 0.5 + mask_in_bbox * 0.5
        if score > best_iou and mask_in_bbox > 0.3:
            best_iou = score
            best_mask = mask.astype(np.uint8)

    return best_mask


def draw_frame(frame, detections, fps):
    display = frame.copy()
    h, w = display.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det["color"]
        conf = det["conf"]
        label = det["label"]
        bin_type = det["bin"]
        mask = det.get("mask")

        if mask is not None and mask.any():
            # Draw colored mask overlay
            overlay = display.copy()
            overlay[mask > 0] = (
                np.array(overlay[mask > 0], dtype=np.float32) * 0.5 +
                np.array(color, dtype=np.float32) * 0.5
            ).astype(np.uint8)
            display = overlay

            # Draw mask contour outline
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, color, 3)
        else:
            # Fallback: bounding box
            overlay = display.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # Label
        bin_txt = "RECYCLE" if bin_type == "Recycling" else "TRASH"
        line1 = f"{label} ({conf:.0%})"
        line2 = f"-> {bin_txt}"
        (tw1, th1), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        (tw2, th2), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tw = max(tw1, tw2)

        # Position label at top of mask or bbox
        if mask is not None and mask.any():
            ys = np.where(mask > 0)[0]
            xs = np.where(mask > 0)[1]
            top_y = int(ys.min())
            lx = max(0, int(xs.min()))
        else:
            top_y = y1
            lx = max(0, x1)

        ly = top_y - 8 if top_y > th1 + th2 + 20 else y2 + 8

        cv2.rectangle(display, (lx-1, ly-2), (lx+tw+8, ly+th1+th2+14), color, -1)
        cv2.putText(display, line1, (lx+3, ly+th1+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(display, line2, (lx+3, ly+th1+th2+8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # HUD
    cv2.putText(display, f"FPS: {fps:.1f} | YOLO-World + FastSAM | Q:Quit",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(display, f"Detections: {len(detections)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return display


def merge_same_class(detections):
    """Merge overlapping/contained detections of the same class into one."""
    if len(detections) <= 1:
        return detections

    # Group by label
    groups = {}
    for det in detections:
        label = det["label"]
        if label not in groups:
            groups[label] = []
        groups[label].append(det)

    merged = []
    for label, dets in groups.items():
        if len(dets) == 1:
            merged.append(dets[0])
            continue

        # Repeatedly merge overlapping boxes until stable
        changed = True
        while changed:
            changed = False
            new_dets = []
            used = [False] * len(dets)

            for i in range(len(dets)):
                if used[i]:
                    continue
                current = dets[i]
                cx1, cy1, cx2, cy2 = current["bbox"]
                best_conf = current["conf"]

                for j in range(i + 1, len(dets)):
                    if used[j]:
                        continue
                    ox1, oy1, ox2, oy2 = dets[j]["bbox"]

                    # Check overlap or containment
                    ix1 = max(cx1, ox1)
                    iy1 = max(cy1, oy1)
                    ix2 = min(cx2, ox2)
                    iy2 = min(cy2, oy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                    area1 = (cx2 - cx1) * (cy2 - cy1)
                    area2 = (ox2 - ox1) * (oy2 - oy1)
                    smaller = min(area1, area2)

                    # Merge if: any overlap OR one box is mostly inside the other
                    overlap_ratio = inter / max(smaller, 1)
                    union = area1 + area2 - inter
                    iou = inter / max(union, 1)

                    if iou > 0.1 or overlap_ratio > 0.5:
                        # Expand current bbox to encompass both
                        cx1 = min(cx1, ox1)
                        cy1 = min(cy1, oy1)
                        cx2 = max(cx2, ox2)
                        cy2 = max(cy2, oy2)
                        best_conf = max(best_conf, dets[j]["conf"])
                        used[j] = True
                        changed = True

                new_dets.append({
                    "label": current["label"],
                    "conf": best_conf,
                    "bin": current["bin"],
                    "color": current["color"],
                    "bbox": (cx1, cy1, cx2, cy2),
                    "mask": None,
                })
                used[i] = True

            dets = new_dets

        merged.extend(dets)

    return merged


def main():
    # Load both models
    print("[INFO] Loading YOLO-World model...")
    yolo = YOLOWorld("yolov8s-worldv2.pt")
    yolo.set_classes(CLASSES)
    print(f"[INFO] Detecting: {', '.join(CLASSES)}")

    print("[INFO] Loading FastSAM model...")
    fastsam = FastSAM("FastSAM-s.pt")
    print("[INFO] FastSAM loaded.")

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
    seg_interval = 3  # run FastSAM every N frames (YOLO runs every frame)
    frame_count = 0
    last_detections = []

    print("[INFO] Running. Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_count += 1

        # YOLO-World runs every frame (fast)
        results = yolo(frame, conf=0.1, verbose=False)

        # Extract detections from YOLO
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = result.names[cls_id]
                    info = CLASS_INFO.get(label, ("Unknown", (128, 128, 128)))
                    bin_type, color = info
                    detections.append({
                        "label": label, "conf": conf,
                        "bin": bin_type, "color": color,
                        "bbox": (x1, y1, x2, y2), "mask": None,
                    })

        # Merge same-class detections that overlap
        detections = merge_same_class(detections)

        # Run FastSAM periodically to get masks for detected objects
        if detections and frame_count % seg_interval == 0:
            try:
                fastsam_results = fastsam(frame, imgsz=640, conf=0.3, iou=0.7,
                                          retina_masks=True, verbose=False)
                if fastsam_results and len(fastsam_results) > 0:
                    fsam_result = fastsam_results[0]
                    if fsam_result.masks is not None:
                        h, w = frame.shape[:2]
                        all_masks = fsam_result.masks.data.cpu().numpy()

                        # Resize all masks once
                        resized_masks = []
                        for mask in all_masks:
                            if mask.shape != (h, w):
                                mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
                            resized_masks.append(mask.astype(np.uint8))

                        for det in detections:
                            bx1, by1, bx2, by2 = det["bbox"]
                            # Center point of YOLO detection
                            center_x = (bx1 + bx2) // 2
                            center_y = (by1 + by2) // 2

                            best_mask = None
                            best_area = float('inf')  # pick SMALLEST

                            for mask in resized_masks:
                                # Does this mask contain the YOLO center point?
                                if mask[center_y, center_x] == 0:
                                    continue

                                # Also check the mask covers a decent chunk of the YOLO bbox
                                bbox_region = mask[by1:by2, bx1:bx2]
                                bbox_coverage = bbox_region.sum() / max((by2-by1)*(bx2-bx1), 1)
                                
                                if bbox_coverage < 0.3:
                                    continue  # mask barely touches the detection

                                # Pick the smallest mask that contains the center
                                # (tightest fit around the object)
                                area = mask.sum()
                                if area < best_area:
                                    best_area = area
                                    best_mask = mask

                            if best_mask is not None:
                                det["mask"] = best_mask
                                # Expand bbox to cover the full mask
                                ys, xs = np.where(best_mask > 0)
                                det["bbox"] = (int(xs.min()), int(ys.min()),
                                               int(xs.max()), int(ys.max()))
            except Exception as e:
                print(f"[WARN] FastSAM: {e}")

            last_detections = detections
        elif detections:
            # Reuse last masks for non-segmentation frames
            for det in detections:
                for old in last_detections:
                    if det["label"] == old["label"] and old.get("mask") is not None:
                        det["mask"] = old["mask"]
                        break
        else:
            last_detections = []

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 0.001))
        prev_time = now

        display = draw_frame(frame, detections, fps)
        cv2.imshow("Recyclable Detector", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()