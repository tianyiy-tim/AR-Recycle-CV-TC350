"""
Camera & OpenCV Diagnostic Script
Run this to figure out what's going wrong.
"""
import cv2
import numpy as np
import time
import sys

print("=" * 60)
print("CAMERA & DISPLAY DIAGNOSTICS")
print("=" * 60)

# ── Test 1: OpenCV build info ──
print("\n[TEST 1] OpenCV version and backend")
print(f"  OpenCV version: {cv2.__version__}")
print(f"  Build info backends: ", end="")
# Check available backends
backends = []
for name in dir(cv2):
    if name.startswith("CAP_"):
        backends.append(name)
print(f"{len(backends)} backends available")

# ── Test 2: Camera open ──
print("\n[TEST 2] Opening camera...")
cap = cv2.VideoCapture(0)
print(f"  isOpened: {cap.isOpened()}")

if not cap.isOpened():
    print("  FAIL: Camera did not open. Trying AVFoundation backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    print(f"  AVFoundation isOpened: {cap.isOpened()}")
    if not cap.isOpened():
        print("  FAIL: Cannot open camera at all. Exiting.")
        sys.exit(1)

# ── Test 3: Camera properties ──
print("\n[TEST 3] Camera properties")
print(f"  Width:  {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"  Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"  FPS:    {cap.get(cv2.CAP_PROP_FPS)}")
print(f"  Backend: {cap.getBackendName()}")

# ── Test 4: Read frames ──
print("\n[TEST 4] Reading frames (10 attempts with delays)...")
time.sleep(1)

frames_ok = 0
frames_black = 0
frames_fail = 0
first_good_frame = None

for i in range(10):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  Frame {i}: FAILED (ret={ret})")
        frames_fail += 1
    else:
        mean_val = np.mean(frame)
        is_black = mean_val < 5
        status = "BLACK" if is_black else f"OK (mean brightness: {mean_val:.1f})"
        print(f"  Frame {i}: shape={frame.shape}, dtype={frame.dtype}, {status}")
        if is_black:
            frames_black += 1
        else:
            frames_ok += 1
            if first_good_frame is None:
                first_good_frame = frame.copy()
    time.sleep(0.3)

print(f"\n  Summary: {frames_ok} good, {frames_black} black, {frames_fail} failed")

# ── Test 5: Save a frame to disk ──
print("\n[TEST 5] Saving frame to disk...")
ret, frame = cap.read()
if ret and frame is not None:
    cv2.imwrite("debug_frame.jpg", frame)
    mean_val = np.mean(frame)
    print(f"  Saved debug_frame.jpg (mean brightness: {mean_val:.1f})")
    print(f"  If brightness is near 0, the camera is sending black frames.")
    print(f"  Open debug_frame.jpg to check visually.")
else:
    print("  FAILED to capture frame for saving.")

# ── Test 6: Display test ──
print("\n[TEST 6] Display test - showing color gradient (3 seconds)...")
test_img = np.zeros((400, 600, 3), dtype=np.uint8)
# Draw colored rectangles so we know the image isn't just black
cv2.rectangle(test_img, (0, 0), (200, 400), (255, 0, 0), -1)      # Blue
cv2.rectangle(test_img, (200, 0), (400, 400), (0, 255, 0), -1)    # Green
cv2.rectangle(test_img, (400, 0), (600, 400), (0, 0, 255), -1)    # Red
cv2.putText(test_img, "If you see this, display works!", (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

cv2.imshow("Display Test", test_img)
key = cv2.waitKey(3000)
cv2.destroyAllWindows()
print(f"  Window shown for 3 sec. Key pressed: {key}")

# ── Test 7: Live camera display ──
print("\n[TEST 7] Live camera feed (5 seconds)...")
print("  You should see your webcam. If black, the camera is the problem.")

start = time.time()
frame_count = 0
while time.time() - start < 5:
    ret, frame = cap.read()
    if ret and frame is not None:
        frame_count += 1
        mean_val = np.mean(frame)
        # Add debug text on the frame itself
        cv2.putText(frame, f"Frame {frame_count} | Brightness: {mean_val:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Live Camera Test", frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()
print(f"  Displayed {frame_count} frames in 5 seconds")

cap.release()

# ── Test 8: Check if frame data is actually black ──
print("\n[TEST 8] Analyzing saved frame...")
saved = cv2.imread("debug_frame.jpg")
if saved is not None:
    mean_b, mean_g, mean_r = [np.mean(saved[:, :, i]) for i in range(3)]
    print(f"  Channel means - B: {mean_b:.1f}, G: {mean_g:.1f}, R: {mean_r:.1f}")
    if mean_b < 5 and mean_g < 5 and mean_r < 5:
        print("  DIAGNOSIS: Camera is sending completely black frames.")
        print("  This is a macOS camera permission issue.")
        print("  Try: tccutil reset Camera")
        print("  Then re-run this script from Terminal.app")
    elif mean_b < 30 and mean_g < 30 and mean_r < 30:
        print("  DIAGNOSIS: Camera frames are very dark but not black.")
        print("  Your camera might need a moment to adjust exposure.")
        print("  Try covering/uncovering the lens.")
    else:
        print("  DIAGNOSIS: Frame has real image data!")
        print("  If the window was still black, it's a display/rendering issue.")
        print("  Try: pip install opencv-python==4.8.1.78")
else:
    print("  Could not read saved frame.")

print("\n" + "=" * 60)
print("DIAGNOSTICS COMPLETE")
print("=" * 60)