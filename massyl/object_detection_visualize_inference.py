import cv2
import os
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "massyl/od_models/yolov8s_best_2.pt"
IMAGE_DIR = "massyl/data/lost_and_found_left_od_optimized/valid/images"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
IMG_SIZE = 640

# Load model
print("Loading model...")
model = YOLO(MODEL_PATH)

# Collect images
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".png"))
])

if not image_files:
    raise RuntimeError("No images found")

print("q = previous | d = next | esc = quit")

idx = 0
n = len(image_files)

while True:
    img_path = os.path.join(IMAGE_DIR, image_files[idx])
    frame = cv2.imread(img_path)

    if frame is None:
        idx = (idx + 1) % n
        continue

    # --- INFERENCE ---
    results = model.predict(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False
    )

    annotated = results[0].plot()
    inf_ms = results[0].speed["inference"]

    cv2.putText(
        annotated,
        f"{image_files[idx]} ({idx+1}/{n}) | {inf_ms:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Road Obstacle Detection", annotated)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):        # previous
        idx = (idx - 1) % n
    elif key == ord('d'):      # next
        idx = (idx + 1) % n
    elif key == 27:            # ESC
        break

cv2.destroyAllWindows()
