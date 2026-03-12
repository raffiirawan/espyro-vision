import cv2
import numpy as np
import time
from ultralytics import YOLO

# True = tampilkan jendela video opencv
# False = headless mode (utk raspi)
SHOW_VIDEO = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X_SCREEN = int(FRAME_WIDTH / 2)
CENTER_Y_SCREEN = int(FRAME_HEIGHT / 2)

# === Load Model ===
MODEL_PATH = "models/best_fatur_float16.tflite"

print(f"Loading model {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded")
except Exception as e:
    print("Failed to load model: {e}")
    exit()

# === Kamera / Video ===
# SOURCE = "hasil_testing_model/TRAINING_OREN.mp4"     #0 = pake camera, bisa dganti path video testing
SOURCE = 0
cap = cv2.VideoCapture(SOURCE)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

print("Detection started... Press 'q' to stop")

prev_time = 0   #Variable utk menghitung kecepatan proses

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n [INFO] Video habis atau kamera terputus")
        break
    
    detection_info = "No Target"

    # 1. YOLO INFERENCE
    results = model(frame, conf=0.5, imgsz=960, verbose=False)

    # 2. Ekstraksi Data Target
    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        best_box = boxes[np.argmax(boxes.conf.cpu().numpy())]

        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        conf = float(best_box.conf[0])

        # Hitung lebar, tinggi, dan titik tengah (Centroid)
        w = x2 - x1
        h = y2 - y1
        cx = int(x1 + (w / 2))
        cy = int(y1 + (h / 2))

        # --- VISUALISASI (Hanya dieksekusi kalau SHOW_VIDEO = True) ---
        if SHOW_VIDEO:
            # Visualisasi Kotak
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Visualisasi Titik Tengah (Dot Merah)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            # Garis sniper
            cv2.line(frame, (CENTER_X_SCREEN, CENTER_Y_SCREEN), (cx, cy), (255, 0, 0), 1)
        
        # Simpan info untuk di-print
        detection_info = f"TARGET LOCKED | Posisi: X={cx}, Y={cy} | Conf: {conf:.2f}"

    # 3. Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # 4. Tampilan & Logging
    if SHOW_VIDEO:
        cv2.putText(frame, detection_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Vision View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print ke terminal
    print(f"\r[FPS: {fps:4.1f}] {detection_info:<55}", end="")

cap.release()
cv2.destroyAllWindows()