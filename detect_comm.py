from ultralytics import YOLO
from pymavlink import mavutil
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import sys

# ================= K O N F I G U R A S I =================
# Sesuaikan dengan setup terbarumu (Pin TX/RX)
CONNECTION_STRING = '/dev/serial0' 
BAUDRATE = 57600

# Path model INT8 yang super ringan
MODEL_PATH = "models/terpal_416_int8.tflite" 
CONF_THRESHOLD = 0.60
CAMERA_INDEX = 0

HEADLESS_MODE = True # Wajib True kalau dijalankan tanpa monitor
# =========================================================

def main():
    print(">>> MEMULAI PROGRAM VISION LITE (DETEKSI & LAPOR) <<<")

    # --- 1. KONEKSI PIXHAWK ---
    print(f"[INIT] Menghubungkan ke {CONNECTION_STRING} pada baud {BAUDRATE}...")
    try:
        master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUDRATE, source_system=1, source_component=191)
        master.wait_heartbeat()
        print(f"[INIT] ✅ TERHUBUNG ke Pixhawk")
    except Exception as e:
        print(f"[ERROR] Gagal Konek Pixhawk: {e}")
        sys.exit()

    def send_telemetry(text, severity=6):
        """ Fungsi untuk mengirim teks ke layar Mission Planner """
        master.mav.statustext_send(severity, text.encode())
        print(f"[TELEM] {text}")

    # --- 2. PERSIAPAN VISION ---
    print(f"[INIT] Loading Model AI {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        send_telemetry("GAGAL LOAD MODEL AI!", 2)
        sys.exit()

    print("[INIT] Menyalakan Thread Kamera...")
    vs = VideoStream(src=CAMERA_INDEX, resolution=(640, 480)).start()
    time.sleep(2.0) # Tunggu sensor panas

    send_telemetry("Sistem Vision Siap! Memulai Pemindaian...", 6)

    # --- 3. LOOPING UTAMA ---
    last_report_time = 0  # Timer untuk mencegah spam pesan
    
    try:
        while True:
            # Baca frame dari kamera background
            frame = vs.read()
            if frame is None:
                continue

            # Jalankan deteksi YOLO (Resolusi 416, CPU)
            results = model(frame, conf=CONF_THRESHOLD, imgsz=416, device='cpu', verbose=False)

            # Jika ada objek yang terdeteksi
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                # Ambil kotak dengan nilai keyakinan (confidence) paling tinggi
                best_box_idx = np.argmax(boxes.conf.cpu().numpy())
                best_box = boxes[best_box_idx]
                
                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])
                class_name = model.names[cls_id] # "Terpal-Oren" atau "Terpal-Biru"

                # FITUR ANTI-SPAM: Kirim pesan maksimal 1 kali setiap 3 detik
                if time.time() - last_report_time > 3.0:
                    pesan = f"TARGET DETECTED: {class_name} ({conf:.2f})"
                    
                    # severity=2 akan membuat teks berwarna MERAH/KUNING di Mission Planner
                    send_telemetry(pesan, 2) 
                    
                    last_report_time = time.time()

            # (Opsional) Tampilkan gambar jika sedang tes di darat pakai monitor
            if not HEADLESS_MODE:
                annotated_frame = results[0].plot()
                cv2.imshow("Raspi View", annotated_frame)
                if cv2.waitKey(1) == ord('q'):
                    break
            
            # Istirahatkan CPU sedikit agar suhu terjaga
            time.sleep(0.05) 

    except KeyboardInterrupt:
        print("\nDihentikan Manual.")
    finally:
        vs.stop()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        print("Sistem Shutdown Mulus.")

if __name__ == "__main__":
    main()