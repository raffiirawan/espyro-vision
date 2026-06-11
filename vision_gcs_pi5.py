import sys
import ai_edge_litert
# TRIK ILUSI: Membohongi YOLO agar membaca LiteRT baru sebagai tflite_runtime lawas
sys.modules['tflite_runtime'] = ai_edge_litert

from ultralytics import YOLO
from pymavlink import mavutil
from picamera2 import Picamera2
import cv2
import numpy as np
import time
import psutil

# ================= K O N F I G U R A S I =================
# Koneksi via TCP lokal ke MAVLink-Router
CONNECTION_STRING = 'tcp:127.0.0.1:5760'

# Path model yolo11s tflite
MODEL_PATH = "models/yolo11n_416_int8_update.tflite"

CONF_THRESHOLD = 0.60
IMG_SIZE = 416

# 🛠️ SWITCH UTAMA OPSI JALUR KAMERA (PILIH SALAH SATU) 🛠️
# OPSI 1: Matikan line di bawah ini (beri #) untuk terbang murni Picamera2 (Direct RAM, No GStreamer)
# OPSI 2: Aktifkan line di bawah ini (hilangkan #) jika ingin streaming ke GCS HUD & VLC via GStreamer
# CAMERA_INDEX = "/dev/video99"

# Parameter dasar kamera
FRAME_WIDTH = 640
FPS_CAMERA = 30
# =========================================================

# --- LOGIKA OTOMATIS DETECTION SWITCH ---
# Menentukan opsi terbang berdasarkan status CAMERA_INDEX di atas
if 'CAMERA_INDEX' in globals():
    USE_GSTREAMER = True
    FRAME_HEIGHT = 480  # Otomatis 4:3 untuk GStreamer Virtual Cam
else:
    USE_GSTREAMER = False
    FRAME_HEIGHT = 360  # Otomatis 16:9 untuk Picamera2 Native
# =========================================================

def get_cpu_temperature():
    """Membaca suhu CPU Raspi langsung dari sensor hardware Linux"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.read()
        return float(temp_str) / 1000.0
    except Exception as e:
        print(f"[ERROR] Gagal membaca suhu: {e}")
        return 0.0

def control_servo(master, servo_no, pwm_value):
    print(f"Mengirim perintah buka servo {servo_no} dengan PWM {pwm_value}")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
        servo_no, pwm_value, 0, 0, 0, 0, 0
    )

def main():
    if USE_GSTREAMER:
        print(">>> [MODE FLIGHT] OPSI 2: STREAMING ACTIVE (GSTREAMER + GCS HUD) <<<")
    else:
        print(">>> [MODE FLIGHT] OPSI 1: PURE VISION ACTIVE (DIRECT PICAMERA2) <<<")

    # --- 1. KONEKSI KE MAVLINK-ROUTER ---
    print(f"[INIT] Menghubungkan ke Pintu Virtual {CONNECTION_STRING}...")
    try:
        master = mavutil.mavlink_connection(CONNECTION_STRING, source_system=1, source_component=191)
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        master.wait_heartbeat(timeout=5)
        print(f"[INIT] TERHUBUNG ke Jaringan MAVLink!")
    except Exception as e:
        print(f"[ERROR] Gagal Konek MAVLink: {e}")
        sys.exit(1)

    def send_telemetry(text, severity=6):
        try:
            master.mav.statustext_send(severity, text.encode())
            print(f"[TELEM] -> {text}")
        except Exception as e:
            print(f"[ERROR] Gagal kirim telemetri: {e}")

    # --- 2. PERSIAPAN VISION & WARM-UP AI ---
    print(f"[INIT] Loading Model AI {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task="detect")
        print(f"Daftar Class di model AI ini: {model.names}")
    except Exception as e:
        print(f"[ERROR] GAGAL LOAD MODEL AI! Error: {e}")
        sys.exit(1)

    print("[INIT] Memanaskan Otak AI...")
    try:
        dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        model.predict(dummy_frame, imgsz=IMG_SIZE, device="cpu", verbose=False)
        print("[INIT] AI Selesai Pemanasan.")
    except Exception as e:
        print(f"[ERROR] Pemanasan AI Gagal: {e}")

    # --- 3. INISIALISASI KAMERA DINAMIS OBJEK ---
    picam2 = None
    vs = None

    if USE_GSTREAMER:
        print(f"[INIT] Membuka Virtual Camera {CAMERA_INDEX}...")
        vs = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not vs.isOpened():
            print(f"[ERROR] Gagal membuka {CAMERA_INDEX}! Jalankan GStreamer di terminal sebelah.")
            sys.exit(1)
        time.sleep(1.0)
    else:
        print("[INIT] Menyalakan Camera Module 3 via Picamera2...")
        try:
            picam2 = Picamera2()
            camera_config = picam2.create_video_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": FPS_CAMERA}
            )
            picam2.configure(camera_config)
            picam2.start()
            time.sleep(2.0)

            try:
                from libcamera import controls
                picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                print("[INIT] Autofocus Continuous aktif.")
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] Gagal membuka Picamera2: {e}")
            sys.exit(1)

    send_telemetry("Mata Pesawat (Vision) AKTIF & TERKONEKSI!", 6)

    # --- 4. LOOPING UTAMA ---
    last_report_time = 0
    frame_count = 0 
    batch_start_time = time.time()
    last_temp_time = 0      
    TEMP_INTERVAL = 30.0    
    psutil.cpu_percent()

    try:
        while True:
            # Penangkapan Frame Dinamis sesuai Opsi yang Aktif
            if USE_GSTREAMER:
                ret, frame = vs.read()
                if not ret or frame is None:
                    print("[WARN] Frame GStreamer kosong! Menunggu suplai video...")
                    time.sleep(1)
                    continue
            else:
                frame = picam2.capture_array()
                if frame is None:
                    print("[WARN] Frame Picamera2 kosong! Mengecek kamera...")
                    time.sleep(1)
                    continue

            # Deteksi YOLO
            results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, device="cpu", verbose=False)

            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_box_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                best_box = boxes[best_box_idx]

                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])
                class_name = model.names[cls_id]

                if time.time() - last_report_time > 3.0:
                    pesan = f"TARGET DETECTED: {class_name} ({conf:.2f})"
                    send_telemetry(pesan, 2)

                    if class_name == "Terpal-Orange" or class_name == "terpal":
                        pesan = f"{class_name}: ({conf:.2f}), sikat drop"
                        send_telemetry(pesan, 2)
                        control_servo(master, 7, 1900)
                    
                    elif class_name == "Terpal-Biru":
                        pesan = f"{class_name}: ({conf:.2f}), jangan drop"
                        send_telemetry(pesan, 2)
                    
                    last_report_time = time.time()

            # PERFORMA BENCHMARK & MONITORING SUHU
            frame_count += 1
            if frame_count >= 30:
                elapsed_time = time.time() - batch_start_time
                fps = frame_count / elapsed_time
                cpu_usage = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                print(f"[BENCHMARK] {fps:.2f} FPS | CPU: {cpu_usage:04.1f}% | RAM: {ram_percent}%")
                
                frame_count = 0
                batch_start_time = time.time()

            if time.time() - last_temp_time > TEMP_INTERVAL:
                suhu = get_cpu_temperature()
                print(f"[HW STATS] Suhu CPU Raspi: {suhu:.1f} °C")
                
                if suhu >= 80.0:
                    send_telemetry(f"CRITICAL! RASPI OVERHEAT: {suhu:.1f}C", 2)
                elif suhu >= 70.0:
                    send_telemetry(f"WARNING! Raspi Hot: {suhu:.1f}C", 4)
                else:
                    send_telemetry(f"Sys Temp: {suhu:.1f}C", 6)
                
                last_temp_time = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan Manual oleh User.")
    finally:
        print("[INFO] Membersihkan resource hardware...")
        # Penutupan Resource Dinamis demi Mulusnya Sistem Shutdown
        if picam2 is not None:
            picam2.stop()
        if vs is not None and vs.isOpened():
            vs.release()
        print("[INFO] Sistem Shutdown Mulus.")

if __name__ == "__main__":
    main()