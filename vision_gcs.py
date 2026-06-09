import sys
import ai_edge_litert
# TRIK ILUSI: Membohongi YOLO agar membaca LiteRT baru sebagai tflite_runtime lawas
sys.modules['tflite_runtime'] = ai_edge_litert

from ultralytics import YOLO
from pymavlink import mavutil
from picamera2 import Picamera2
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

# Konfigurasi Camera Module 3 (IMX708) Raspi 5
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FPS_CAMERA = 30
# =========================================================

# Fungsi pembaca sensor suhu hardware Raspi
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
    # Fungsi utk controll servo dropping
    print(f"Mengirim perintah buka servo {servo_no} dengan PWM {pwm_value}")

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,          # Confirmation
        servo_no,   # Param 1: Nomor pin servo
        pwm_value,  # Param 2: Target nilai PWM
        0, 0, 0, 0, 0   # Param 3-7: Tidak digunakan
    )

def main():
    print(">>> MEMULAI VISION VIA MAVLINK-ROUTER (RASPI 5 + CAM V3) <<<")

    # --- 1. KONEKSI KE MAVLINK-ROUTER ---
    print(f"[INIT] Menghubungkan ke Pintu Virtual {CONNECTION_STRING}...")
    try:
        # ID komponen 191 agar GCS tahu ini pesan dari Companion Computer
        master = mavutil.mavlink_connection(CONNECTION_STRING, source_system=1, source_component=191)

        # MAVLink via TCP/UDP butuh "pancingan" detak jantung
        master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        master.wait_heartbeat(timeout=5)
        print(f"[INIT] TERHUBUNG ke Jaringan MAVLink!")
    except Exception as e:
        print(f"[ERROR] Gagal Konek MAVLink: {e}")
        sys.exit(1)

    def send_telemetry(text, severity=6):
        """ Fungsi untuk mengirim teks ke layar Mission Planner """
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
        print(f"[ERROR] GAGAL LOAD MODEL AI! Pastikan file ada di folder models/. Error: {e}")
        sys.exit(1)

    # TRIK WARM-UP AI (PEMANASAN CPU)
    print("[INIT] Memanaskan Otak AI (Tunggu 10-60 detik)...")
    try:
        dummy_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        model.predict(dummy_frame, imgsz=IMG_SIZE, device="cpu", verbose=False)
        print("[INIT] AI Selesai Pemanasan.")
    except Exception as e:
        print(f"[ERROR] Pemanasan AI Gagal: {e}")

    # --- 3. INIT CAMERA MODULE 3 ---
    print("[INIT] Menyalakan Camera Module 3...")
    try:
        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": FPS_CAMERA}
        )
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(2.0) # Tunggu sensor kamera stabil

        # Aktifkan Autofocus
        try:
            from libcamera import controls
            picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        except Exception as e:
            pass
            
    except Exception as e:
        print(f"[ERROR] Gagal membuka kamera: {e}")
        sys.exit(1)

    send_telemetry("Mata Pesawat (Vision) AKTIF & TERKONEKSI!", 6)

    # --- 4. LOOPING UTAMA ---
    last_report_time = 0
    frame_count = 0 
    batch_start_time = time.time()
    
    # Variabel kontrol suhu
    last_temp_time = 0      
    TEMP_INTERVAL = 30.0    # Cek dan lapor suhu setiap 30 detik
    
    # Pancing sensor psutil di awal
    psutil.cpu_percent()

    try:
        while True:
            # Baca frame dari Camera Module 3 (Direct RAM Access)
            frame = picam2.capture_array()
            
            if frame is None:
                print("[WARN] Frame kosong! Mengecek ulang kamera...")
                time.sleep(1)
                continue

            # Deteksi YOLO (Resolusi 416, CPU)
            results = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, device="cpu", verbose=False)

            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                # Cari deteksi dengan keyakinan tertinggi
                best_box_idx = int(np.argmax(boxes.conf.cpu().numpy()))
                best_box = boxes[best_box_idx]

                conf = float(best_box.conf[0])
                cls_id = int(best_box.cls[0])
                class_name = model.names[cls_id]

                # FITUR ANTI-SPAM (1 pesan per 3 detik via GSM)
                if time.time() - last_report_time > 3.0:
                    pesan = f"TARGET DETECTED: {class_name} ({conf:.2f})"
                    send_telemetry(pesan, 2) # Severity 2 = Merah/Kuning di MP

                    # Logic filtering deteksi
                    if class_name == "Terpal-Orange" or class_name == "terpal":
                        pesan = f"{class_name}: ({conf:.2f}), sikat drop"
                        send_telemetry(pesan, 2)
                        control_servo(master, 7, 1900)
                    
                    elif class_name == "Terpal-Biru":
                        pesan = f"{class_name}: ({conf:.2f}), jangan drop"
                        send_telemetry(pesan, 2)
                    
                    else:
                        print(f"Objek lain terdeteksi: {class_name}, yawes")
                    
                    last_report_time = time.time()

            # =======================================================
            # FITUR MONITORING PERFORMA & SUHU (Terminal & GCS)
            # =======================================================
            frame_count += 1
            if frame_count >= 30:
                # 1. Kalkulasi FPS, CPU, RAM (Untuk Terminal)
                elapsed_time = time.time() - batch_start_time
                fps = frame_count / elapsed_time
                cpu_usage = psutil.cpu_percent()
                ram_percent = psutil.virtual_memory().percent
                
                print(f"[BENCHMARK] {fps:.2f} FPS | CPU: {cpu_usage:04.1f}% | RAM: {ram_percent}%")
                
                frame_count = 0
                batch_start_time = time.time()

            # 2. Pengiriman Suhu ke Mission Planner
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
            # =======================================================

            # Istirahatkan CPU sedikit agar suhu terjaga (0.01 detik agar FPS maksimal)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan Manual oleh User.")
    finally:
        print("[INFO] Membersihkan resource hardware...")
        picam2.stop()
        print("[INFO] Sistem Shutdown Mulus.")

if __name__ == "__main__":
    main()