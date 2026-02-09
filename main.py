from ultralytics import YOLO
from pymavlink import mavutil
import cv2
import time
import sys

# ================= K O N F I G U R A S I =================

# --- 1. KOMUNIKASI (Sesuai Tes Sukses Kamu) ---
# Di Raspi biasanya '/dev/ttyACM0'. Kalau gagal, coba '/dev/ttyAMA0'
CONNECTION_STRING = '/dev/ttyACM0' 
BAUDRATE = 115200

# --- 2. VISION (Mata) ---
MODEL_PATH = "models/small_960.pt"  # Pastikan file ada di folder ini
CONF_THRESHOLD = 0.60               # Ambang keyakinan (60%)
CAMERA_INDEX = 0                    # 0 = Kamera USB/Webcam utama

# --- 3. SERVO (Tangan) ---
SERVO_PORT = 9      # AUX 1 di Pixhawk
PWM_LOCK = 1000     # Posisi Mengunci
PWM_DROP = 2000     # Posisi Melepas (Drop)

# --- 4. SYSTEM (Penting!) ---
# Set True agar Raspi tidak error mencari monitor (Headless)
HEADLESS_MODE = True 

# =========================================================

def main():
    print(">>> MEMULAI MISI SMART DROPPING... <<<")

    # --- STEP 1: KONEKSI KE PIXHAWK ---
    print(f"[INIT] Menghubungkan ke {CONNECTION_STRING}...")
    try:
        # Kita menyamar jadi Component 191 (Onboard Computer) biar akrab sama Pixhawk
        master = mavutil.mavlink_connection(
            CONNECTION_STRING, 
            baud=BAUDRATE,
            source_system=1,
            source_component=191
        )
        master.wait_heartbeat()
        print(f"[INIT] ✅ TERHUBUNG ke System {master.target_system}")
    except Exception as e:
        print(f"[ERROR] Gagal Konek: {e}")
        sys.exit()

    # Fungsi kirim pesan ke Mission Planner
    def send_telemetry(text, severity=6):
        # Severity: 2=Critical (Merah), 6=Info (Putih)
        master.mav.statustext_send(severity, text.encode())
        print(f"[TELEM] {text}")

    # Fungsi gerak servo
    def set_servo(pwm):
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
            SERVO_PORT, pwm, 0, 0, 0, 0, 0
        )

    # --- STEP 2: PERSIAPAN AWAL ---
    # Kunci payload dulu biar aman
    set_servo(PWM_LOCK)
    send_telemetry("Misi Dimulai. Payload TERKUNCI.", 6)

    # Load Model YOLO
    print(f"[INIT] Loading Model {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except:
        send_telemetry("GAGAL LOAD MODEL! Cek path file.", 2)
        sys.exit()

    # Buka Kamera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        send_telemetry("ERROR: Kamera tidak terdeteksi!", 2)
        sys.exit()

    # --- STEP 3: LOOPING UTAMA (Jantung Misi) ---
    payload_dropped = False
    last_report = time.time()

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Frame error")
                continue

            # 1. INFERENCE (Mendeteksi)
            # verbose=False biar log bersih, device='cpu' biar aman di Raspi
            results = model(frame, conf=CONF_THRESHOLD, imgsz=960, device='cpu', verbose=False)
            
            target_detected = False
            current_conf = 0.0

            # 2. ANALISIS HASIL
            for r in results:
                for box in r.boxes:
                    current_conf = float(box.conf[0])
                    
                    # Jika yakin > threshold
                    if current_conf > CONF_THRESHOLD:
                        target_detected = True
                        
                        # LOGIKA DROP: Hanya jika belum pernah drop
                        if not payload_dropped:
                            msg = f"TARGET LOCKED ({current_conf:.2f})! DROPPING NOW!"
                            send_telemetry(msg, 2) # Pesan MERAH di MP
                            
                            set_servo(PWM_DROP)    # Buka Pengait
                            payload_dropped = True # Tandai selesai
                        
                        break # Cukup 1 target

            # 3. PELAPORAN BERKALA (Agar kita tau sistem hidup)
            # Lapor setiap 3 detik
            if time.time() - last_report > 3.0:
                if payload_dropped:
                    send_telemetry("Misi Selesai. RTB.", 6)
                elif target_detected:
                    # Kalau terdeteksi tapi sudah drop, lapor saja
                    pass 
                else:
                    send_telemetry("Scanning Area...", 6)
                
                last_report = time.time()

            # 4. VISUALISASI DEBUG (Opsional)
            # Hanya jalan kalau HEADLESS_MODE = False
            if not HEADLESS_MODE:
                annotated = results[0].plot()
                small_view = cv2.resize(annotated, (640, 480))
                cv2.imshow("Raspi Vision", small_view)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nBerhenti Manual.")
        
    finally:
        # Cleanup saat mati
        set_servo(PWM_LOCK) # Kunci lagi biar aman (opsional)
        cap.release()
        cv2.destroyAllWindows()
        print("Program Selesai.")

if __name__ == "__main__":
    main()