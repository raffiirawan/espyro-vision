from ultralytics import YOLO
from pymavlink import mavutil
from imutils.video import VideoStream
import numpy as np
import cv2
import time
import sys
import math

# ================= K O N F I G U R A S I =================

# --- 1. KOMUNIKASI (Hardware) ---
CONNECTION_STRING = '/dev/ttyACM0' # Ganti ke /dev/ttyAMA0 jika pakai pin GPIO
BAUDRATE = 115200
SERVO_PORT = 9      # AUX 1 di Pixhawk
PWM_LOCK = 1000     # Posisi Mengunci
PWM_DROP = 2000     # Posisi Melepas (Drop)

# --- 2. VISION (AI TFLite) ---
# Gunakan model hasil export INT8 untuk performa maksimal di CPU Raspi
MODEL_PATH = "models/terpal_int8_416.tflite" 
CONF_THRESHOLD = 0.60
CAMERA_INDEX = 0

# --- 3. NAVIGASI (Attack Run) ---
JARAK_MUNDUR = 250  # Jarak ancang-ancang agar sayap stabil (meter)
TARGET_ALTITUDE = 0 # (Akan dioverride otomatis ke ketinggian pesawat saat deteksi)

HEADLESS_MODE = True # True = Tanpa tampilan UI (Wajib untuk flight)

# =========================================================
# FUNGSI BANTUAN MATEMATIKA (HAVERSINE)
# =========================================================

def get_gps_offset(lat, lon, distance_m, bearing_deg):
    R = 6378137.0 
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    bearing_rad = math.radians(bearing_deg)
    lat_baru = math.asin(math.sin(lat_rad) * math.cos(distance_m/R) + 
                         math.cos(lat_rad) * math.sin(distance_m/R) * math.cos(bearing_rad))
    lon_baru = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance_m/R) * math.cos(lat_rad), 
                                    math.cos(distance_m/R) - math.sin(lat_rad) * math.sin(lat_baru))
    return math.degrees(lat_baru), math.degrees(lon_baru)

def get_distance_m(lat1, lon1, lat2, lon2):
    R = 6378137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# =========================================================
# PROGRAM UTAMA
# =========================================================

def main():
    print(">>> MEMULAI MISI SMART DROPPING KRTI <<<")

    # --- KONEKSI PIXHAWK ---
    print(f"[INIT] Menghubungkan ke {CONNECTION_STRING}...")
    try:
        master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUDRATE, source_system=1, source_component=191)
        master.wait_heartbeat()
        print(f"[INIT] ✅ TERHUBUNG ke Pixhawk")
    except Exception as e:
        print(f"[ERROR] Gagal Konek Pixhawk: {e}")
        sys.exit()

    def send_telemetry(text, severity=6):
        master.mav.statustext_send(severity, text.encode())
        print(f"[TELEM] {text}")

    def set_servo(pwm):
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0, SERVO_PORT, pwm, 0, 0, 0, 0, 0
        )

    def go_to_waypoint(lat, lon, alt):
        master.mav.set_position_target_global_int_send(
            0, master.target_system, master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_INT, int(0b110111111000), 
            int(lat * 1e7), int(lon * 1e7), alt, 0, 0, 0, 0, 0, 0, 0, 0
        )

    # --- PERSIAPAN AWAL ---
    set_servo(PWM_LOCK)
    send_telemetry("Sistem Siap. Payload TERKUNCI.", 6)

    print(f"[INIT] Loading Model AI {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        send_telemetry("GAGAL LOAD MODEL AI!", 2)
        sys.exit()

    print("[INIT] Menyalakan Thread Kamera...")
    vs = VideoStream(src=CAMERA_INDEX, resolution=(640, 480)).start()
    time.sleep(2.0) # Tunggu sensor panas

    # Tunggu Lock GPS sebelum mulai
    print("[INIT] Menunggu sinyal GPS...")
    current_lat, current_lon, current_hdg, current_alt = 0, 0, 0, 0
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg and msg.lat != 0:
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            current_hdg = msg.hdg / 100
            current_alt = msg.relative_alt / 1000.0 # Ketinggian relatif dlm meter
            send_telemetry("GPS Valid. Memulai Patroli Vision.", 6)
            break

    # --- VARIABEL STATE MACHINE ---
    # 0 = Patroli (AUTO)
    # 1 = Loiter & Voting (GUIDED)
    # 2 = Breakout/Reposition (GUIDED)
    # 3 = Attack Run (GUIDED)
    state = 0 
    
    suara_oren = 0
    suara_biru = 0
    waktu_mulai_vote = 0
    
    suspect_lat = 0
    suspect_lon = 0
    suspect_alt = 0
    heading_misi_awal = 0
    
    titik_putar_lat = 0
    titik_putar_lon = 0
    jarak_terdekat = 99999

    # ================= LOOPING JANTUNG UTAMA =================
    try:
        while True:
            # 1. BACA SENSOR GPS TERBARU (Kuras Antrian Pesan)
            msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
            while msg:
                current_lat = msg.lat / 1e7
                current_lon = msg.lon / 1e7
                current_hdg = msg.hdg / 100
                current_alt = msg.relative_alt / 1000.0
                msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)

            # 2. BACA KAMERA BACKGROUND
            frame = vs.read()
            if frame is None:
                continue

            # ================= FASE 0: PATROLI =================
            if state == 0:
                # YOLO Inference (Resolusi 416)
                results = model(frame, conf=CONF_THRESHOLD, imgsz=416, device='cpu', verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Apapun warnanya, curigai dulu dan trigger Loiter!
                    suspect_lat = current_lat
                    suspect_lon = current_lon
                    suspect_alt = current_alt # Tahan di ketinggian terbang saat ini
                    heading_misi_awal = current_hdg
                    
                    send_telemetry("TERPAL TERDETEKSI! Memulai Loiter...", 2)
                    master.mav.set_mode_send(master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 15) # GUIDED
                    go_to_waypoint(suspect_lat, suspect_lon, suspect_alt)
                    
                    state = 1
                    waktu_mulai_vote = time.time()
                    suara_oren = 0
                    suara_biru = 0

            # ================= FASE 1: VOTING (SAAT LOITER) =================
            elif state == 1:
                results = model(frame, conf=CONF_THRESHOLD, imgsz=416, device='cpu', verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    best_box = boxes[np.argmax(boxes.conf.cpu().numpy())]
                    class_name = model.names[int(best_box.cls[0])]
                    
                    if "oren" in class_name.lower():
                        suara_oren += 1
                    else:
                        suara_biru += 1

                # Waktu loiter habis (misal 25 detik)
                if time.time() - waktu_mulai_vote > 25.0:
                    if suara_oren > suara_biru:
                        send_telemetry(f"TARGET VALID (OREN). Breakout!", 2)
                        
                        # Kalkulasi mundur sejajar rute awal
                        heading_mundur = (heading_misi_awal + 180) % 360
                        titik_putar_lat, titik_putar_lon = get_gps_offset(suspect_lat, suspect_lon, JARAK_MUNDUR, heading_mundur)
                        
                        go_to_waypoint(titik_putar_lat, titik_putar_lon, suspect_alt)
                        state = 2
                    else:
                        send_telemetry("TARGET SIPIL (BIRU). Mengabaikan...", 6)
                        master.mav.set_mode_send(master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 10) # AUTO
                        time.sleep(5) # Jeda agar tidak langsung re-detect objek yang sama
                        state = 0

            # ================= FASE 2: BREAKOUT (MENUJU ANCANG-ANCANG) =================
            elif state == 2:
                # YOLO dimatikan sementara agar CPU fokus ke Navigasi
                jarak = get_distance_m(current_lat, current_lon, titik_putar_lat, titik_putar_lon)
                
                if jarak < 30:
                    send_telemetry("Titik Serang Tercapai. MELUNCUR!", 2)
                    go_to_waypoint(suspect_lat, suspect_lon, suspect_alt)
                    jarak_terdekat = 99999
                    state = 3
                
                time.sleep(0.05) # Throttle loop agar CPU Raspi tidak meledak

            # ================= FASE 3: ATTACK RUN (DROP PAYLOAD) =================
            elif state == 3:
                jarak = get_distance_m(current_lat, current_lon, suspect_lat, suspect_lon)
                
                if jarak < 10 or (jarak < 40 and jarak > jarak_terdekat):
                    send_telemetry("BOMBS AWAY! Payload Dropped!", 2)
                    set_servo(PWM_DROP)
                    
                    # RTB / Lanjut Misi
                    master.mav.set_mode_send(master.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 10) # AUTO
                    time.sleep(2) # Biarkan MAVLink memproses
                    send_telemetry("Sistem Vision Selesai. Lanjut AUTO.", 6)
                    break # Keluar dari loop utama
                
                jarak_terdekat = jarak
                time.sleep(0.05)

            # Debug Visualisasi
            if not HEADLESS_MODE and state in [0, 1]:
                cv2.imshow("Raspi View", frame)
                if cv2.waitKey(1) == ord('q'): break

    except KeyboardInterrupt:
        print("\nDihentikan Manual.")
    finally:
        set_servo(PWM_LOCK)
        vs.stop()
        cv2.destroyAllWindows()
        print("Sistem Shutdown Mulus.")

if __name__ == "__main__":
    main()