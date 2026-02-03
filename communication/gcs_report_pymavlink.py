from pymavlink import mavutil
import time
import sys

# --- KONFIGURASI RASPI ---
# Biasanya Pixhawk terbaca sebagai /dev/ttyACM0 di Raspi
CONNECTION_STRING = '/dev/ttyACM0'
BAUDRATE = 115200 # Baudrate standar koneksi USB

print(f"Connecting to Pixhawk at {CONNECTION_STRING}...")

try:
    # 1. Buka Koneksi ke Pixhawk
    # master = mavutil.mavlink_connection(CONNECTION_STRING, baud=BAUDRATE)
    master = mavutil.mavlink_connection(
    CONNECTION_STRING, 
    baud=BAUDRATE, 
    source_system=1,    # ID System sama dengan drone
    source_component=191 # ID Component Onboard Computer
)
    
    # 2. Tunggu Heartbeat (Wajib!)
    master.wait_heartbeat()
    print(f"✅ Connected to System {master.target_system}")

    # 3. Kirim Pesan Loop
    count = 0
    while True:
        count += 1
        msg_text = f"PYMAVLINK TEST: {count}"
        
        # Kirim ke Pixhawk. Pixhawk akan otomatis forward ke Telemetry Radio.
        # Severity 2 = Critical (Merah), 6 = Info (Kuning/Putih)
        severity = 2 if count % 5 == 0 else 6
        
        master.mav.statustext_send(severity, msg_text.encode())
        print(f"Sent to Pixhawk: {msg_text}")
        
        time.sleep(1) # Kirim tiap 1 detik

except Exception as e:
    print(f"ERROR: {e}")

except KeyboardInterrupt:
    print("Test Stopped.")