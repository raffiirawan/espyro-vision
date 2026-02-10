from pymavlink import mavutil
import sys
import time
import math

# --- KONFIGURASI KONEKSI ---
# Ingat, kita pakai port 14551 yang sudah kita "bor" tadi di SITL
connection_string = 'udpin:127.0.0.1:14551'

def main():
    print(f"Menghubungkan ke {connection_string} ...")
    
    # 1. Buat Koneksi
    try:
        master = mavutil.mavlink_connection(connection_string)
    except Exception as e:
        print(f"ERROR: Gagal konek. Pastikan SITL jalan! ({e})")
        sys.exit()

    # 2. Tunggu Detak Jantung (Wajib)
    print("Menunggu Heartbeat dari pesawat...")
    master.wait_heartbeat()
    print(f"✅ TERHUBUNG! System ID: {master.target_system}, Component: {master.target_component}")
    print("Mulai monitoring attitude... (Tekan Ctrl+C untuk berhenti)\n")

    # 3. LOOPING UTAMA
    try:
        while True:
            # Minta pesan 'ATTITUDE'
            # blocking=True artinya script akan PAUSE sampai data baru masuk
            msg = master.recv_match(type='ATTITUDE', blocking=True)
            
            if msg:
                # --- KONVERSI DATA ---
                # Data asli mavlink itu RADIAN (-3.14 sampe 3.14)
                # Kita ubah ke DERAJAT biar enak dibaca
                roll = math.degrees(msg.roll)
                pitch = math.degrees(msg.pitch)
                yaw = math.degrees(msg.yaw)

                # Koreksi Yaw biar range-nya 0-360 derajat (bukan minus)
                if yaw < 0:
                    yaw += 360

                # --- TAMPILAN KEREN ---
                # \r artinya 'Carriage Return', balik ke awal baris tanpa enter
                output = f"\r[TELEM] Roll: {roll:6.2f}° | Pitch: {pitch:6.2f}° | Yaw: {yaw:6.2f}°   "
                
                sys.stdout.write(output)
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nMonitoring berhenti.")

if __name__ == '__main__':
    main()