from pymavlink import mavutil
import time
import sys

# === CONFIG ===
CONNECTION_PATH = 'udpin:127.0.0.1:14551'
TARGET_DROP = 4
SERVO_NUMBER = 9
PWM_LOCK = 1000
PWM_DROP = 1900

def set_servo(master, servo_num, pwm_value):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        servo_num,
        pwm_value,
        0, 0, 0, 0, 0
    )

def main():
    # BUAT KONEKSI
    print(">>> Dropping Mission <<<")
    print(f"Connecting to {CONNECTION_PATH} ...")
    master = mavutil.mavlink_connection(CONNECTION_PATH)

    # Waiting for Heartbeat
    print("Waiting for heartbeat from plane...")
    master.wait_heartbeat()
    print(f"Connected to {master.target_system}")

    set_servo(master, SERVO_NUMBER, PWM_LOCK)

    payload_dropped = False

    # LOOPING UTAMA
    try:
        while True:
            # Terima semua pesan yang masuk (Blocking biar gak spam CPU)
            msg = master.recv_match(blocking=True)

            # Cek kalau pesan kosong (biar aman)
            if not msg:
                continue

            if msg.get_type() == 'MISSION_ITEM_REACHED':
                wp_sekarang = msg.seq
                print(f"[EVENT] Sampai di waypoint {wp_sekarang}")

                if wp_sekarang == TARGET_DROP and not payload_dropped:
                    print(">>> Sampai di waypoint drop! DROPPING <<<")

                    # Buka servo
                    set_servo(master, SERVO_NUMBER, PWM_DROP)
                    payload_dropped = True
            
            elif msg.get_type() == 'MISSION_CURRENT':
                sys.stdout.write(f"\r[STATUS] OTW ke WP {msg.seq}...")
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nProgram Berhenti")

if __name__ == '__main__':
    main()