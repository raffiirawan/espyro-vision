from pymavlink import mavutil
import time

# --- SETUP KONEKSI ---
# === Connect SITL ===
print("Menghubungkan ke SITL...")
# connection = mavutil.mavlink_connection('udp:127.0.0.1:14550')
# connection = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
connection = mavutil.mavlink_connection('COM7', baud=115200)

# Tunggu Heartbeat
connection.wait_heartbeat()
print(f"Terhubung ke Pesawat {connection.target_system}")

# --- FUNGSI GERAK SERVO ---
def gerak_servo(nomor_servo, pwm):
    """
    nomor_servo: 9 (untuk AUX1)
    pwm: 1000 (Tutup) s/d 2000 (Buka)
    """

    # Perintah MAV_CMD_DO_SET_SERVO (Kode 183)
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, # Command ID
        0,              # Confirmation
        nomor_servo,    # Param 1: Nomor Servo
        pwm,            # Param 2: Nilai PWM
        0, 0, 0, 0, 0   # Param 3-7 (Kosong)
    )
    print(f"Mengirim Perintah: Servo {nomor_servo} -> PWM {pwm}")

# --- LOOPING BUKA-TUTUP ---
SERVO_DROPPING = 9

try:
    while True:
        # Gerakan 1: BUKA (DROP)
        gerak_servo(SERVO_DROPPING, 2000)
        time.sleep(3)

        gerak_servo(SERVO_DROPPING, 1000)
        time.sleep(3)

except KeyboardInterrupt:
    print("\nProgram berhenti")
