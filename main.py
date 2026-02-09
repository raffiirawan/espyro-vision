from ultralytics import YOLO
from pymavlink import mavutil
import time
import sys
import cv2

# ======= SETUP =======
# ----- Komunikasi -----
connection = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200, source_system=1, source_component=191)
connection.wait_heartbeat()

# ----- Vision -----
model = 'models/small_960.pt'

# ----- Servo Dropping -----
servoPort = 9
PWM_LOCK = 1000
PWM_DROP = 2000

# HEADLESS_MODE = Raspi only tanpa monitor
HEADLESS_MODE = True
# ==========================================

