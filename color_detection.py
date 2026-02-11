import cv2
import json
import numpy as np

# --- KONFIGURASI MISI ---
# True = Tampilkan jendela video (Berat, pakai saat testing di Laptop)
# False = Headless mode (Ringan, pakai saat Raspi terbang)
SHOW_VIDEO = True 
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X_SCREEN = int(FRAME_WIDTH / 2)
CENTER_Y_SCREEN = int(FRAME_HEIGHT / 2)

# --- LOAD MODEL (Sesuaikan path json kamu) ---
# Pastikan file json ada, atau kode ini error.
# Kalau mau tes tanpa json, bisa hardcode array-nya manual.
MODEL_FILES = {
    "Blue": "models/blue_model.json",   # Aktifkan kalau butuh
    "Orange": "models/orange_model.json"
}

loaded_models = {}
for name, path in MODEL_FILES.items():
    try:
        with open(path, "r") as f:
            data = json.load(f)
            loaded_models[name] = {
                "lower": np.array(data["lower"]),
                "upper": np.array(data["upper"])
            }
        print(f"✅ {name} Loaded.")
    except:
        print(f"❌ {name} Error/Not Found.")

# --- KAMERA ---
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH) 
cap.set(4, FRAME_HEIGHT)

print("Mulai Deteksi... Tekan 'q' stop.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Blur & HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    detection_info = "No Target"

    for name, model in loaded_models.items():
        # Masking
        mask = cv2.inRange(hsv, model["lower"], model["upper"])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Cari Kontur
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Ambil yang terbesar (Target Utama)
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            
            if area > 1000: # Filter noise
                # DAPATKAN KOTAK PEMBUNGKUS
                x, y, w, h = cv2.boundingRect(c)
                
                # --- RUMUS MENCARI TITIK TENGAH (PENTING!) ---
                cx = int(x + (w / 2))
                cy = int(y + (h / 2))
                
                # Visualisasi Kotak
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Visualisasi Titik Tengah (Dot Merah)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Visualisasi Garis ke Tengah Layar (Optional, biar kayak sniper)
                cv2.line(frame, (CENTER_X_SCREEN, CENTER_Y_SCREEN), (cx, cy), (255, 0, 0), 1)

                # Simpan info untuk di-print
                detection_info = f"TARGET: {name} | Posisi: X={cx}, Y={cy}"

    # Tampilkan Text Koordinat
    cv2.putText(frame, detection_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Print ke terminal (Bakal dipakai logic drone nanti)
    # Gunakan \r biar nge-replace baris yang sama
    print(f"\r{detection_info:<50}", end="")

    if SHOW_VIDEO:
        cv2.imshow("Vision View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()