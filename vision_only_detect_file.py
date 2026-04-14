import os
import glob
from ultralytics import YOLO
import numpy as np
import cv2
import time
import sys

# ================= K O N F I G U R A S I =================
# Pastikan path model NCNN kamu benar
MODEL_PATH = "models/ncnn-yolo11_ncnn_model"

# Nama FOLDER tempat kamu menyimpan foto-foto asli dari lab
INPUT_FOLDER = "testing"

CONF_THRESHOLD = 0.60
# =========================================================

def main():
    print(">>> MEMULAI BATCH VISION (HEADLESS / PURE TERMINAL) <<<")
    print(f"[INFO] Model AI     : {MODEL_PATH}")
    print(f"[INFO] Folder Input : {INPUT_FOLDER}/\n")
    
    # --- 1. PERSIAPAN FOLDER ---
    if not os.path.exists(INPUT_FOLDER):
        print(f"[ERROR] Folder '{INPUT_FOLDER}' tidak ditemukan!")
        sys.exit()

    # Kumpulkan semua file gambar
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG'):
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
        
    if len(image_files) == 0:
        print(f"[WARNING] Kosong! Tidak ada file gambar di dalam folder '{INPUT_FOLDER}'.")
        sys.exit()
        
    print(f"[INFO] Ditemukan {len(image_files)} gambar. Memulai eksekusi...\n")
    print("-" * 55)

    # --- 2. LOAD MODEL AI ---
    try:
        # Load model NCNN
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"[ERROR] GAGAL LOAD MODEL! Error: {e}")
        sys.exit()

    # --- 3. LOOPING DETEKSI MASSAL ---
    total_waktu = 0
    target_ketemu = 0

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Menganalisa: {filename:<15} |", end=" ")

        # Baca file
        frame = cv2.imread(img_path)
        if frame is None:
            print("❌ GAGAL DIBACA")
            continue

        # Catat waktu
        start_time = time.time()
        
        # MIKIR: Deteksi YOLO (Tanpa render visual)
        results = model(frame, conf=CONF_THRESHOLD, imgsz=416, device='cpu', verbose=False)
        
        inference_time = (time.time() - start_time) * 1000 # milidetik
        total_waktu += inference_time

        # --- 4. LOGIKA TERMINAL ---
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_box_idx = np.argmax(boxes.conf.cpu().numpy())
            best_box = boxes[best_box_idx]

            conf = float(best_box.conf[0])
            
            print(f"✅ KETEMU (Akurasi: {conf*100:.1f}%) | Mikir: {inference_time:.1f} ms")
            target_ketemu += 1
        else:
            print(f"❌ TIDAK ADA             | Mikir: {inference_time:.1f} ms")

    # --- 5. KESIMPULAN ---
    print("-" * 55)
    print("[SELESAI] Rekapitulasi Ujian AI:")
    print(f"- Total Gambar     : {len(image_files)}")
    print(f"- Terpal Ditemukan : {target_ketemu}")
    
    rata_rata = total_waktu / len(image_files) if len(image_files) > 0 else 0
    print(f"- Rata-rata Mikir  : {rata_rata:.1f} ms per gambar")
    
    # Estimasi FPS murni tanpa beban I/O memori
    estimasi_fps = 1000 / rata_rata if rata_rata > 0 else 0
    print(f"- Estimasi FPS     : {estimasi_fps:.1f} FPS (Raw AI Power)")

if __name__ == "__main__":
    main()