import cv2
import os
import numpy as np
import mediapipe as mp
from time import sleep

# --- Konfigurasi ---
DATA_PATH = r"G:\File Arya\NEW_TA_MB_LSTM_OKT\data\Kata"
# Anda bisa menambahkan lebih banyak aksi di sini
ACTIONS = ["Terimakasih"] 
NO_SEQUENCES = 30  # Sesuai proposal, 30 data per kelas
SEQUENCE_LENGTH = 30
CANVAS_SIZE = 600
PADDING_RATIO = 0.1

# --- Inisialisasi MediaPipe ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# --- Fungsi-Fungsi Inti (Tidak Berubah) ---
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, left_hand, right_hand])

def get_full_body_bbox(results, img_shape):
    xs = []
    ys = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            xs.append(int(lm.x * img_shape[1]))
            ys.append(int(lm.y * img_shape[0]))
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            xs.append(int(lm.x * img_shape[1]))
            ys.append(int(lm.y * img_shape[0]))
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            xs.append(int(lm.x * img_shape[1]))
            ys.append(int(lm.y * img_shape[0]))
    
    if not xs or not ys:
        return None
    
    min_x, min_y = max(0, min(xs)), max(0, min(ys))
    max_x, max_y = min(img_shape[1], max(xs)), min(img_shape[0], max(ys))
    
    pad_w = int((max_x - min_x) * PADDING_RATIO)
    pad_h = int((max_y - min_y) * PADDING_RATIO)
    min_x = max(0, min_x - pad_w)
    min_y = max(0, min_y - pad_h)
    max_x = min(img_shape[1], max_x + pad_w)
    max_y = min(img_shape[0], max_y + pad_h)
    
    return min_x, min_y, max_x, max_y

def create_canvas_crop(img, bbox):
    if bbox is None:
        return np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    min_x, min_y, max_x, max_y = bbox
    roi = img[min_y:max_y, min_x:max_x]
    
    if roi.size == 0:
        return canvas
    
    h, w = roi.shape[:2]
    scale = min((CANVAS_SIZE*0.9)/w, (CANVAS_SIZE*0.9)/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(roi, (new_w, new_h))
    
    x_offset = (CANVAS_SIZE - new_w)//2
    y_offset = (CANVAS_SIZE - new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# --- Fungsi UI (Baru) ---
def draw_ui(image, action, seq, frame_num, state):
    """Menampilkan informasi dan instruksi di layar."""
    # Warna berdasarkan state
    if state == 'PREPARING':
        color = (255, 255, 0) # Kuning
        status_text = f'BERSAPAN: {action}'
    elif state == 'RECORDING':
        color = (0, 0, 255) # Merah
        status_text = f'MEREKAM: {action}'
    else: # FINISHED
        color = (0, 255, 0) # Hijau
        status_text = f'SEQUENCE {seq} SELESAI!'

    cv2.putText(image, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(image, f'Sequence: {seq}/{NO_SEQUENCES}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    if state == 'RECORDING':
        cv2.putText(image, f'Frame: {frame_num}/{SEQUENCE_LENGTH}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        # Indikator rekaman (kotak merah di pojok)
        cv2.rectangle(image, (20, 140), (40, 160), (0, 0, 255), -1)

    cv2.putText(image, 'Tekan "r" untuk ulang sequence | "q" untuk keluar', (20, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)


# --- Program Utama ---
# 1. Cek kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

# 2. Buat folder dataset
for action in ACTIONS:
    for seq in range(NO_SEQUENCES):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)))
        except:
            pass

# 3. Mulai pengumpulan data
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        for seq in range(NO_SEQUENCES):
            # Loop untuk setiap sequence, bisa diulang jika ada kesalahan
            while True:
                # --- Tahap Persiapan (Countdown) ---
                for countdown in range(5, 0, -1):
                    ret, frame = cap.read()
                    if not ret: continue
                    
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    draw_ui(image, action, seq + 1, 0, 'PREPARING')
                    
                    cv2.putText(image, f'Mulai dalam: {countdown}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Collecting Data', image)
                    cv2.waitKey(1000)

                # --- Tahap Perekaman ---
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret: continue
                        
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    draw_ui(image, action, seq + 1, frame_num + 1, 'RECORDING')

                    # Simpan data
                    landmarks = extract_landmarks(results)
                    npy_path = os.path.join(DATA_PATH, action, str(seq), f'{frame_num}.npy')
                    np.save(npy_path, landmarks)

                    bbox = get_full_body_bbox(results, frame.shape)
                    canvas_crop = create_canvas_crop(frame, bbox)
                    jpg_path = os.path.join(DATA_PATH, action, str(seq), f'{frame_num}.jpg')
                    cv2.imwrite(jpg_path, canvas_crop)
                    
                    cv2.imshow('Collecting Data', image)

                    # Kontrol keyboard
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('r'):
                        print(f"Mengulang sequence {seq+1} untuk action '{action}'...")
                        break  # Keluar dari loop perekaman
                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()
                else:
                    # Ini dijalankan jika loop for selesai tanpa 'break' (tidak ada 'r' yang ditekan)
                    # --- Tampilan Selesai ---
                    ret, frame = cap.read()
                    if not ret: continue
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    draw_ui(image, action, seq + 1, 0, 'FINISHED')
                    cv2.imshow('Collecting Data', image)
                    cv2.waitKey(2000) # Tunggu 2 detik
                    break # Keluar dari while True, lanjut ke sequence berikutnya
                # Jika 'break' karena 'r' ditekan, while True akan berlanjut ke atas (countdown lagi)

cap.release()
cv2.destroyAllWindows()
print("\nData collection completed!")
print(f"Total actions: {len(ACTIONS)}")
print(f"Sequences per action: {NO_SEQUENCES}")
print(f"Frames per sequence: {SEQUENCE_LENGTH}")
print(f"Total frames collected: {len(ACTIONS) * NO_SEQUENCES * SEQUENCE_LENGTH}")