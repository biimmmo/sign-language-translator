import cv2
import os
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import time
from gtts import gTTS          # Library TTS baru
from playsound import playsound # Library untuk memutar suara

# ==============================
# KONFIGURASI PATH
# ==============================
BASE_DIR = r"G:\File Arya\New_TA_MB_LSTM_OKT"
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model.h5")
LABEL_CLASSES_PATH = os.path.join(BASE_DIR, "models", "label_classes.npy")

# ==============================
# LOAD MODEL & LABELS
# ==============================
# Load model LSTM yang sudah dilatih
model = load_model(MODEL_PATH)
print(f"[INFO] Model loaded from {MODEL_PATH}")

# --- TAMBAHKAN BARIS INI UNTUK VERIFIKASI ---
print("\n--- MODEL SUMMARY ---")
model.summary()
print("-----------------------\n")

# Load model MobileNetV2 untuk ekstraksi fitur (harus sama dengan saat preprocessing)
mobilenet_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
print("[INFO] MobileNetV2 feature extractor loaded.")

# Load label kelas secara dinamis
actions = np.load(LABEL_CLASSES_PATH)
print(f"[INFO] Labels loaded: {actions}")

# ==============================
# SETUP TEXT-TO-SPEECH (gTTS)
# ==============================
# Nama file sementara untuk audio
TEMP_AUDIO_FILE = "temp_prediction.mp3"
TTS_ENABLED = True

# Cek apakah library gTTS berfungsi (memerlukan koneksi internet)
try:
    tts_test = gTTS(text='test', lang='id')
    tts_test.save(TEMP_AUDIO_FILE)
    os.remove(TEMP_AUDIO_FILE)
    print("[INFO] gTTS engine is ready.")
except Exception as e:
    print(f"[WARNING] Could not initialize gTTS (check internet connection): {e}. Voice output will be disabled.")
    TTS_ENABLED = False

# ==============================
# SETUP MEDIAPIPE
# ==============================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Gunakan confidence yang sama dengan saat pengambilan data
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==============================
# FUNGSI-FUNGSI PENDUKUNG (DARI SCRIPT SEBELUMNYA)
# ==============================
CANVAS_SIZE = 600
PADDING_RATIO = 0.1

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))

def extract_landmarks(results):
    # --- PERBAIKAN: Selalu kembalikan array dengan panjang TETAP ---
    # Pose: 33 titik * 4 koordinat (x, y, z, visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Left Hand: 21 titik * 3 koordinat (x, y, z) = 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right Hand: 21 titik * 3 koordinat (x, y, z) = 63
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Gabungkan semuanya. Panjangnya akan SELALU 132 + 63 + 63 = 258
    return np.concatenate([pose, lh, rh])

def get_full_body_bbox(results, img_shape):
    xs, ys = [], []
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
    
    if not xs or not ys: return None
    
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
    if bbox is None: return np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    min_x, min_y, max_x, max_y = bbox
    roi = img[min_y:max_y, min_x:max_x]
    if roi.size == 0: return canvas
    h, w = roi.shape[:2]
    scale = min((CANVAS_SIZE*0.9)/w, (CANVAS_SIZE*0.9)/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(roi, (new_w, new_h))
    x_offset = (CANVAS_SIZE - new_w)//2
    y_offset = (CANVAS_SIZE - new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# ==============================
# VARIABEL GLOBAL UNTUK REAL-TIME
# ==============================
sequence = []
sentence = []
threshold = 0.9
last_prediction_time = 0
PREDICTION_COOLDOWN = 2 # detik antar prediksi untuk menghindari pengulangan

# ==============================
# REALTIME CAPTURE LOOP
# ==============================
cap = cv2.VideoCapture(0) # Gunakan 0 jika webcam internal

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Proses dengan MediaPipe
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)
    person_detected = results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks
    
    if person_detected:
        # --- LOGIKA EKSTRAKSI & PREDIKSI ---
        keypoints = extract_landmarks(results)
        
        # Jika ada landmark yang terdeteksi, lakukan proses lengkap
        if np.any(keypoints != 0):
            # 1. Dapatkan citra yang di-crop
            bbox = get_full_body_bbox(results, frame.shape)
            canvas_crop = create_canvas_crop(frame, bbox)

            # 2. Praproses citra untuk MobileNetV2
            img_rgb = cv2.cvtColor(canvas_crop, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_rgb, (224, 224))
            preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(resized_img)
            img_batch = np.expand_dims(preprocessed_img, axis=0)

            # 3. Ekstrak fitur dengan MobileNetV2
            mobilenet_features = mobilenet_model.predict(img_batch, verbose=0).flatten()

            # 4. Gabungkan fitur dengan landmark
            fused_features = np.concatenate([mobilenet_features, keypoints])
            
            sequence.append(fused_features)
        else:
            # Jika tidak ada landmark, tambahkan vektor nol untuk menjaga panjang sequence
            sequence.append(np.zeros(1505)) # 1280 + 225

            # Update status UI
            status_text = "No Person Detected"
            status_color = (0, 0, 255) # Merah

    # Simpan 30 frame terakhir
    sequence = sequence[-30:]

    # --- LOGIKA PREDIKSI ---
    if len(sequence) == 30:
        # Prediksi hanya jika cooldown telah terlewati
        current_time = time.time()
        if current_time - last_prediction_time > PREDICTION_COOLDOWN:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            if res[np.argmax(res)] > threshold:
                predicted_label = actions[np.argmax(res)]
                
                # Tambahkan ke kalimat jika berbeda dari prediksi terakhir
                if len(sentence) == 0 or predicted_label != sentence[-1]:
                    sentence.append(predicted_label)
                    last_prediction_time = current_time # Update waktu prediksi

                    # --- PERBAIKAN: Ucapkan hasil prediksi dengan gTTS ---
                    if TTS_ENABLED:
                        try:
                            # Buat file suara dengan bahasa Indonesia
                            tts = gTTS(text=predicted_label, lang='id')
                            tts.save(TEMP_AUDIO_FILE)
                            
                            # Mainkan file suara
                            playsound(TEMP_AUDIO_FILE)
                            
                            # Hapus file suara setelah diputar
                            os.remove(TEMP_AUDIO_FILE)
                        except Exception as e:
                            # Jika gagal, nonaktifkan TTS untuk menghindari error berulang
                            print(f"[WARNING] Failed to play sound: {e}. Disabling TTS.")
                            TTS_ENABLED = False

            # Batasi panjang kalimat yang ditampilkan
            if len(sentence) > 5:
                sentence = sentence[-5:]

    # --- TAMPILKAN UI ---
    # Kotak untuk hasil teks
    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Status
    cv2.putText(image, 'Collecting Frames...', (3, 480),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(image, 'Press "R" to Reset | "Q" to Quit', (3, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('Realtime Sign Language Recognition', image)

    # --- KONTROL KEYBOARD ---
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        sentence.clear()

# Pastikan untuk membersihkan file audio sementara jika ada saat program keluar
if os.path.exists(TEMP_AUDIO_FILE):
    os.remove(TEMP_AUDIO_FILE)

cap.release()
cv2.destroyAllWindows()