import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import tempfile
import os
import time
from tensorflow.keras.models import load_model  # type: ignore
from gtts import gTTS

# ==============================
# KONFIGURASI DASAR
# ==============================
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model.h5")
LABEL_CLASSES_PATH = os.path.join(BASE_DIR, "models", "label_classes.npy")
TEMP_AUDIO_FILE = os.path.join(tempfile.gettempdir(), "temp_prediction.mp3")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_all_models():
    model = load_model(MODEL_PATH)
    mobilenet_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    actions = np.load(LABEL_CLASSES_PATH)
    return model, mobilenet_model, actions

model, mobilenet_model, actions = load_all_models()

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==============================
# FUNGSI PENDUKUNG
# ==============================
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def draw_styled_landmarks(image, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_landmarks(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def get_bbox(results, shape):
    xs, ys = [], []
    for lm_set in [results.pose_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
        if lm_set:
            for lm in lm_set.landmark:
                xs.append(int(lm.x * shape[1]))
                ys.append(int(lm.y * shape[0]))
    if not xs or not ys:
        return None
    return max(0, min(xs)), max(0, min(ys)), min(shape[1], max(xs)), min(shape[0], max(ys))

def create_canvas_crop(img, bbox):
    CANVAS_SIZE = 600
    if bbox is None:
        return np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    x1, y1, x2, y2 = bbox
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    h, w = roi.shape[:2]
    scale = min((CANVAS_SIZE*0.9)/w, (CANVAS_SIZE*0.9)/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(roi, (new_w, new_h))
    canvas = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255
    x_offset, y_offset = (CANVAS_SIZE-new_w)//2, (CANVAS_SIZE-new_h)//2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='id')
        tts.save(TEMP_AUDIO_FILE)
        st.audio(TEMP_AUDIO_FILE, format='audio/mp3', autoplay=True)
    except Exception as e:
        st.warning(f"Voice output error: {e}")

# ==============================
# STREAMLIT UI
# ==============================
st.title("🤟 Real-time Sign Language Translator")
st.markdown("Aplikasi ini menerjemahkan bahasa isyarat ke teks dan suara secara real-time menggunakan kamera.")

col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Mulai Deteksi")
with col2:
    stop_button = st.button("Hentikan")

FRAME_WINDOW = st.image([])
sentence_placeholder = st.empty()

# ==============================
# LOOP STREAMING
# ==============================
sequence = []
sentence = []
threshold = 0.9
last_prediction_time = 0
COOLDOWN = 2

if start_button:
    cap = cv2.VideoCapture(0)
    st.info("Kamera aktif. Tekan 'Hentikan' untuk berhenti.")
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.warning("Tidak dapat membaca frame dari kamera.")
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_landmarks(results)
        if np.any(keypoints != 0):
            bbox = get_bbox(results, frame.shape)
            canvas_crop = create_canvas_crop(frame, bbox)

            img_rgb = cv2.cvtColor(canvas_crop, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(img_rgb, (224, 224))
            preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(resized_img)
            mobilenet_features = mobilenet_model.predict(np.expand_dims(preprocessed_img, axis=0), verbose=0).flatten()

            fused_features = np.concatenate([mobilenet_features, keypoints])
            sequence.append(fused_features)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                current_time = time.time()
                if current_time - last_prediction_time > COOLDOWN:
                    res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    if res[np.argmax(res)] > threshold:
                        predicted_label = actions[np.argmax(res)]
                        if len(sentence) == 0 or predicted_label != sentence[-1]:
                            sentence.append(predicted_label)
                            last_prediction_time = current_time
                            text_to_speech(predicted_label)

        if len(sentence) > 5:
            sentence = sentence[-5:]

        FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        sentence_placeholder.markdown(f"### 🗣️ Prediksi: {' '.join(sentence)}")

    cap.release()
    st.success("Deteksi dihentikan.")
