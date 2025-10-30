# src/preprocess_data.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

# --- Konfigurasi ---
DATA_PATH = "G:/File Arya/NEW_TA_MB_LSTM_OKT/data/Kata" 
OUTPUT_FILE = "G:/File Arya/NEW_TA_MB_LSTM_OKT/data/fused_keypoints_data.npz"

SEQUENCE_LENGTH = 30
IMG_SIZE = 224

# --- 1. Muat Model MobileNetV2 ---
print("[INFO] Loading MobileNetV2 model...")
try:
    mobilenet_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    print("[SUCCESS] Model loaded.")
except Exception as e:
    print(f"[ERROR] Failed to load MobileNetV2 model: {e}")
    exit()

# --- 2. Siapkan Label dan Path ---
print(f"\n[INFO] Scanning for actions in: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] The path {DATA_PATH} does not exist. Please check the DATA_PATH variable.")
    exit()

try:
    actions = [action for action in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, action))]
    actions.sort()
    if not actions:
        print(f"[ERROR] No action folders found in {DATA_PATH}.")
        exit()
    print(f"[SUCCESS] Found {len(actions)} actions: {actions}")
except Exception as e:
    print(f"[ERROR] Could not list directories in {DATA_PATH}: {e}")
    exit()

label_map = {label: num for num, label in enumerate(actions)}
print(f"[INFO] Label map created: {label_map}")

# --- 3. Proses Setiap Data ---
print("\n[INFO] Starting data preprocessing...")
sequences, labels = [], []
processed_sequences = 0
skipped_sequences = 0

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    print(f"\n--- Processing action: {action} ---")
    
    try:
        sequences_in_action = os.listdir(action_path)
        print(f"[INFO] Found {len(sequences_in_action)} sequences in {action_path}")
    except Exception as e:
        print(f"[ERROR] Could not list sequences for action {action}: {e}")
        continue

    for sequence in tqdm(sequences_in_action, desc=f"Processing {action}"):
        sequence_path = os.path.join(action_path, sequence)
        window = []
        
        try:
            npy_files = [f for f in os.listdir(sequence_path) if f.endswith('.npy')]
            if len(npy_files) != SEQUENCE_LENGTH:
                print(f"\n[WARNING] Sequence {sequence} in {action} is incomplete. Found {len(npy_files)} .npy frames, expected {SEQUENCE_LENGTH}. Skipping.")
                skipped_sequences += 1
                continue
        except Exception as e:
            print(f"\n[ERROR] Could not list frames in {sequence_path}: {e}")
            skipped_sequences += 1
            continue

        for frame_num in range(SEQUENCE_LENGTH):
            res_path = os.path.join(sequence_path, f"{frame_num}.npy")
            img_path = os.path.join(sequence_path, f"{frame_num}.jpg")

            if not os.path.exists(res_path) or not os.path.exists(img_path):
                print(f"\n[ERROR] Missing .npy or .jpg file in {sequence_path} for frame {frame_num}. Skipping sequence.")
                window = [] 
                skipped_sequences += 1
                break

            try:
                # --- PERBAIKAN AKHIR: Muat landmark secara langsung ---
                # File .npy sudah berisi array yang sudah digabung, jadi kita langsung muat saja.
                landmarks = np.load(res_path, allow_pickle=True)

                img = cv2.imread(img_path)
                if img is None:
                    print(f"\n[ERROR] Failed to read image {img_path}. It might be corrupted. Skipping sequence.")
                    window = []
                    skipped_sequences += 1
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
                preprocessed_img = tf.keras.applications.mobilenet_v2.preprocess_input(resized_img)
                img_batch = np.expand_dims(preprocessed_img, axis=0)
                mobilenet_features = mobilenet_model.predict(img_batch, verbose=0).flatten()
                
                # Gabungkan fitur MobileNetV2 dengan landmark yang sudah benar
                fused_features = np.concatenate([mobilenet_features, landmarks])
                window.append(fused_features)

            except Exception as e:
                print(f"\n[ERROR] Could not process frame {frame_num} in sequence {sequence}. Error: {e}. Skipping sequence.")
                window = [] 
                skipped_sequences += 1
                break
        
        if len(window) == SEQUENCE_LENGTH:
            sequences.append(window)
            labels.append(label_map[action])
            processed_sequences += 1

# --- 4. Simpan Hasil ---
print("\n--- Preprocessing Summary ---")
print(f"Total sequences successfully processed: {processed_sequences}")
print(f"Total sequences skipped: {skipped_sequences}")

if processed_sequences > 0:
    print("[INFO] Saving processed data...")
    X = np.array(sequences)
    y = np.array(labels)

    np.savez(OUTPUT_FILE, X=X, y=y, labels=actions, label_map=label_map)

    print(f"\n[SUCCESS] Dataset saved at {OUTPUT_FILE}")
    print(f"[INFO] Final X shape: {X.shape}")
    print(f"[INFO] Final y shape: {y.shape}")
else:
    print("\n[ERROR] No data was processed. The output file will not be created.")
    print("Please check the warnings and errors above to diagnose the problem.")