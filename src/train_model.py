import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PASTIKAN INI ADALAH FILE OUTPUT DARI SCRIPT PREPROCESSING YANG SUDAH DIPERBAIKI
DATA_PATH = os.path.join(BASE_DIR, "data", "fused_keypoints_data.npz") 
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model.h5")

# === CEK DAN LOAD DATA ===
print(f"[INFO] Looking for data file at: {DATA_PATH}")

# 1. Cek apakah file data benar-benar ada
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Data file not found at: {DATA_PATH}")
    print("Pastikan Anda sudah menjalankan script `preprocess_data.py` dengan benar dan file tersebut berhasil dibuat.")
    exit() # Hentikan script jika file tidak ada

# 2. Coba muat data dan periksa isinya
try:
    print("[INFO] Loading dataset...")
    data = np.load(DATA_PATH, allow_pickle=True)
    
    # Cek apakah kunci 'X' dan 'y' ada di dalam file
    if "X" not in data or "y" not in data:
        print(f"[ERROR] File .npz di {DATA_PATH} tidak memiliki kunci 'X' atau 'y'.")
        print("Mungkin ini adalah file lama. Pastikan Anda menggunakan file output dari script preprocessing terbaru.")
        exit()

    X, y = data["X"], data["y"]
    label_names = data['labels']

    # 3. CEK KRUSIAL: Apakah datasetnya kosong?
    if X.shape[0] == 0:
        print("[ERROR] Dataset yang dimuat kosong (0 sampel).")
        print("Ini kemungkinan besar terjadi karena script preprocessing tidak menemukan data citra (.jpg) atau landmark (.npy) untuk diproses.")
        print("Silakan periksa kembali folder data Anda dan pastikan struktur foldernya benar.")
        exit()

except Exception as e:
    print(f"[ERROR] Terjadi kesalahan saat memuat file data: {e}")
    exit()

# Jika semua pengecekan lolos, lanjutkan
print(f"[SUCCESS] Dataset loaded successfully!")
print(f"[INFO] Features shape (X): {X.shape}")
print(f"[INFO] Labels shape (y): {y.shape}")
print(f"[INFO] Labels found: {label_names}")

# === ENCODE LABELS ===
# Label sudah dalam bentuk numerik dari script preprocessing, jadi encoding tidak perlu
# Tapi kita tetap memerlukan encoder untuk menyimpan kelasnya
print("[INFO] Preparing labels...")
le = LabelEncoder()
le.fit(label_names) # Fit dengan nama label asli
# y sudah berupa index, tidak perlu diubah lagi

# Simpan kelas untuk realtime test
np.save(os.path.join(BASE_DIR, "models", "label_classes.npy"), le.classes_)

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y # Ubah test_size jadi 15% untuk val, 15% untuk test
)

# === REVISED: AUGMENTATION FUNCTIONS ===
# Kita perlu memisahkan fitur MobileNetV2 dan Landmark
MOBILENET_FEATURE_SIZE = 1280
LANDMARK_FEATURE_SIZE = X.shape[2] - MOBILENET_FEATURE_SIZE

def horizontal_flip_landmarks(sequence):
    """Balik koordinat X pada bagian landmark saja."""
    landmark_part = sequence[:, MOBILENET_FEATURE_SIZE:]
    # Asumsi struktur landmark: [x1, y1, z1, x2, y2, z2, ...]
    landmark_part[:, 0::3] *= -1
    return landmark_part

def add_jitter_landmarks(sequence, sigma=0.005):
    """Tambah noise kecil pada bagian landmark."""
    landmark_part = sequence[:, MOBILENET_FEATURE_SIZE:]
    noise = np.random.normal(0, sigma, landmark_part.shape)
    return landmark_part + noise

def random_scale_landmarks(sequence, scale_range=(0.95, 1.05)):
    """Skalakan koordinat pada bagian landmark."""
    landmark_part = sequence[:, MOBILENET_FEATURE_SIZE:]
    factor = np.random.uniform(*scale_range)
    return landmark_part * factor

def random_shift_landmarks(sequence, shift_range=(-0.03, 0.03)):
    """Geser semua keypoints pada bagian landmark."""
    landmark_part = sequence[:, MOBILENET_FEATURE_SIZE:]
    shift_x = np.random.uniform(*shift_range)
    shift_y = np.random.uniform(*shift_range)
    shifted = landmark_part.copy()
    shifted[:, 0::3] += shift_x
    shifted[:, 1::3] += shift_y
    return shifted

def frame_dropout(sequence, drop_prob=0.1):
    """Random hilangkan beberapa frame, lalu pad ke timesteps asli."""
    mask = np.random.rand(sequence.shape[0]) > drop_prob
    dropped = sequence[mask] if mask.any() else sequence
    if dropped.shape[0] < sequence.shape[0]:
        pad_len = sequence.shape[0] - dropped.shape[0]
        dropped = np.pad(dropped, ((0, pad_len), (0,0)), 'edge')
    return dropped

# === APPLY AUGMENTATION PADA TRAINING DATA ===
print("[INFO] Applying data augmentation...")
timesteps = X.shape[1]
aug_X, aug_y = [], []

for xi, yi in zip(X_train, y_train):
    # Frame dropout bisa diterapkan ke seluruh sequence
    aug_X.append(frame_dropout(xi)); aug_y.append(yi)
    
    # Untuk augmentasi lainnya, kita pisah dan gabungkan kembali
    mobilenet_part = xi[:, :MOBILENET_FEATURE_SIZE]
    
    # Horizontal Flip
    flipped_landmarks = horizontal_flip_landmarks(xi)
    aug_X.append(np.concatenate([mobilenet_part, flipped_landmarks], axis=1)); aug_y.append(yi)
    
    # Jitter
    jittered_landmarks = add_jitter_landmarks(xi)
    aug_X.append(np.concatenate([mobilenet_part, jittered_landmarks], axis=1)); aug_y.append(yi)
    
    # Scale
    scaled_landmarks = random_scale_landmarks(xi)
    aug_X.append(np.concatenate([mobilenet_part, scaled_landmarks], axis=1)); aug_y.append(yi)
    
    # Shift
    shifted_landmarks = random_shift_landmarks(xi)
    aug_X.append(np.concatenate([mobilenet_part, shifted_landmarks], axis=1)); aug_y.append(yi)

# Gabungkan dengan data asli
X_train = np.concatenate([X_train, np.array(aug_X)], axis=0)
y_train = np.concatenate([y_train, np.array(aug_y)], axis=0)

print(f"[INFO] After augmentation: X_train={X_train.shape}, y_train={y_train.shape}")

# === BUILD MODEL ===
num_classes = len(np.unique(y))
features = X.shape[2]

model = Sequential([
    LSTM(256, return_sequences=True, input_shape=(timesteps, features)), # REVISED: Tambah unit LSTM
    Dropout(0.3),
    BatchNormalization(),

    LSTM(128, return_sequences=False), # REVISED: Tambah unit LSTM
    Dropout(0.3),
    BatchNormalization(),

    Dense(128, activation="relu"), # REVISED: Tambah unit Dense
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# === CALLBACKS ===
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True) # NEW: Pastikan folder ada
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1), # REVISED: Tambah patience
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=6, verbose=1, min_lr=1e-6), # REVISED: Lebih agresif
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss", verbose=1)
]

# === TRAIN MODEL ===
print("[INFO] Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, # REVISED: Tambah epochs, EarlyStopping akan menghentikannya
    batch_size=32,
    callbacks=callbacks
)

# === SAVE FINAL MODEL ===
model.save(MODEL_PATH)
print(f"[INFO] Model saved at {MODEL_PATH}")

# === NEW: PLOT TRAINING HISTORY ===
def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Training history plot saved at {save_path}")
    plt.show()

plot_history(history, os.path.join(BASE_DIR, "data", "training_history.png"))