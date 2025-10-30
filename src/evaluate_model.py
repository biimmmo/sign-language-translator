# src/evaluate_model.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.models import load_model # type: ignore
from sklearn.model_selection import train_test_split

# === Konfigurasi Path ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "keypoints_datarev1.npz")
MODEL_PATH = os.path.join(BASE_DIR, "models", "sign_model1536.h5")

# ACTIONS = ["Huruf/A", "Huruf/B", "Kata/Halo", "Kata/Nama", "Kata/Saya"]
ACTIONS = ["Huruf/A", "Huruf/B", "Huruf/I", "Huruf/M" , "Huruf/O" , "Kata/Halo" , "Kata/Nama","Kata/Saya"]
# === Load Dataset ===
print("🔄 Loading dataset...")
data = np.load(DATA_PATH, allow_pickle=True)
X, y = data["X"], data["y"]

# Bagi lagi menjadi train/val dengan random_state sama seperti di train_model.py
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Validation data: {X_val.shape}, Labels: {y_val.shape}")

# === Load Model ===
print(f"🔄 Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# === Evaluasi Model ===
print("\n📊 Evaluating model...")
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(y_val, y_pred_classes)
precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred_classes, average="weighted")

print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall   : {recall:.4f}")
print(f"✅ F1-Score : {f1:.4f}")

print("\n📝 Classification Report:")
print(classification_report(y_val, y_pred_classes, target_names=ACTIONS))

# === Confusion Matrix ===
cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=ACTIONS, yticklabels=ACTIONS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
