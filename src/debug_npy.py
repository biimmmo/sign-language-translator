import os
import numpy as np

# --- Konfigurasi Path ---
# Sesuaikan path ini dengan lokasi data Anda
DATA_PATH = "G:/File Arya/NEW_TA_MB_LSTM_OKT/data/Kata" 


# Cari satu file .npy untuk diuji
# Script akan mencoba mencari file '0.npy' di dalam folder sequence '0' dari aksi pertama
try:
    first_action = os.listdir(DATA_PATH)[0]
    first_sequence = os.listdir(os.path.join(DATA_PATH, first_action))[0]
    TEST_FILE_PATH = os.path.join(DATA_PATH, first_action, first_sequence, "0.npy")
    print(f"--- MENGUJI FILE: {TEST_FILE_PATH} ---")
except Exception as e:
    print(f"[ERROR] Tidak bisa menemukan file .npy untuk diuji. Periksa path dan struktur folder. Error: {e}")
    exit()

# --- Muat dan Cetak Struktur Data ---
try:
    # Memuat data
    loaded_data = np.load(TEST_FILE_PATH, allow_pickle=True)
    
    print("\n[INFO] Hasil np.load(..., allow_pickle=True):")
    print(f"Tipe objek yang dimuat: {type(loaded_data)}")
    
    # Jika hasilnya adalah 'scalar' (objek tunggal), kita perlu mengambil isinya dengan .item()
    if loaded_data.ndim == 0:
        data_dict = loaded_data.item()
        print("\n[INFO] Mengambil objek dengan .item()...")
        print(f"Tipe setelah .item(): {type(data_dict)}")
        print(f"Kunci-kunci dalam dictionary: {data_dict.keys()}")

        print("\n--- DETAIL SETIAP KUNCI ---")
        for key, value in data_dict.items():
            print(f"\nKunci: '{key}'")
            print(f"  - Tipe nilai: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"  - Shape array: {value.shape}")
                print(f"  - 5 elemen pertama: {value.flatten()[:5]}")
            else:
                print(f"  - Isi nilai: {value}")
    else:
        print("\n[INFO] Data bukan scalar, langsung menampilkan isinya.")
        print(f"Isi data: {loaded_data}")
        if isinstance(loaded_data, np.ndarray):
            print(f"Shape array: {loaded_data.shape}")

except Exception as e:
    print(f"\n[ERROR] Gagal memuat atau memproses file: {e}")
