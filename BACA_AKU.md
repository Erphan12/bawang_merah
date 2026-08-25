# Panduan Jalankan Sistem Deteksi Penyakit Bawang Merah (CNN)

UNTUK PREVIEW LEBIH BAIK : CTRL + SHIFT + V

## 1. Install Library
```bash
pip install -r requirements.txt
```

## 2. Cara Menjalankan Website (Langsung Siap Pakai)
Jalankan server backend & frontend (FastAPI):
```bash
uvicorn src.main_api:app --reload --port 8000
```
Lalu buka browser di: **`http://localhost:8000`**

---

## 3. Tahapan Pelatihan Ulang Model (Opsional)
Jika ingin memproses dataset & melatih ulang model dari awal:

1. **Cek Dataset Awal:**
   ```bash
   python src/check_dataset.py
   ```
2. **Bagi Dataset (Train/Val/Test):**
   ```bash
   python src/split_dataset.py
   ```
3. **Visualisasi Data:**
   ```bash
   python src/visualize_and_check.py
   ```
4. **Latih Model CNN:**
   ```bash
   python src/train_model.py
   ```
5. **Evaluasi Model:**
   ```bash
   python src/evaluate_model.py
   ```

