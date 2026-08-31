# PANDUAN REVISI DOKUMEN SKRIPSI (NATASHA_TUTUP.docx)
**Judul Penelitian:** Sistem Klasifikasi Penyakit Daun Bawang Merah Menggunakan Deep Learning Berbasis Convolutional Neural Network (CNN) MobileNetV2

Dokumen ini memuat seluruh data terbaru yang **wajib diselaraskan** pada draf skripsi [NATASHA_TUTUP.docx](file:///d:/TUTUP/bawang_merah/NATASHA_TUTUP.docx) setelah penambahan dataset `non_bawang` menjadi 842 citra dan optimasi model Transfer Learning MobileNetV2 (Deep Fine-Tuning + Dual Data Augmentation).

---

## 1. Rangkuman Perubahan Utama (Sebelum vs Sesudah)

| Parameter / Metrik | Nilai Lama (Draf Skripsi) | Nilai Baru (Aktual Sistem) | Keterangan Perubahan |
|---|:---:|:---:|---|
| **Total Dataset Keseluruhan** | 2.000 citra | **2.342 citra** | Penambahan 342 citra variasi negatif *non-bawang* |
| **Data Kelas `non_bawang`** | 500 citra | **842 citra** | *Hard negatives* (tanah, gulma, daun lain, objek kebun) |
| **Data Training (70%)** | 1.400 citra | **1.639 citra** | moler: 350, non_bawang: 589, sehat: 350, trotol: 350 |
| **Data Validasi (15%)** | 300 citra | **351 citra** | moler: 75, non_bawang: 126, sehat: 75, trotol: 75 |
| **Data Testing (15%)** | 300 citra | **352 citra** | moler: 75, non_bawang: 127, sehat: 75, trotol: 75 |
| **Test Accuracy** | 93,67% (atau 94,33%) | **96,59%** | Naik +2,92% (Peningkatan signifikan) |
| **Jumlah Salah Prediksi (Test Set)** | 17 - 19 citra salah | **Hanya 12 citra salah** | Dari total 352 data uji |
| **Akurasi Kelas `non_bawang`** | 98,67% | **100,00% (Sempurna)** | Precision 1.0, Recall 1.0, F1-Score 1.0 |
| **Macro Average Precision** | 93,83% | **96,11%** | Rata-rata presisi antar seluruh 4 kelas |
| **Macro Average Recall** | 93,67% | **96,00%** | Rata-rata daya ingat antar seluruh 4 kelas |
| **Macro Average F1-Score** | 93,70% | **96,03%** | Rata-rata F1 antar seluruh 4 kelas |
| **Weighted Average F1-Score** | 93,70% | **96,61%** | Rata-rata tertimbang proporsi sampel |

---

## 2. Tabel-Tabel Bab IV yang Perlu Diperbarui di Word

### A. Pembaruan Tabel 4.1: Distribusi Jumlah Sampel Dataset
*Gantikan Tabel 4.1 di skripsi dengan data berikut:*

| Kelas Kategori | Data Training (70%) | Data Validasi (15%) | Data Testing (15%) | Total Sampel |
|---|:---:|:---:|:---:|:---:|
| **Moler (Layu Fusarium)** | 350 | 75 | 75 | 500 |
| **Bukan Bawang (Non-Bawang)** | **589** | **126** | **127** | **842** |
| **Sehat** | 350 | 75 | 75 | 500 |
| **Trotol (Bercak Ungu)** | 350 | 75 | 75 | 500 |
| **Total Overall** | **1.639** | **351** | **352** | **2.342** |

---

### B. Pembaruan Tabel 4.2: Confusion Matrix (Data Uji 352 Sampel)
*Gantikan Tabel 4.2 di skripsi dengan matriks berikut:*

| Label Aktual \ Label Prediksi | Moler | Bukan Bawang | Sehat | Trotol | Total Aktual |
|---|:---:|:---:|:---:|:---:|:---:|
| **Moler** | **71** | 0 | 3 | 1 | **75** |
| **Bukan Bawang** | 0 | **127** | 0 | 0 | **127** |
| **Sehat** | 0 | 0 | **71** | 4 | **75** |
| **Trotol** | 0 | 0 | 4 | **71** | **75** |
| **Total Prediksi** | **71** | **127** | **78** | **76** | **352** |

---

### C. Pembaruan Tabel 4.3: Perhitungan Metrik Evaluasi Kuantitatif
*Gantikan Tabel 4.3 di skripsi dengan metrik berikut:*

| Kelas Kategori | Precision | Recall | F1-Score | Jumlah Sampel (Support) |
|---|:---:|:---:|:---:|:---:|
| **Moler** | 100,00% | 94,67% | 97,26% | 75 |
| **Bukan Bawang** | 100,00% | 100,00% | 100,00% | 127 |
| **Sehat** | 91,03% | 94,67% | 92,81% | 75 |
| **Trotol** | 93,42% | 94,67% | 94,04% | 75 |
| **Rata-rata Makro (Macro Avg)** | **96,11%** | **96,00%** | **96,03%** | **352** |
| **Rata-rata Tertimbang (Weighted Avg)** | **96,69%** | **96,59%** | **96,61%** | **352** |
| **Akurasi Keseluruhan (Accuracy)** | — | — | **96,59%** | **352** |

---

## 3. Teks Revisi untuk Bagian-Bagian Tertentu

### A. Revisi Abstrak Bahasa Indonesia
```text
Bawang merah merupakan komoditas hortikultura strategis dengan nilai ekonomi tinggi, namun produktivitasnya kerap terkendala oleh penyakit daun seperti trotol dan layu fusarium/moler, yang sulit dikenali oleh petani sejak tahap awal. Penelitian ini bertujuan untuk mengimplementasikan sistem klasifikasi penyakit bawang merah berbasis Deep Learning menggunakan arsitektur Convolutional Neural Network (CNN) dengan model MobileNetV2, yang diintegrasikan ke dalam aplikasi web agar mudah diakses oleh petani maupun penyuluh pertanian. Penelitian ini menggunakan metode kuantitatif dengan pendekatan deep learning CNN meliputi pengumpulan dataset di Balai Penyuluhan Pertanian (BPP) Kecamatan Anggeraja Kabupaten Enrekang sebanyak 2.342 citra daun dan umbi bawang merah yang terbagi ke dalam empat kelas, yaitu moler, trotol, sehat, dan non_bawang, dengan pembagian data latih, validasi, dan uji masing-masing sebesar 70% (1.639 citra), 15% (351 citra), dan 15% (352 citra). Model dilatih menggunakan pendekatan transfer learning dengan fine-tuning dan dievaluasi menggunakan confusion matrix, accuracy, precision, recall, dan F1-score, kemudian diimplementasikan pada sistem berbasis web yang memungkinkan pengguna mengunggah atau menangkap citra daun secara langsung untuk memperoleh hasil diagnosis. Hasil pengujian menunjukkan bahwa model mencapai test accuracy sebesar 96,59% pada 352 data uji independen, dengan macro average precision sebesar 96,11%, recall sebesar 96,00%, dan F1-score sebesar 96,03%. Kelas non_bawang berhasil dikenali secara sempurna (100%), sementara kesalahan klasifikasi minim (hanya 12 citra salah dari 352 data uji) terjadi akibat kemiripan visual gejala infeksi pada fase sangat awal. Pengujian fungsionalitas black-box menunjukkan bahwa seluruh fitur aplikasi dan pipeline ekstraksi layer CNN berjalan sesuai rancangan, sehingga membuktikan bahwa sistem ini sangat layak dan akurat digunakan sebagai alat bantu diagnosis dini penyakit bawang merah.
```

### B. Revisi Abstrak Bahasa Inggris (Abstract)
```text
Shallots are a strategic horticultural commodity with high economic value, but their productivity is often hindered by leaf diseases such as purple blotch (trotol) and fusarium wilt (moler), which are difficult for farmers to recognize in the early stages. This study aims to implement a shallot disease classification system based on Deep Learning using the Convolutional Neural Network (CNN) architecture with the MobileNetV2 model, integrated into a web application for easy access by farmers and agricultural extension officers. This study employs a quantitative method with a CNN deep learning approach, including dataset collection at the Agricultural Extension Center (BPP) of Anggeraja District, Enrekang Regency, consisting of 2,342 images of shallot leaves and tubers divided into four classes: moler, trotol, sehat, and non_bawang, with training, validation, and testing distributions of 70% (1,639 images), 15% (351 images), and 15% (352 images), respectively. The model was trained using a transfer learning fine-tuning approach and evaluated using a confusion matrix, accuracy, precision, recall, and F1-score, then deployed into a web-based system allowing users to upload or capture leaf images directly. Testing results indicate that the model achieved a test accuracy of 96.59% on 352 independent test data, with a macro average precision of 96.11%, recall of 96.00%, and an F1-score of 96.03%. The non_bawang class was recognized with 100% accuracy, while misclassifications were minimal (only 12 images out of 352 test samples) caused by visual subtleties during early infection phases. Black-box functionality testing confirmed that all web features and CNN layer extraction pipelines function strictly as designed, validating the system as highly feasible and accurate for early shallot disease diagnosis.
```

### C. Revisi Batasan Masalah (Bab I - Butir 3 & 4)
* **Poin 3:** *"Jumlah dataset citra yang digunakan dalam penelitian ini yaitu sebanyak 2.342 gambar dengan format .jpg/.jpeg yang terbagi dalam 4 kelas (Moler, Sehat, Trotol, dan Non-Bawang)."*
* **Poin 4:** *"Dataset terbagi menjadi 3 bagian dengan proporsi 70% data training (1.639 citra), 15% data validation (351 citra), dan 15% data testing (352 citra)."*

### D. Revisi Pembahasan Evaluasi (Bab IV)
* **Akurasi Keseluruhan:** *"Berdasarkan hasil evaluasi pada Tabel 4.3, model berhasil mengklasifikasikan **340 dari 352 data uji dengan benar**, sehingga mencapai tingkat akurasi pengujian (Test Accuracy) sebesar **96,59%** dengan nilai Loss sebesar 0,3261."*
* **Analisis Kesalahan (Misclassification):** *"Dari total 352 data uji, hanya terdapat **12 citra (3,41%) yang mengalami kesalahan prediksi**. Rincian kesalahan mencakup 4 citra sehat terprediksi trotol, 4 citra trotol terprediksi sehat, 3 citra moler terprediksi sehat, dan 1 citra moler terprediksi trotol. Kelas non-bawang berhasil diklasifikasikan dengan sempurna tanpa satu pun kesalahan (127 dari 127 citra benar, Precision 100%, Recall 100%)."*

### E. Revisi Kesimpulan (Bab V)
* **Butir 1:** *"Sistem klasifikasi penyakit bawang merah berbasis CNN MobileNetV2 berhasil dikembangkan dengan memanfaatkan dataset sebanyak 2.342 citra yang mencakup 4 kelas target (Moler, Trotol, Sehat, dan Bukan Bawang) serta diintegrasikan ke dalam antarmuka web interaktif yang menyajikan visualisasi 7 tahap layer ekstraksi fitur."*
* **Butir 2:** *"Sistem mampu mengklasifikasikan penyakit bawang merah dengan sangat baik, mencapai tingkat Test Accuracy sebesar **96,59%**, Macro Precision **96,11%**, Macro Recall **96,00%**, dan Macro F1-Score **96,03%** pada 352 data uji independen."*

---

## 4. Lokasi File Gambar Laporan Terkait
Seluruh grafik dan visualisasi terbaru sudah siap dipakai untuk menggantikan gambar lama di dokumen:
- **Confusion Matrix:** `reports/confusion_matrix.png`
- **Kurva Training & Validasi (Akurasi & Loss):** `reports/training_history.png`
- **Sampel Misklasifikasi:** `reports/misclassified_samples.png`
