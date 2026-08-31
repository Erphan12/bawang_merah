# LAPORAN KOMPREHENSIF EKSPLORASI & PENGELOLAAN DATASET
## Klasifikasi Penyakit Daun Bawang Merah Menggunakan Deep Learning CNN (MobileNetV2)

---

### 1. Ringkasan Eksekutif Dataset
Dataset yang digunakan dalam penelitian ini difokuskan pada identifikasi kondisi kesehatan tanaman bawang merah (*Allium cepa var. aggregatum*) berdasarkan citra visual daun. Dataset terdiri atas **4 kelas klasifikasi**, yaitu 3 kelas penyakit/kondisi tanaman bawang merah dan 1 kelas negatif (bukan daun bawang merah) untuk mencegah *false positive* pada lingkungan implementasi riil.

| Parameter | Keterangan / Nilai |
| :--- | :--- |
| **Total Jumlah Sampel** | **2.342 Citra** |
| **Jumlah Kelas Target** | **4 Kelas** (`moler`, `non_bawang`, `sehat`, `trotol`) |
| **Jumlah Data Tanpa Corrupt** | 2.342 Citra (100% Valid & Dapat Diproses) |
| **Rasio Pembagian Dataset** | 70% Training, 15% Validation, 15% Testing |
| **Dimensi Input Model** | 224 x 224 piksel (3 Channel RGB) |
| **Normalisasi Rentang Nilai** | Rescale $1/255$ $\rightarrow [0.0, 1.0]$ |

---

### 2. Deskripsi Karakteristik Visual Setiap Kelas

1. **Kelas Moler (Layu Fusarium / *Fusarium oxysporum*)**
   - **Karakteristik Visual**: Daun mengalami pembelitan (*twisting*), layu terkulai, bentuk abnormal spiral/keriting, menguning secara bertahap dari ujung hingga pangkal helai daun.
   - **Tantangan Deteksi**: Bentuk morfologi daun yang meliuk membutuhkan sudut pandang luas dan variasi rotasi agar tidak tertukar dengan daun sehat yang melengkung alami.

2. **Kelas Trotol (Bercak Ungu / *Alternaria porri*)**
   - **Karakteristik Visual**: Muncul bercak cekung berukuran kecil hingga sedang berwarna putih kelabu dengan tepi keunguan, berkembang menjadi lesi konsentris berwarna gelap kecokelatan hingga ungu tua.
   - **Tantangan Deteksi**: Lesi/bercak kecil seringkali memerlukan fitur spasial mikro dan variasi kecerahan/zoom agar fitur tekstur lesi tetap tertangkap jelas oleh convolutional filter.

3. **Kelas Sehat**
   - **Karakteristik Visual**: Daun berwarna hijau segar, silindris tegak, permukaan mulus tanpa bercak nekrotik, tanpa klorosis, dan tanpa pembelitan.

4. **Kelas Non Bawang (Kelas Negatif / *Out-of-Distribution*)**
   - **Karakteristik Visual**: Objek selain daun bawang merah (misalnya: rumput liar, gulma, tanah, tangan, latar belakang umum, atau tanaman lain di sekitar lahan pertanian).
   - **Tujuan**: Mencegah sistem melakukan klasifikasi keliru ketika pengguna mengunggah foto acak atau latar belakang non-tanaman.

---

### 3. Distribusi Sampel dan Statistik File Citra

Berikut adalah tabel rincian data per kelas sebelum dilakukan pemisahan (*raw dataset*):

| No | Nama Kelas | Jumlah Citra | Persentase | Komposisi Format Berkas | Variasi Resolusi Asli Terbanyak |
| :-: | :--- | :-: | :-: | :--- | :--- |
| 1 | `moler` | 500 | 21.35% | JPEG (162), PNG (338) | $500\times500$, $4000\times3000$, $3060\times4080$ |
| 2 | `non_bawang` | 842 | 35.95% | JPEG (829), WEBP (10), PNG (3) | $224\times224$, $626\times418$, $1280\times720$ |
| 3 | `sehat` | 500 | 21.35% | JPEG (500) | $3060\times3060$, $640\times640$, $3060\times4080$ |
| 4 | `trotol` | 500 | 21.35% | JPEG (500) | $640\times640$, $3060\times4080$, $256\times256$ |
| **Total** | **4 Kelas** | **2.342** | **100.0%** | **JPEG (2.001), PNG (341), WEBP (10)** | **100% File Valid (0 Corrupt)** |

---

### 4. Strategi Pembagian Data (*Dataset Splitting*)

Dataset dipisahkan menggunakan teknik *Stratified Random Splitting* dengan seed tetap (`seed = 42`) untuk menjamin sifat *reproducibility*. Komposisi pembagian adalah:
- **Training Set (70%)**: Digunakan untuk optimasi bobot model (*weight updates*).
- **Validation Set (15%)**: Digunakan untuk evaluasi performa selama pelatihan, deteksi *overfitting*, dan *checkpointing*.
- **Testing Set (15%)**: Digunakan untuk pengujian akhir yang independen (*unseen data*).

| Kelas Target | Train (70%) | Validation (15%) | Test (15%) | Total per Kelas |
| :--- | :---: | :---: | :---: | :---: |
| `moler` | 350 | 75 | 75 | **500** |
| `non_bawang` | 589 | 126 | 127 | **842** |
| `sehat` | 350 | 75 | 75 | **500** |
| `trotol` | 350 | 75 | 75 | **500** |
| **TOTAL KESELURUHAN** | **1.639** | **351** | **352** | **2.342** |

---

### 5. Tahapan Pra-Pemrosesan (*Data Preprocessing*)

Sebelum citra diumpankan ke dalam arsitektur CNN MobileNetV2, diterapkan tahapan pra-pemrosesan standar:
1. **Penyelarasan Dimensi (*Image Resizing*)**:
   Semua citra beresolusi heterogen diubah ukurannya secara konsisten menjadi dimensi $224 \times 224$ piksel menggunakan interpolasi bilinear.
2. **Normalisasi Nilai Piksel (*Rescaling*)**:
   Nilai intensitas kanal warna asli $[0, 255]$ diskalakan secara linear ke dalam rentang mengambang $[0.0, 1.0]$ melalui operasi perkalian matriks $X_{\text{norm}} = X \times \frac{1}{255}$. Hal ini memastikan stabilitas gradien saat proses *backpropagation*.
3. **Penyandian Label (*Categorical One-Hot Encoding*)**:
   Setiap label kelas diubah menjadi representasi biner 4-dimensi:
   - `moler` $\rightarrow [1, 0, 0, 0]$
   - `non_bawang` $\rightarrow [0, 1, 0, 0]$
   - `sehat` $\rightarrow [0, 0, 1, 0]$
   - `trotol` $\rightarrow [0, 0, 0, 1]$

---

### 6. Strategi Augmentasi Data (*Data Augmentation*)

Augmentasi data dilakukan secara *real-time (on-the-fly)* hanya pada subset data latih (*training set*) menggunakan `tf.keras.preprocessing.image.ImageDataGenerator`. Tujuannya adalah memperkaya variabilitas data, mencegah *overfitting*, dan meningkatkan ketahanan model terhadap kondisi lingkungan riil di lapangan.

| Teknik Augmentasi | Parameter / Konfigurasi | Alasan & Relevansi Lapangan |
| :--- | :---: | :--- |
| **Rotasi Acak** (*Rotation*) | Rentang $\pm 25^{\circ}$ | Mengantisipasi orientasi kemiringan ponsel petani saat memotret daun. |
| **Pergeseran Horizontal/Vertikal** (*Shift*) | 20% ($0.20$) | Mengakomodasi objek daun yang tidak berada tepat di titik tengah *frame*. |
| **Transformasi Geser** (*Shear*) | 20% ($0.20$) | Menstimulasi variasi sudut pengambilan gambar (*camera perspective*). |
| **Perbesaran / Perkecilan** (*Zoom*) | 30% ($0.30$) | Membantu model menangkap fitur mikro bercak trotol baik dari jarak dekat maupun jauh. |
| **Variasi Kecerahan** (*Brightness*) | Rentang $70\% - 130\%$ | Mengatasi fluktuasi intensitas sinar matahari alami di lahan pertanian (terang terik vs teduh). |
| **Pergeseran Kanal Warna** (*Channel Shift*) | $20.0$ | Memperkuat kemampuan model mengenali bercak keunguan pada daun trotol pada intensitas warna yang berbeda. |
| **Pembalikan Horizontal & Vertikal** (*Flip*) | `True` | Menggandakan variasi geometris struktur helai daun dari berbagai arah. |
| **Metode Pengisian Piksel** (*Fill Mode*) | `nearest` | Mengisi area kosong di tepi citra akibat pergeseran dan rotasi tanpa merusak konteks visual. |

---

### 7. Verifikasi dan Artefak Terkait

Grafik dan visualisasi data tersimpan pada direktori laporan proyek:
- `reports/distribusi_dataset.png` : Diagram batang distribusi dataset (Train, Val, Test).
- `reports/sample_images.png` : Sampel visual citra dari masing-masing kelas.
- `reports/confusion_matrix.png` : Hasil matriks konfusi pengujian model.
- `reports/metrics_per_class.png` : Perbandingan performa per kelas (Precision, Recall, F1-Score).
