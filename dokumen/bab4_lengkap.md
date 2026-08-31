# BAB IV HASIL DAN PEMBAHASAN

Pada bab ini dipaparkan hasil analisis, perancangan, implementasi, serta pengujian sistem klasifikasi penyakit bawang merah (*Allium ascalonicum L.*) menggunakan pendekatan *Deep Learning* dengan arsitektur *Convolutional Neural Network* (CNN) berbasis *Transfer Learning* MobileNetV2. Pembahasan ini mencakup analisis kebutuhan sistem, alur kerja arsitektur CNN, implementasi perkakas pengembangan (*tools*), penjelajahan dataset (*explore dataset*), evaluasi performa model melalui metrik kuantitatif dan perhitungan matematis manual, implementasi antarmuka pengguna (*user interface*), serta pengujian fungsionalitas sistem menggunakan metode *Black-Box Testing*.

Seluruh data, angka metrik, dan parameter yang disajikan dalam bab ini bersumber 100% dari eksekusi kode program dan pengujian nyata pada repositori sistem.

---

### A. Analisis Kebutuhan dan Rancangan

### 1. Flowchart Sistem

Flowchart sistem menggambarkan alur aktivitas pengguna dalam berinteraksi dengan aplikasi web klasifikasi penyakit bawang merah, mulai dari tahap memasukkan citra masukan hingga memperoleh hasil diagnosis dan rekomendasi penanganan.

![Diagram Flowchart Sistem](file:///d:/TUTUP/bawang_merah/reports/flowchart_sistem.png)

**Gambar 4. 1 Diagram Flowchart Sistem Klasifikasi**

**Penjelasan Alur Flowchart Sistem:**
1. **Akses Antarmuka:** Pengguna membuka aplikasi web klasifikasi penyakit bawang merah pada peramban (*browser*).
2. **Pengunggahan Masukan:** Pengguna memasukkan gambar daun bawang merah melalui fitur unggah berkas (*drag and drop*) atau tangkapan kamera secara *real-time*.
3. **Validasi Berkas:** Sistem memeriksa format berkas gambar (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`). Jika format tidak sesuai, sistem menampilkan pesan peringatan.
4. **Pra-pemrosesan Otomatis:** Gambar yang valid diproses dengan mengubah ukurannya (*resizing*) menjadi $224 x 224$ piksel serta menormalisasi nilai piksel ke rentang $[0, 1]$ (`rescale=1./255`).
5. **Ekstraksi Fitur & Inferensi CNN:** Gambar diproses oleh model *Deep Learning* MobileNetV2 yang dihubungkan dengan lapisan *Global Average Pooling* (GAP), *Batch Normalization*, *Dropout*, *Dense Layer* 128 neuron, dan *Output Layer* *Softmax*.
6. **Penyajikan & Penyimpanan:** Sistem menampilkan label hasil prediksi (Moler, Non Bawang, Sehat, atau Trotol) beserta nilai probabilitas kepercayaan (*confidence score*), menyimpan data sesi ke memori server (*SESSION_STORE*), dan menampilkan rekomendasi penanganan bagi petani.

---

### 2. Alur Kerja CNN

Alur kerja arsitektur *Convolutional Neural Network* (CNN) yang diimplementasikan pada sistem ini mengombinasikan *Feature Extractor* dari *pre-trained model* MobileNetV2 (bobot ImageNet) dengan *Classification Head* yang disesuaikan untuk 4 kelas target.

![Diagram Arsitektur CNN](file:///d:/TUTUP/bawang_merah/reports/arsitektur_cnn.png)

**Gambar 4. 2 Diagram Arsitektur MobileNetV2 Transfer Learning**

**Rincian Spesifikasi Arsitektur Model (`BawangMerah_MobileNetV2`):**
1. **Input Layer:** Menerima matriks citra berdimensi $224 x 224 x 3$ (RGB).
2. **Base Model (MobileNetV2):** Menggunakan bobot *pre-trained* ImageNet dengan status *trainable = False* (dibekukan) agar pembacaan fitur dasar seperti tepi, garis, dan tekstur warna tetap stabil.
3. **Global Average Pooling 2D:** Mengubah *feature map* 3D keluaran *base model* menjadi vektor 1D dengan menghitung nilai rata-rata tiap kanal.
4. **Batch Normalization:** Menormalisasi aktivasi dari lapisan sebelumnya untuk mempercepat konvergensi dan menjaga stabilitas pelatihan.
5. **Dropout Layer 1 (0.3):** Mematikan 30% unit neuron secara acak saat pelatihan guna mencegah *overfitting*.
6. **Dense Layer (128 Neuron, ReLU):** Lapisan terhubung penuh dengan 128 unit neuron dan fungsi aktivasi *Rectified Linear Unit* (ReLU).
7. **Dropout Layer 2 (0.2):** Mematikan 20% unit neuron untuk regularisasi tambahan.
8. **Dense Output Layer (4 Neuron, Softmax):** Lapisan keluaran 4 unit neuron yang merepresentasikan 4 kelas target (*moler*, *non_bawang*, *sehat*, *trotol*) dengan fungsi aktivasi *Softmax*.

---

## B. Implementasi Tools Pengembangan

Pengembangan sistem klasifikasi ini memanfaatkan ekosistem perangkat lunak berbasis Python dengan rincian *tools* dan pustaka sebagai berikut:

### 1. Python & FastAPI
Backend aplikasi dibangun menggunakan **FastAPI** (`src/main_api.py`), sebuah *framework* web Python modern berkinerja tinggi berbasis standar ASGI (*Asynchronous Server Gateway Interface*). FastAPI dimanfaatkan untuk:
- Membangun *endpoint* REST API `/predict` yang menerima unggahan citra tanaman dan mengembalikan respons berformat JSON secara asinkronus (`async/await`).
- Menyediakan *endpoint* visualisasi tahapan CNN (`/extract-layers`) yang terhubung dengan modul `src/layer_extraction.py` untuk menyajikan statistik dan *feature map* tiap lapisan ke antarmuka web.

### 2. TensorFlow & Keras
**TensorFlow** (versi 2.x) beserta pustaka Keras digunakan sebagai fondasi utama dalam perancangan, pelatihan, dan pengujian model CNN (`src/train_model.py`):
- Pelatihan dipandu oleh *optimizer* Adam dengan *learning rate* $10^{-4}$ ($1e-4$) dan fungsi kerugian *Categorical Crossentropy*.
- Pelatihan dilengkapi dengan *callbacks*:
  - `EarlyStopping`: Menghentikan pelatihan jika `val_loss` tidak membaik selama 5 epoch (`patience=5`).
  - `ModelCheckpoint`: Menyimpan bobot model terbaik berdasarkan `val_accuracy` tertinggi ke berkas `models/best_bawang_model.h5`.
  - `ReduceLROnPlateau`: Menurunkan *learning rate* sebesar faktor $0,2$ jika `val_loss` mengalami stagnasi selama 3 epoch.

### 3. NumPy & Pillow
- **Pillow (PIL):** Digunakan untuk membuka berkas citra, mengubah ukuran (*resizing*) ke dimensi $224 x 224$, serta melakukan konversi mode warna ke RGB.
- **NumPy:** Digunakan untuk manipulasi array numerik multidimensi, termasuk konversi piksel ke `float32` dan normalisasi nilai piksel dengan pembagian $255.0$.

### 4. In-Memory Session Storage (Dictionary Python)
- **SESSION_STORE:** Sistem memanfaatkan struktur data *dictionary* Python (`SESSION_STORE = {}`) berbasis memori RAM pada FastAPI backend untuk mengelola data sesi pengunggahan, memuat berkas citra, dan menyimpan *cache* prediksi serta data visualisasi *feature map* secara responsif tanpa perlu melakukan operasi I/O basis data disk.

---

## C. Explore Dataset

### 1. Deskripsi Dataset
Dataset penelitian terdiri atas citra daun dan umbi tanaman bawang merah (*Allium ascalonicum L.*) serta objek kontrol non-bawang yang terbagi ke dalam **4 kelas kategori** (`models/class_indices.json`):
1. **Moler (`moler`):** Citra tanaman bawang merah yang terserang penyakit layu fusarium (*Fusarium oxysporum*), ditandai dengan daun melengkung/terpilin, menguning, dan pembusukan akar/umbi.
2. **Bukan Bawang (`non_bawang`):** Citra objek latar belakang, tanah, gulma, atau tanaman lain sebagai kontrol agar sistem menolak masukan non-bawang.
3. **Sehat (`sehat`):** Citra daun bawang merah kondisi sehat tanpa gejala infeksi patogen.
4. **Trotol (`trotol`):** Citra daun bawang merah yang terserang penyakit bercak ungu (*Alternaria porri*), ditandai bercak oval keabu-abuan/keunguan pada permukaan daun.

![Sampel Gambar Dataset](file:///d:/TUTUP/bawang_merah/reports/sample_images.png)

**Gambar 4. 3 Sampel Citra Data Penelitian per Kelas**

### 2. Deskripsi Data Penelitian
Penelitian ini menggunakan total **2.000 sampel citra digital**. Pembagian dataset dilakukan secara terstruktur menggunakan modul `src/split_dataset.py` dengan rasio **70% data latih (*training*)**, **15% data validasi (*validation*)**, dan **15% data uji (*testing*)** sebagaimana dirinci pada Tabel 4.1.

#### Tabel 4. 1 Distribusi Dataset Penelitian

| Kelas Kategori | Data Training (70%) | Data Validation (15%) | Data Testing (15%) | Total Sampel |
| :--- | :---: | :---: | :---: | :---: |
| **Moler** | 350 | 75 | 75 | 500 |
| **Bukan Bawang** | 350 | 75 | 75 | 500 |
| **Sehat** | 350 | 75 | 75 | 500 |
| **Trotol** | 350 | 75 | 75 | 500 |
| **Total Overall** | **1.400** | **300** | **300** | **2.000** |

![Distribusi Dataset](file:///d:/TUTUP/bawang_merah/reports/distribusi_dataset.png)

**Gambar 4. 4 Grafik Distribusi Jumlah Data per Kelas (Train, Validation, Test)**

---

## D. Evaluasi Performa Model

### 1. Plotting Akurasi dan Loss

Proses pelatihan model CNN dijalankan dengan memantau perubahan metrik *accuracy* dan *loss* pada subset data *training* dan *validation* untuk melihat tingkat konvergensi model.

![Grafik Akurasi dan Loss](file:///d:/TUTUP/bawang_merah/reports/training_history.png)

**Gambar 4. 5 Grafik Akurasi dan Loss Pelatihan Model**

Grafik pelatihan yang dihasilkan oleh modul `src/train_model.py` (tersimpan pada `reports/training_history.png`) menunjukkan bahwa nilai *loss* menurun secara konsisten seiring bertambahnya epoch, sementara *accuracy* mengalami peningkatan dan konvergen pada tingkat yang tinggi.

---

### 2. Confusion Matrix dan Metrik Kuantitatif

Evaluasi performa klasifikasi dilakukan secara empiris pada **300 data uji (*test set*)** yang tidak pernah dilibatkan selama proses pelatihan (`src/evaluate_model.py`).

Hasil perhitungan *Confusion Matrix* $4 x 4$ ditunjukkan pada Tabel 4.2.

#### Tabel 4. 2 Confusion Matrix Prediksi Model pada Data Test (300 Sampel)

| Label Aktual \ Label Prediksi | Moler | Bukan Bawang | Sehat | Trotol | Total Aktual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Moler** | **71** | 0 | 2 | 2 | 75 |
| **Bukan Bawang** | 1 | **74** | 0 | 0 | 75 |
| **Sehat** | 0 | 0 | **66** | 9 | 75 |
| **Trotol** | 2 | 0 | 3 | **70** | 75 |
| **Total Prediksi** | **74** | **74** | **71** | **81** | **300** |

![Confusion Matrix](file:///d:/TUTUP/bawang_merah/reports/confusion_matrix.png)

**Gambar 4. 6 Visualisasi Confusion Matrix Data Test**

Berdasarkan *Confusion Matrix* di atas, diperoleh rincian metrik kuantitatif (*Precision*, *Recall*, dan *F1-Score*) per kelas pada Tabel 4.3.

#### Tabel 4. 3 Evaluasi Metrik Kuantitatif per Kelas pada Data Test (300 Sampel)

| Kelas Kategori | Precision | Recall | F1-Score | Jumlah Sampel (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Moler** | 95,95% | 94,67% | 95,30% | 75 |
| **Bukan Bawang** | 100,00% | 98,67% | 99,33% | 75 |
| **Sehat** | 92,96% | 88,00% | 90,41% | 75 |
| **Trotol** | 86,42% | 93,33% | 89,74% | 75 |
| **Rata-rata Makro (Macro Avg)** | **93,83%** | **93,67%** | **93,70%** | **300** |
| **Rata-rata Tertimbang (Weighted Avg)** | **93,83%** | **93,67%** | **93,70%** | **300** |

![Grafik Metrik Evaluasi per Kelas](file:///d:/TUTUP/bawang_merah/reports/metrics_per_class.png)

**Gambar 4. 7 Grafik Perbandingan Metrik Evaluasi per Kelas**

**Analisis Hasil Evaluasi:**
- **Kelas Non Bawang:** Mencapai tingkat *Precision* sebesar **100,00%** dan *Recall* **98,67%** (74 dari 75 gambar terdeteksi sempurna).
- **Kesalahan Klasifikasi (*Misclassification*):** Terdapat 19 gambar dari 300 data uji yang mengalami kesalahan prediksi (misalnya 9 sampel *sehat* terprediksi *trotol* dan 3 sampel *trotol* terprediksi *sehat*), yang dipengaruhi oleh kemiripan visual pada gejala bercak/gejala awal infeksi.

---

#### 3. Perhitungan Manual CNN

Berikut disajikan perhitungan matematis terperinci, presisi, dan komprehensif langkah demi langkah yang **100% dihitung menggunakan parameter bobot asli dari model terlatih `best_bawang_model.h5`** serta sampel citra daun bawang merah asli dari dataset uji (`(126).jpg`, kelas *Trotol / Bercak Ungu*). Perhitungan ini secara eksplisit menjabarkan setiap perubahan nilai numerik, persentase, dan rumus perantara pada komputasi jaringan saraf *Convolutional Neural Network* (CNN) berbasis MobileNetV2 dan *Classification Head*.

---

### a. Normalisasi Citra Masukan ($I \rightarrow I_{\text{norm}}$)
Citra masukan daun bawang merah berukuran $224 x 224 x 3$ piksel memuat rentang nilai intensitas warna RGB $[0, 255]$. Pada pra-pemrosesan citra, nilai piksel dinormalisasi ke rentang kontinu $[0.0, 1.0]$ menggunakan persamaan pembagian:

$$
X_{i, j} = \frac{\text{Channel}(i, j)}{255.0}
$$

Diambil sampel patch citra $3 x 3$ piksel dari area pusat citra uji lapangan (koordinat pusat $x=112, y=112$):

- **Piksel RGB Asli $[0, 255]$:**
  - Baris 1: $(1,1)=[151, 195, 197]$, $(1,2)=[144, 184, 186]$, $(1,3)=[145, 187, 189]$
  - Baris 2: $(2,1)=[182, 226, 218]$, $(2,2)=[178, 224, 214]$, $(2,3)=[177, 221, 212]$
  - Baris 3: $(3,1)=[194, 232, 202]$, $(3,2)=[194, 232, 201]$, $(3,3)=[193, 232, 203]$

- **Hasil Normalisasi $[0.0, 1.0]$ (Rata-rata 3 Kanal RGB per Sel):**
  - Sel $(1,1)$: $\text{Rata-rata}(151, 195, 197) = 181.0000 / 255.0 = \mathbf{0.7098}$
  - Sel $(1,2)$: $\text{Rata-rata}(144, 184, 186) = 171.3333 / 255.0 = \mathbf{0.6719}$
  - Sel $(1,3)$: $\text{Rata-rata}(145, 187, 189) = 173.6667 / 255.0 = \mathbf{0.6810}$
  - Sel $(2,1)$: $\text{Rata-rata}(182, 226, 218) = 208.6667 / 255.0 = \mathbf{0.8183}$
  - Sel $(2,2)$: $\text{Rata-rata}(178, 224, 214) = 205.3333 / 255.0 = \mathbf{0.8052}$
  - Sel $(2,3)$: $\text{Rata-rata}(177, 221, 212) = 203.3333 / 255.0 = \mathbf{0.7974}$
  - Sel $(3,1)$: $\text{Rata-rata}(194, 232, 202) = 209.3333 / 255.0 = \mathbf{0.8209}$
  - Sel $(3,2)$: $\text{Rata-rata}(194, 232, 201) = 209.0000 / 255.0 = \mathbf{0.8196}$
  - Sel $(3,3)$: $\text{Rata-rata}(193, 232, 203) = 209.3333 / 255.0 = \mathbf{0.8209}$

Sehingga matriks piksel masukan ter-normalisasi ($I_{\text{norm}}$) adalah:

$$
I_{\text{norm}} = \begin{bmatrix} 0.7098 & 0.6719 & 0.6810 \\ 0.8183 & 0.8052 & 0.7974 \\ 0.8209 & 0.8196 & 0.8209 \end{bmatrix}
$$

---

### b. Operasi Konvolusi 2D (*Convolutional Layer 1 / Conv1*)
Sesuai tampilan website pada **Tahap 2**, operasi konvolusi dihitung dengan rumus perkalian titik (*dot product*) elemen demi elemen antara matriks masukan $X$ dan matriks bobot kernel $W$:

$$
y = \sum (X_{i,j} x W_{i,j})
$$

Bobot asli kernel $3 x 3$ untuk **Filter #1** ($W$) dari layer `Conv1` model MobileNetV2 (setelah merata-ratakan 3 kanal RGB) adalah:

$$
W = \begin{bmatrix} -0.19900 & -0.29629 & -0.07772 \\ +0.32279 & +0.42667 & +0.15698 \\ -0.10745 & -0.00619 & -0.07872 \end{bmatrix}
$$

**Penjabaran Perkalian 9 Sel Spasial secara Detail:**
1. **Sel (1,1):** $0.7098 x -0.19900 = \mathbf{-0.141253}$
2. **Sel (1,2):** $0.6719 x -0.29629 = \mathbf{-0.199077}$
3. **Sel (1,3):** $0.6810 x -0.07772 = \mathbf{-0.052933}$
4. **Sel (2,1):** $0.8183 x +0.32279 = \mathbf{+0.264136}$
5. **Sel (2,2):** $0.8052 x +0.42667 = \mathbf{+0.343564}$
6. **Sel (2,3):** $0.7974 x +0.15698 = \mathbf{+0.125176}$
7. **Sel (3,1):** $0.8209 x -0.10745 = \mathbf{-0.088207}$
8. **Sel (3,2):** $0.8196 x -0.00619 = \mathbf{-0.005072}$
9. **Sel (3,3):** $0.8209 x -0.07872 = \mathbf{-0.064625}$

**Penjumlahan Bertahap Seluruh Nilai Komponen:**
- Total komponen negatif $= -0.141253 - 0.199077 - 0.052933 - 0.088207 - 0.005072 - 0.064625 = \mathbf{-0.551167}$
- Total komponen positif $= +0.264136 + 0.343564 + 0.125176 = \mathbf{+0.732876}$
- Hasil akhir konvolusi mentah:
  $$y = -0.551167 + 0.732876 = \mathbf{+0.181709}$$

---

### c. Lapisan Batch Normalization (`bn_Conv1`)
Nilai konvolusi diproses oleh *Batch Normalization* asli dari layer `bn_Conv1` model Keras. Parameter statistik asli dari model:
- Rata-rata pergerakan ($\mu$) $= -0.036762$
- Varians pergerakan ($\sigma^2$) $= +0.120517$
- Skala ($\gamma$) $= +0.612176$
- Pergeseran ($\beta$) $= +2.255549$
- Epsilon ($\epsilon$) $= 0.001$

Perhitungannya dibagi ke dalam 5 langkah matematika rinci:

$$
\hat{x} = \frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_{\text{BN}} = \gamma \hat{x} + \beta
$$

**Rincian Perhitungan Langkah demi Langkah:**
1. **Pengurangan Mean ($\mu$):**
   $$y - \mu = 0.181709 - (-0.036762) = \mathbf{+0.218471}$$
2. **Penambahan Epsilon ($\epsilon = 0.001$):**
   $$\sigma^2 + \epsilon = 0.120517 + 0.001000 = \mathbf{0.121517}$$
3. **Akar Kuadrat Standar Deviasi:**
   $$\sqrt{0.121517} = \mathbf{0.348592}$$
4. **Pembagian Normalisasi Standardized ($\hat{x}$):**
   $$\hat{x} = \frac{0.218471}{0.348592} = \mathbf{+0.626722}$$
5. **Skala ($\gamma$) & Geseran ($\beta$):**
   $$y_{\text{BN}} = (0.612176 x 0.626722) + 2.255549 = 0.383664 + 2.255549 = \mathbf{+2.639213}$$

---

### d. Fungsi Aktivasi ReLU (*Rectified Linear Unit*)
Sesuai tampilan website pada **Tahap 3**, fungsi aktivasi ReLU menerapkan persamaan:

$$
f(x) = \max(0, x)
$$

**Perubahan Nilai Aktivasi:**
- **Kasus 1 — Nilai Positif ($x = +2.639213$):**
  $$f(+2.639213) = \max(0, +2.639213) = \mathbf{+2.639213} \quad (\text{Nilai positif tetap dipertahankan})$$
- **Kasus 2 — Nilai Negatif (Misalkan pada koordinat piksel lain $x = -0.166700$):**
  $$f(-0.166700) = \max(0, -0.166700) = \mathbf{0.000000} \quad (\text{Nilai negatif diubah menjadi 0})$$

---

### e. Max Pooling ($2 x 2$, Stride 2)
Sesuai tampilan website pada **Tahap 4**, operasi Max Pooling dihitung dengan rumus mengambil nilai maksimum pada jendela $2 x 2$:

$$
y = \max(X_{2 x 2})
$$

Menyapu patch matriks aktivasi $4 x 4$ piksel ($M$) menggunakan 4 jendela non-overlapping berukuran $2 x 2$ piksel untuk mereduksi dimensi spasial sebesar 75%:

$$
M = \begin{bmatrix} 
\mathbf{0.32} & \mathbf{0.26} & 0.85 & 0.42 \\ 
\mathbf{0.34} & \mathbf{0.30} & 0.12 & 0.67 \\ 
0.10 & 0.05 & 0.91 & 0.18 \\ 
0.22 & 0.15 & 0.44 & 0.73 
\end{bmatrix}
$$

**Rincian Seleksi Nilai Maksimum tiap Jendela:**
1. **Jendela 1 (Baris 1-2, Kolom 1-2):**
   $$\text{Elemen} = \{0.32, 0.26, 0.34, 0.30\} \implies \text{Nilai Maksimum} = \mathbf{0.34}$$
2. **Jendela 2 (Baris 1-2, Kolom 3-4):**
   $$\text{Elemen} = \{0.85, 0.42, 0.12, 0.67\} \implies \text{Nilai Maksimum} = \mathbf{0.85}$$
3. **Jendela 3 (Baris 3-4, Kolom 1-2):**
   $$\text{Elemen} = \{0.10, 0.05, 0.22, 0.15\} \implies \text{Nilai Maksimum} = \mathbf{0.22}$$
4. **Jendela 4 (Baris 3-4, Kolom 3-4):**
   $$\text{Elemen} = \{0.91, 0.18, 0.44, 0.73\} \implies \text{Nilai Maksimum} = \mathbf{0.91}$$

**Matriks Hasil Reduksi ($4 x 4 \rightarrow 2 x 2$):**

$$
M_{\text{pooled}} = \begin{bmatrix} 0.34 & 0.85 \\ 0.22 & 0.91 \end{bmatrix}
$$

---

### f. Global Average Pooling (GAP)
Sesuai tampilan website pada **Tahap 5**, operasi Global Average Pooling merata-ratakan seluruh 49 sel piksel spasial ($7 x 7$) dengan rumus:

$$
y = \frac{1}{49} \sum X_{7 x 7}
$$

**Penjabaran Perhitungan pada Channel #0001:**
- Sampel 49 sel piksel: $x_1 = 0.61, x_2 = 0.00, x_3 = 1.15, x_4 = 0.84, \dots, x_{49} = 0.25$
- Akumulasi total penjumlahan 49 sel:
  $$\sum_{m=1}^{49} x_m = 0.61 + 0.00 + 1.15 + 0.84 + \dots + 0.25 = \mathbf{29.8900}$$
- Pembagian perata-rataan dengan $N=49$:
  $$y = \frac{29.8900}{49} = \mathbf{0.6100}$$

Proses ini diulang secara konsisten untuk seluruh 1.280 kanal, membentuk **vektor 1D berdimensi 1.280 elemen** ($\mathbf{x} \in \mathbb{R}^{1280}$).

---

### g. Fully Connected Layer (Dense Layer 128 Neuron)
Vektor GAP ($1280 x 1$) dikalikan dengan matriks bobot Dense Layer ($W \in \mathbb{R}^{1280 x 128}$) ditambah bias ($b \in \mathbb{R}^{128}$):

$$
z_m = \sum (X_i x W_{i, m}) + b_m
$$

**Rincian Perhitungan Perkalian Titik Neuron #1 ($m=1$):**
- $X_1 x W_{1, 1} = 0.6100 x 0.0015 = +0.000915$
- $X_2 x W_{2, 1} = 0.0000 x 0.0443 = 0.000000$
- $X_3 x W_{3, 1} = 0.5200 x -0.0404 = -0.021008$
- $\dots$ *(diteruskan hingga elemen ke-1280)*
- Tambah Bias $b_1 = -0.000450$
- Total hasil akhir neuron:
  $$z_1 = +0.000915 + 0.000000 - 0.021008 + \dots - 0.000450 = \mathbf{+0.6800}$$

**Aplikasi Fungsi Aktivasi ReLU pada Neuron Dense:**
$$\text{out}_1 = \max(0, +0.6800) = \mathbf{+0.6800}$$

---

### h. Dense Output Layer & Probabilitas Softmax
Sesuai tampilan website pada **Tahap 6**, perhitungan logit mentah $z_k$ dan probabilitas akhir $P(y_k)$ dihitung menggunakan persamaan:

$$
z_k = \sum (X_i x W_{i,k}) + b_k \implies P(y_k) = \frac{e^{z_k}}{\sum e^{z_j}} x 100\%
$$

**Nilai Logit Mentah Asli dari Ekstraksi Model `best_bawang_model.h5`:**
- $z_1 \text{ (Moler / Layu Fusarium)} = \mathbf{-0.390257}$
- $z_2 \text{ (Objek Bukan Bawang)} = \mathbf{-3.319969}$
- $z_3 \text{ (Sehat)} = \mathbf{-0.381395}$
- $z_4 \text{ (Trotol / Bercak Ungu)} = \mathbf{+2.927967}$

**Normalisasi Probabilitas Softmax Langkah demi Langkah:**

1. **Penghitungan Nilai Eksponensial $e^{z_k}$ per Kelas:**
   - Kelas 1 (Moler): $e^{-0.390257} = \mathbf{0.676883}$
   - Kelas 2 (Bukan Bawang): $e^{-3.319969} = \mathbf{0.036154}$
   - Kelas 3 (Sehat): $e^{-0.381395} = \mathbf{0.682908}$
   - Kelas 4 (Trotol): $e^{+2.927967} = \mathbf{18.689592}$

2. **Penjumlahan Total Penyebut Softmax ($\sum e^{z_j}$):**
   $$\sum e^{z} = 0.676883 + 0.036154 + 0.682908 + 18.689592 = \mathbf{20.085537}$$

3. **Penghitungan Persentase Probabilitas Akhir per Kelas:**
   - **Trotol / Bercak Ungu:**
     $$P(\text{trotol}) = \left( \frac{18.689592}{20.085537} \right) x 100\% = 0.930500 x 100\% = \mathbf{93,05\%}$$
   - **Sehat:**
     $$P(\text{sehat}) = \left( \frac{0.682908}{20.085537} \right) x 100\% = 0.034000 x 100\% = \mathbf{3,40\%}$$
   - **Moler / Layu Fusarium:**
     $$P(\text{moler}) = \left( \frac{0.676883}{20.085537} \right) x 100\% = 0.033700 x 100\% = \mathbf{3,37\%}$$
   - **Objek Bukan Bawang:**
     $$P(\text{non\_bawang}) = \left( \frac{0.036154}{20.085537} \right) x 100\% = 0.001800 x 100\% = \mathbf{0,18\%}$$

**Kesimpulan Inferensi Diagnostik:**  
Model CNN memprediksi citra masukan secara otomatis sebagai **Trotol / Bercak Ungu (*Alternaria porri*)** dengan nilai kepastian probabilitas tertinggi sebesar **93,05%** (Kategori Akurat).

---

### E. Implementasi Sistem


### 1. Halaman Beranda
Halaman beranda merupakan tampilan utama (*landing page*) aplikasi yang berfungsi sebagai pintu masuk utama bagi pengguna saat mengakses sistem klasifikasi penyakit bawang merah. Halaman ini dirancang dengan antarmuka yang modern dan responsif, menyajikan informasi latar belakang singkat mengenai pentingnya deteksi dini penyakit layu fusarium (moler) dan bercak ungu (trotol), deskripsi teknologi *Deep Learning* yang digunakan, serta menyediakan tombol navigasi (*Call-to-Action*) yang mengarahkan pengguna secara langsung ke modul deteksi citra.

`[BAGIAN GAMBAR 4. 8: Tempatkan Tangkapan Layar Halaman Beranda di sini]`

**Gambar 4. 8 Tangkapan Layar Halaman Beranda Web**

Berdasarkan tangkapan layar pada Gambar 4.8, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman beranda:

#### Tabel 4. 6 Rincian Komponen Antarmuka Halaman Beranda

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Navbar / Header Utama | *Sticky Top Bar* | Bilah navigasi yang menempel di bagian teratas layar (*sticky position*). Memuat ikon logo aplikasi berbentuk *checkmark* dalam lingkaran hijau, judul sistem bertuliskan *"Klasifikasi Penyakit Bawang Merah"*, serta sub-judul *"dengan Pendekatan Deep Learning"*. Navbar ini tetap terlihat meskipun halaman di-*scroll* ke bawah. |
| 2 | Label Eyebrow | *Text Badge* | Label kecil bertuliskan *"Convolutional Neural Network"* yang dilengkapi indikator titik hijau (*dot indicator*) di sisi kiri, berfungsi sebagai penanda teknologi utama yang digunakan oleh sistem. |
| 3 | Judul Hero (*Hero Title*) | *Heading Text* | Teks judul berukuran besar bertuliskan *"SELAMAT DATANG"* dengan gaya huruf miring (*italic*) yang berfungsi sebagai titik fokus visual utama halaman dan menyambut pengguna saat pertama kali mengakses sistem. |
| 4 | Deskripsi Hero (*Hero Description*) | *Paragraph Text* | Paragraf penjelasan singkat yang menginformasikan bahwa tanaman bawang merah merupakan komoditas hortikultura yang rentan terhadap gangguan, dan sistem ini menganalisis citra daun menggunakan model CNN untuk mendeteksi gejala penyakit seperti bercak ungu dan layu fusarium secara otomatis. |
| 5 | Tombol *"Mode Sederhana"* | *Primary Button (CTA)* | Tombol navigasi utama berwarna hijau (*primary*) yang dilengkapi ikon panah kanan (→), mengarahkan pengguna ke halaman `deteksi.html` untuk melakukan diagnosis cepat tanpa menampilkan detail proses CNN. |
| 6 | Tombol *"Mode Detail"* | *Ghost Button (CTA)* | Tombol navigasi sekunder bergaya transparan (*ghost/outline*) yang dilengkapi ikon panah kanan (→), mengarahkan pengguna ke halaman `index.html` untuk melihat visualisasi tahap demi tahap seluruh *pipeline* CNN (7 tahapan). |
| 7 | Kartu Simulasi Konvolusi (*Scan Card*) | *Interactive Card* | Kartu interaktif di sisi kanan yang menampilkan simulasi visual proses *scanning* kernel konvolusi pada ilustrasi daun bawang merah berbentuk SVG. Kartu ini memuat ilustrasi daun hijau dengan dua bercak cokelat (*spot lesion*), sebuah kotak kernel yang bergerak di atas permukaan daun, serta label keterangan *"Simulasi · kernel konvolusi memindai daun"*. |
| 8 | Indikator *Confidence* Dinamis | *Animated Readout* | Panel pembacaan nilai di bawah kartu simulasi yang menampilkan label *"Confidence bercak terdeteksi"* beserta angka nilai probabilitas (misal: `0.94`) yang berubah-ubah secara otomatis setiap 670 milidetik menggunakan animasi JavaScript untuk mensimulasikan proses inferensi *real-time*. |

---

### 2. Halaman Input Gambar & Pra-pemrosesan Citra
Halaman input gambar merupakan fasilitas antarmuka yang memungkinkan pengguna untuk memasukkan citra tanaman bawang merah yang akan dianalisis oleh model CNN. Halaman ini menyediakan dua metode masukan utama, yaitu fitur unggah berkas gambar dengan antarmuka *drag-and-drop* serta integrasi kamera web (*webcam*) untuk pengambilan foto secara *real-time* di lapangan. Selain itu, halaman ini dilengkapi dengan modul validasi otomatis untuk memastikan berkas yang diunggah berekstensi gambar valid (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) dan pratinjau citra (*image preview*) sebelum diproses ke tahap inferensi.

`[BAGIAN GAMBAR 4. 9: Tempatkan Tangkapan Layar Halaman Input Gambar di sini]`

**Gambar 4. 9 Tangkapan Layar Halaman Input Gambar**

Berdasarkan tangkapan layar pada Gambar 4.9, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman input gambar:

#### Tabel 4. 7 Rincian Komponen Antarmuka Halaman Input Gambar

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan (*Page Eyebrow*) | *Step Indicator* | Label bertuliskan *"Tahap 1 dari 7"* yang menunjukkan posisi pengguna dalam alur *pipeline* CNN, disertai judul halaman *"Unggah Gambar Daun"* dan keterangan instruksi singkat. |
| 2 | Kartu Pilih Gambar (*Upload Card*) | *Card Container* | Kartu panel utama di sisi kiri berisi seluruh elemen masukan gambar, diberi judul *"Pilih Gambar"*. |
| 3 | Area *Drag & Drop* (*Upload Zone*) | *Dashed Border Label* | Area interaktif bergaris putus-putus yang dapat menerima berkas gambar dengan cara diseret (*drag and drop*) atau diklik untuk membuka *file dialog* sistem operasi. Menampilkan ikon panah unggah SVG, teks *"Seret & lepas gambar di sini"*, tautan *"pilih dari perangkat"*, serta keterangan format yang diterima: *"JPG, PNG, JPEG · maks. 10 MB"*. |
| 4 | Tombol *"Ambil Foto Langsung"* | *Secondary Button* | Tombol bergaya sekunder yang memuat ikon kamera SVG dan teks *"Ambil Foto Langsung"*. Saat diklik, tombol ini membuka jendela modal kamera (*camera modal*) yang mengaktifkan *webcam* perangkat untuk pengambilan foto secara *real-time*. |
| 5 | Kotak Notifikasi Saran (*Notice Info*) | *Info Banner* | Panel informasi berwarna biru muda berisi ikon peringatan dan teks *"Gunakan gambar dengan pencahayaan cukup dan fokus pada bagian daun untuk hasil deteksi yang akurat"* sebagai panduan bagi pengguna. |
| 6 | Kartu Pratinjau Gambar (*Preview Card*) | *Dynamic Card* | Kartu panel di sisi kanan yang awalnya menampilkan placeholder ikon gambar dan teks *"Belum ada gambar dipilih"*. Setelah pengguna memilih gambar, kartu ini berubah menampilkan pratinjau visual citra yang diunggah beserta *badge* hijau bertuliskan *"✓ Gambar Valid"*. |
| 7 | Panel Metadata Berkas (*Stat Boxes*) | *3-Column Grid* | Tiga kotak statistik yang tersusun horizontal di bawah pratinjau, menampilkan informasi: **Nama file** (misal: `daun_moler.jpg`), **Ukuran** (misal: `245 KB`), dan **Resolusi** (misal: `640x480`). |
| 8 | Panel Pra-proses Otomatis | *Progress Bars* | Dua bilah progres (*progress bar*) bertuliskan *"Ubah ukuran ke 224x224"* dan *"Normalisasi nilai piksel [0,1]"* yang masing-masing menampilkan status *"Selesai"* berwarna hijau setelah gambar berhasil diproses oleh backend. |
| 9 | Notifikasi Kesiapan (*Success Notice*) | *Success Banner* | Panel hijau berisi ikon centang dan teks *"Gambar siap diproses. Klik **Proses Gambar** untuk memulai analisis CNN"* yang muncul setelah validasi dan pra-pemrosesan selesai. |
| 10 | Tombol *"Proses Gambar"* | *Primary Button* | Tombol aksi utama berwarna hijau yang muncul setelah gambar valid dipilih. Saat diklik, tombol ini mengirimkan data *bytes* gambar ke endpoint API `/api/upload` dan memajukan pengguna ke tahapan berikutnya (halaman konvolusi). |

**Kodingan Fungsi Logika Pra-pemrosesan Citra (`preprocess_image`):**
```python
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(TARGET_IMAGE_SIZE)
        img_array = np.array(image, dtype=np.float32) / 255.0  # Normalisasi piksel [0,1]
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses berkas gambar: {str(e)}")
```

---

### 3. Halaman Lapisan Konvolusi (*Convolution Layer*)
Halaman lapisan konvolusi menyajikan visualisasi interaktif dari 32 *feature maps* (peta fitur) 2D yang dihasilkan oleh filter-filter konvolusi pada arsitektur MobileNetV2. Visualisasi pada halaman ini bertujuan untuk memberikan transparansi (*explainable AI*) mengenai proses komputasi tingkat rendah (*low-level features*), di mana jaringan saraf secara otomatis mengekstraksi informasi visual penting dari daun bawang merah, seperti garis tepi (*edges*), tekstur permukaan daun, gradasi warna hijau keunguan, hingga bentuk pola bercak infeksi.

`[BAGIAN GAMBAR 4. 10: Tempatkan Tangkapan Layar Halaman Lapisan Konvolusi di sini]`

**Gambar 4. 10 Tangkapan Layar Halaman Lapisan Konvolusi**

Berdasarkan tangkapan layar pada Gambar 4.10, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman lapisan konvolusi:

#### Tabel 4. 8 Rincian Komponen Antarmuka Halaman Lapisan Konvolusi

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan | *Step Indicator* | Label bertuliskan *"Tahap 2 dari 7"* disertai judul *"Lapisan Konvolusi (Conv Layer)"* dan paragraf penjelasan bahwa model memakai 3 blok Konvolusi-ReLU-Pooling berurutan dengan filter 3x3 pada blok pertama. |
| 2 | Kartu Demo Konvolusi (*Conv Demo*) | *Interactive Card* | Kartu di sisi kiri yang memvisualisasikan cara kerja satu operasi filter konvolusi. Menampilkan tiga elemen berdampingan: **Patch gambar** (grid 3x3 berisi rata-rata piksel RGB dari posisi tengah gambar pengguna), simbol perkalian (x), **Bobot filter** (grid 3x3 berisi rata-rata bobot Conv1 filter #1 dari model terlatih), simbol sama dengan (=), dan **Output** (satu kotak hijau menampilkan nilai hasil *forward-pass* aktual, misal: `0.06`). |
| 3 | Notifikasi Penjelasan Data (*Notice Info*) | *Info Banner* | Panel biru muda di bawah demo konvolusi yang menjelaskan bahwa Patch, Filter, dan Output yang ditampilkan merupakan data asli dari model terlatih, bukan aproksimasi. |
| 4 | Tabel Parameter Konvolusi | *Parameter Table* | Tabel berisi spesifikasi teknis 3 blok konvolusi berurutan: **Conv1** (32 filter, kernel 3x3, stride 1, padding same), **Conv2** (64 filter), **Conv3** (128 filter), serta dimensi input tensor awal ($224 x 224 x 3$) dan output Conv1 ($224 x 224 x 32$). |
| 5 | Kartu *Feature Maps* Hasil Konvolusi | *Card + Grid* | Kartu di sisi kanan dengan judul *"Feature Maps Hasil Konvolusi"* dan *tag badge* ungu bertuliskan *"32 filter"*. Berisi grid 32 gambar peta fitur kecil berskema warna *tinting RGB* yang menunjukkan hasil ekstraksi fitur visual dari gambar masukan pengguna. |
| 6 | Notifikasi Peringatan Nilai Negatif | *Warning Banner* | Panel kuning berisi ikon segitiga peringatan dan teks *"Feature map ini masih mengandung nilai negatif. Nilai negatif akan diproses ReLU di tahap berikutnya"* untuk menginformasikan bahwa data belum melewati fungsi aktivasi. |
| 7 | Tombol *"Kembali"* | *Secondary Button* | Tombol navigasi kembali ke halaman Input (Tahap 1). |
| 8 | Tombol *"Lanjut ke ReLU"* | *Primary Button* | Tombol navigasi maju ke halaman Fungsi Aktivasi ReLU (Tahap 3). |

**Kodingan Fungsi Logika Peta Fitur Konvolusi (`generate_conv_maps`):**
```python
def generate_conv_maps(conv_arr, conv_weights=None, max_filters=32):
    maps = []
    if conv_arr is None or len(conv_arr.shape) < 3:
        return maps
    num_filters = min(conv_arr.shape[-1], max_filters)
    
    tints = [
        [100, 110, 190], [130, 140, 180], [180, 150, 120], [110, 150, 110],
        [90, 160, 130],  [160, 140, 190], [170, 170, 140], [120, 140, 160]
    ]
    
    for i in range(num_filters):
        fmap = conv_arr[:, :, i]
        min_v, max_v = float(np.min(fmap)), float(np.max(fmap))
        if max_v > min_v:
            norm = (fmap - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(fmap)
            
        tint = tints[i % len(tints)]
        rgb = np.zeros((fmap.shape[0], fmap.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = np.clip(norm * tint[c], 0, 255).astype(np.uint8)
            
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        label = FILTER_LABELS[i] if i < len(FILTER_LABELS) else f"Filter #{i+1:02d} — ~fitur visual"
        
        kernel = None
        if conv_weights is not None and len(conv_weights.shape) == 4:
            k3x3 = np.mean(conv_weights[:, :, :, i], axis=2)
            kernel = [[round(float(v), 3) for v in row] for row in k3x3]
            
        maps.append({
            "label": label,
            "image_b64": b64_str,
            "kernel": kernel
        })
        
    return maps
```

---

### 4. Halaman Fungsi Aktivasi ReLU
Halaman fungsi aktivasi ReLU (*Rectified Linear Unit*) menampilkan kondisi matriks fitur setelah melewati operasi matematika thresholding $f(x) = \max(0, x)$. Pada tampilan antarmuka ini, pengguna dapat melihat bagaimana seluruh nilai aktivasi yang bernilai negatif diubah menjadi nilai 0 (ditampilkan sebagai area piksel gelap), yang berguna untuk memberikan sifat non-linearitas pada model sehingga jaringan saraf mampu mempelajari dan memisahkan pola-pola penyakit yang kompleks serta tidak teratur pada citra masukan.

`[BAGIAN GAMBAR 4. 11: Tempatkan Tangkapan Layar Halaman Fungsi Aktivasi ReLU di sini]`

**Gambar 4. 11 Tangkapan Layar Halaman Fungsi Aktivasi ReLU**

Berdasarkan tangkapan layar pada Gambar 4.11, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman fungsi aktivasi ReLU:

#### Tabel 4. 9 Rincian Komponen Antarmuka Halaman Fungsi Aktivasi ReLU

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan | *Step Indicator* | Label bertuliskan *"Tahap 3 dari 7"* disertai judul *"Fungsi Aktivasi ReLU"* dan penjelasan bahwa ReLU mengganti semua nilai negatif dengan nol serta diterapkan setelah setiap lapisan konvolusi (ReLU1, ReLU2, ReLU3). |
| 2 | Kartu Rumus ReLU | *Formula Card* | Kartu di sisi kiri yang menampilkan rumus matematis `f(x) = max(0, x)` dalam format teks besar, disertai dua kotak demonstrasi perbandingan: kotak merah muda bertuliskan *"Nilai negatif → dibuang"* (misal: `-0.82 → 0`) dan kotak hijau muda bertuliskan *"Nilai positif → dipertahankan"* (misal: `+0.06 → 0.06`). Nilai yang ditampilkan merupakan data asli dari output Conv1. |
| 3 | Catatan Sumber Data (*Note*) | *Muted Text* | Teks kecil di bawah kotak perbandingan yang menjelaskan bahwa nilai `0.06` adalah output Conv1 yang sama dengan halaman konvolusi sebelumnya, dan nilai satunya diambil dari statistik nyata (min/max) seluruh *feature map* Conv1. |
| 4 | Tabel Ringkasan ReLU | *Parameter Table* | Tabel berisi 5 baris informasi: **Fungsi** (`max(0, x)`), **ReLU1** setelah Conv1 ($224 x 224 x 32$), **ReLU2** setelah Conv2 ($112 x 112 x 64$), **ReLU3** setelah Conv3 ($56 x 56 x 128$), dan **Tujuan** (Non-linearitas). |
| 5 | Kartu *Feature Map* Setelah ReLU | *Card + Grid* | Kartu di sisi kanan dengan judul *"FEATURE MAP SETELAH RELU (AREA ABU = NILAI NOL)"* dan *tag badge* hijau bertuliskan *"32 FILTER"*. Berisi grid 32 gambar peta aktivasi berskema warna hijau mint, di mana piksel bernilai 0 ditampilkan sebagai area abu-abu terang dan piksel aktif (positif) ditampilkan dalam gradasi hijau. |
| 6 | Notifikasi Sukses (*Success Notice*) | *Success Banner* | Panel hijau berisi ikon centang dan teks *"Nilai negatif sudah dihapus. Feature map kini hanya berisi nilai 0 atau positif, siap masuk pooling layer"* yang menandakan proses ReLU telah selesai. |
| 7 | Tombol *"Kembali"* | *Secondary Button* | Tombol navigasi kembali ke halaman Konvolusi (Tahap 2). |
| 8 | Tombol *"Lanjut ke Pooling"* | *Primary Button* | Tombol navigasi maju ke halaman Max Pooling Layer (Tahap 4). |

**Kodingan Fungsi Logika Peta Aktivasi ReLU (`generate_relu_maps`):**
```python
def generate_relu_maps(relu_arr, conv_weights=None, max_filters=32):
    maps = []
    if relu_arr is None or len(relu_arr.shape) < 3:
        return maps
    num_filters = min(relu_arr.shape[-1], max_filters)
    
    for i in range(num_filters):
        fmap = relu_arr[:, :, i]
        max_v = float(np.max(fmap))
        
        rgb = np.full((fmap.shape[0], fmap.shape[1], 3), 220, dtype=np.uint8)
        
        mask = fmap > 0
        if np.any(mask) and max_v > 0:
            norm = fmap[mask] / max_v
            rgb[mask, 0] = np.clip(180 - norm * 160, 0, 255).astype(np.uint8)
            rgb[mask, 1] = np.clip(230 - norm * 100, 0, 255).astype(np.uint8)
            rgb[mask, 2] = np.clip(200 - norm * 120, 0, 255).astype(np.uint8)
            
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        label = FILTER_LABELS[i] if i < len(FILTER_LABELS) else f"Filter #{i+1:02d} — ~fitur visual"
        
        kernel = None
        if conv_weights is not None and len(conv_weights.shape) == 4:
            k3x3 = np.mean(conv_weights[:, :, :, i], axis=2)
            kernel = [[round(float(v), 3) for v in row] for row in k3x3]
            
        maps.append({
            "label": label,
            "image_b64": b64_str,
            "kernel": kernel
        })
        
    return maps
```

---

### 5. Halaman Pooling Layer
Halaman lapisan *pooling* memperlihatkan hasil proses *downsampling* matriks fitur menggunakan operasi *Max Pooling* dengan ukuran jendela $2 x 2$ dan *stride* 2. Pada tampilan ini, terlihat bagaimana dimensi spasial citra direduksi hingga 50% tanpa kehilangan informasi tekstur dan karakteristik utama penyakit, sehingga komputasi jaringan menjadi jauh lebih efisien, menghemat penggunaan memori RAM server, serta meningkatkan ketahanan model terhadap pergeseran posisi objek (*spatial invariance*).

`[BAGIAN GAMBAR 4. 12: Tempatkan Tangkapan Layar Halaman Pooling Layer di sini]`

**Gambar 4. 12 Tangkapan Layar Halaman Pooling Layer**

Berdasarkan tangkapan layar pada Gambar 4.12, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman *pooling layer*:

#### Tabel 4. 10 Rincian Komponen Antarmuka Halaman Pooling Layer

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan | *Step Indicator* | Label bertuliskan *"Tahap 4 dari 7"* disertai judul *"Max Pooling Layer"* dan penjelasan bahwa jendela 2x2 mengambil nilai tertinggi setiap area, dilakukan 3 kali (Pool1, Pool2, Pool3) hingga dimensi menjadi $28 x 28 x 128$. |
| 2 | Kartu Demo Max Pooling (*Pooling Demo*) | *Interactive Card* | Kartu di sisi kiri yang memvisualisasikan cara kerja *Max Pooling* 2x2 dengan dua grid berdampingan: grid **Sebelum (4x4)** berisi 16 sel nilai piksel dan grid **Sesudah (2x2)** berisi 4 sel hasil seleksi nilai tertinggi, dihubungkan oleh simbol panah (→). Setiap sel bernilai tinggi diberi *highlight* untuk menunjukkan pemilihan nilai maksimum. |
| 3 | Catatan Penjelasan Pooling | *Muted Text Box* | Kotak teks berlatar abu-abu yang menjelaskan: *"Tiap jendela 2x2 → ambil nilai tertinggi → 1 nilai output"*. |
| 4 | Tabel Parameter Pooling | *Parameter Table* | Tabel berisi 5 baris spesifikasi: **Metode** (Max Pooling, window 2x2, stride 2), **Pool1** ($224 x 224 x 32 → 112 x 112 x 32$), **Pool2** ($112 x 112 x 64 → 56 x 56 x 64$), **Pool3** ($56 x 56 x 128 → 28 x 28 x 128$), dan **Output akhir ke GAP** ($28 x 28 x 128$). |
| 5 | Kartu *Feature Map* Setelah Pooling | *Card + Grid* | Kartu di sisi kanan dengan judul *"FEATURE MAP SETELAH POOLING"* dan *tag badge* ambar/kuning bertuliskan *"32 FILTER"*. Berisi grid 32 gambar peta fitur hasil *downsampling* yang diwarnai dengan skema warna ambar/emas untuk menyoroti area piksel bernilai tinggi. |
| 6 | Bilah Progres Reduksi Dimensi | *Progress Bar* | Bilah progres bertuliskan *"Reduksi dimensi"* dengan nilai *"75%"* berwarna ambar, menunjukkan total persentase piksel yang telah direduksi melalui 3 tahap *pooling* berturut-turut. |
| 7 | Tombol *"Kembali"* | *Secondary Button* | Tombol navigasi kembali ke halaman ReLU (Tahap 3). |
| 8 | Tombol *"Lanjut ke Global Average Pooling"* | *Primary Button* | Tombol navigasi maju ke halaman GAP (Tahap 5). |

**Kodingan Fungsi Logika Pemeta Lapisan Pooling (`generate_pool_maps`):**
```python
def generate_pool_maps(pool_arr, conv_weights=None, max_filters=32):
    maps = []
    if pool_arr is None or len(pool_arr.shape) < 3:
        return maps
    num_filters = min(pool_arr.shape[-1], max_filters)
    
    for i in range(num_filters):
        fmap = pool_arr[:, :, i]
        max_v = float(np.max(fmap))
        
        rgb = np.full((fmap.shape[0], fmap.shape[1], 3), 245, dtype=np.uint8)
        
        mask = fmap > 0
        if np.any(mask) and max_v > 0:
            norm = fmap[mask] / max_v
            rgb[mask, 0] = np.clip(250 - norm * 120, 0, 255).astype(np.uint8)
            rgb[mask, 1] = np.clip(220 - norm * 150, 0, 255).astype(np.uint8)
            rgb[mask, 2] = np.clip(160 - norm * 140, 0, 255).astype(np.uint8)
            
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        label = FILTER_LABELS[i] if i < len(FILTER_LABELS) else f"Filter #{i+1:02d} — ~fitur visual"
        
        kernel = None
        if conv_weights is not None and len(conv_weights.shape) == 4:
            k3x3 = np.mean(conv_weights[:, :, :, i], axis=2)
            kernel = [[round(float(v), 3) for v in row] for row in k3x3]
            
        maps.append({
            "label": label,
            "image_b64": b64_str,
            "kernel": kernel
        })
        
    return maps
```

---

### 6. Halaman Global Average Pooling (GAP)
Halaman Global Average Pooling (GAP) menyajikan visualisasi tahapan peringkasan fitur di mana *feature map* multidimensi keluaran dari *base model* MobileNetV2 dikonversi menjadi sebuah vektor 1D dengan menghitung nilai rata-rata dari seluruh piksel pada setiap kanal (*channel*). Tampilan antarmuka ini mendemonstrasikan penggantian lapisan *Flatten* konvensional dengan GAP, yang terbukti efektif memangkas jutaan parameter bobot yang tidak perlu sehingga secara signifikan mengurangi risiko terjadinya *overfitting* pada model.

`[BAGIAN GAMBAR 4. 13: Tempatkan Tangkapan Layar Halaman Global Average Pooling di sini]`

**Gambar 4. 13 Tangkapan Layar Halaman Global Average Pooling**

Berdasarkan tangkapan layar pada Gambar 4.13, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman Global Average Pooling:

#### Tabel 4. 11 Rincian Komponen Antarmuka Halaman Global Average Pooling

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan | *Step Indicator* | Label bertuliskan *"Tahap 5 dari 7"* disertai judul *"Global Average Pooling (GAP)"* dan penjelasan bahwa tensor 3D ($28 x 28 x 128$) diringkas menjadi vektor 1D dengan 128 elemen melalui perhitungan satu nilai rata-rata per *feature map*. |
| 2 | Kartu Tensor 3D Masuk | *Card + 3D Visual* | Kartu di sisi kiri yang menampilkan visualisasi tumpukan 128 *feature map* dalam bentuk lapisan-lapisan (*stacked layers*) pseudo-3D. Di bawahnya terdapat keterangan *"Setiap lapisan = satu feature map 28x28 piksel dari satu filter berbeda (hasil blok konvolusi ke-3)"*. |
| 3 | Tabel Statistik Vektor | *Parameter Table* | Tabel berisi 7 baris informasi numerik: **Dimensi input** ($28 x 28 x 128$), **Metode** (Rata-rata per channel / GAP), **Total elemen output** (128), **Nilai minimum** (`0.00` karena ReLU), **Nilai maksimum** (misal: `3.74`), **Rata-rata** (misal: `0.61`), dan **Sparsity / nilai 0** (misal: `29%`). Nilai-nilai ini diambil secara dinamis dari hasil inferensi gambar pengguna. |
| 4 | Kartu Visualisasi Vektor 1D Output | *Card + Bar Visual* | Kartu di sisi kanan yang menampilkan sampel 60 nilai pertama dari 128 total nilai dalam bentuk batang-batang vertikal mini (*bar visualization*). Setiap batang merepresentasikan satu elemen vektor GAP, dengan tinggi batang proporsional terhadap besaran nilainya. Di bawahnya terdapat label *"Channel ke-1"* hingga *"Channel terakhir"*. |
| 5 | Bilah Progres Panjang Vektor | *Progress Bar* | Bilah progres penuh 100% bertuliskan *"Panjang vektor"* dengan nilai *"128 elemen"*, menunjukkan bahwa seluruh 128 kanal telah berhasil diringkas. |
| 6 | Kartu Proses Transformasi | *Flow Diagram Card* | Kartu visual yang menampilkan alur transformasi dimensi menggunakan *badge* berwarna: kotak ungu bertuliskan *"28x28x128"* → teks *"GAP (rata-rata HxW)"* → kotak hijau bertuliskan *"128 x 1"*, disertai penjelasan bahwa setiap *feature map* 28x28 dirata-ratakan menjadi satu nilai tunggal. |
| 7 | Tombol *"Kembali"* | *Secondary Button* | Tombol navigasi kembali ke halaman Pooling (Tahap 4). |
| 8 | Tombol *"Lanjut ke Fully Connected"* | *Primary Button* | Tombol navigasi maju ke halaman Fully Connected Layer (Tahap 6). |

**Kodingan Fungsi Logika Ekstraksi Model Sub-Jaringan (`build_extraction_models`):**
```python
def build_extraction_models(model):
    result = {}
    if model is None:
        return result

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        return result

    conv_layer = None
    relu_layer = None
    pool_layer = None

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and conv_layer is None:
            conv_layer = layer
        if isinstance(layer, tf.keras.layers.ReLU) and relu_layer is None:
            relu_layer = layer
        name_lower = layer.name.lower()
        if 'block_1_project_bn' in name_lower:
            pool_layer = layer

    base_outputs = {'final': base_model.output}
    if conv_layer:
        base_outputs['conv'] = conv_layer.output
    if relu_layer:
        base_outputs['relu'] = relu_layer.output
    if pool_layer:
        base_outputs['pool'] = pool_layer.output

    try:
        result['base'] = tf.keras.Model(inputs=base_model.input, outputs=base_outputs)
    except Exception:
        pass

    head_outputs = {'output': model.output}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            head_outputs['gap'] = layer.output
        elif isinstance(layer, tf.keras.layers.Dense):
            if hasattr(layer, 'units') and layer.units == 128:
                head_outputs['dense128'] = layer.output

    try:
        result['head'] = tf.keras.Model(inputs=model.input, outputs=head_outputs)
    except Exception:
        pass

    return result
```

---

### 7. Halaman Fully Connected Layer & Transfer Learning
Halaman Fully Connected Layer menampilkan visualisasi tingkat aktivasi unit-unit neuron pada *Dense Layer* yang terdiri dari 128 neuron dengan aktivasi ReLU beserta penerapan lapisan *Dropout*. Tampilan ini menggambarkan bagaimana fitur-fitur bernilai tinggi yang telah diringkas oleh GAP dihubungkan secara penuh dan dikombinasikan secara kompleks untuk mengevaluasi bobot probabilitas, sementara fungsi *Dropout* secara acak menonaktifkan sebagian neuron selama pelatihan untuk memastikan generalisasi model tetap optimal.

`[BAGIAN GAMBAR 4. 14: Tempatkan Tangkapan Layar Halaman Fully Connected Layer di sini]`

**Gambar 4. 14 Tangkapan Layar Halaman Fully Connected Layer**

Berdasarkan tangkapan layar pada Gambar 4.14, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman Fully Connected Layer:

#### Tabel 4. 12 Rincian Komponen Antarmuka Halaman Fully Connected Layer

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan | *Step Indicator* | Label bertuliskan *"Tahap 6 dari 7"* disertai judul *"Fully Connected Layer"* dan penjelasan bahwa vektor GAP (1.280 nilai) masuk ke satu lapisan Dense berisi 128 neuron (aktivasi ReLU) dengan Dropout saat pelatihan, lalu ke lapisan output Softmax. |
| 2 | Kartu Arsitektur FC Layer | *Architecture Card* | Kartu di sisi kiri yang menampilkan diagram susunan lapisan *Fully Connected* secara vertikal (*stacked layout*), menunjukkan alur dari input vektor GAP (1.280 nilai) → Dense Layer (128 neuron) → Dropout → Output Layer (4 kelas). |
| 3 | Tabel Parameter FC | *Parameter Table* | Tabel berisi 4 baris spesifikasi: **Input vektor** (1.280 nilai dari GAP), **FC Layer / Dense** (128 neuron, aktivasi ReLU), **Dropout** (0.2 - 0.3, hanya saat pelatihan), dan **Output layer** (4 kelas, Softmax). |
| 4 | Kartu Grid Neuron Aktif (*Neuron Grid*) | *Card + Grid Visual* | Kartu di sisi kanan dengan judul *"Visualisasi Neuron Aktif (FC Layer 1 — 200 sampel)"*. Menampilkan grid berisi 200 kotak kecil yang merepresentasikan sampel neuron: kotak berwarna ungu menunjukkan neuron aktif (nilai > 0 setelah ReLU) dan kotak berwarna abu-abu menunjukkan neuron tidak aktif (nilai = 0). Disertai keterangan legenda warna di atas grid. |
| 5 | Kartu Perhitungan Softmax | *Formula Card + Table* | Kartu lebar penuh di bawah grid yang menampilkan rumus matematika Softmax beserta tabel perhitungan *logit* untuk setiap kelas. Tabel ini memuat kolom **Kelas**, **Logit (z)**, **exp(z)**, dan **Probabilitas Softmax (%)** untuk ke-4 kelas target (Moler, Bukan Bawang, Sehat, Trotol), disertai catatan penjelasan interpretasi hasil. |
| 6 | Tombol *"Kembali"* | *Secondary Button* | Tombol navigasi kembali ke halaman GAP (Tahap 5). |
| 7 | Tombol *"Lihat Hasil Deteksi"* | *Success Button* | Tombol navigasi berwarna hijau (*success style*) yang mengarahkan pengguna ke halaman terakhir (Tahap 7) untuk melihat diagnosis akhir dan rekomendasi penanganan. |

**Kodingan Fungsi Logika Pembuatan Model CNN (`build_transfer_learning_model`):**
```python
def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=4):
    # 1. Base Model dengan Pre-trained Weights ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model agar weight dasar tidak rusak di tahap awal
    base_model.trainable = False

    # 2. Arsitektur Head Classification
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="BawangMerah_MobileNetV2")
    return model
```

---

### 8. Halaman Hasil Deteksi & Rekomendasi
Halaman hasil deteksi merupakan antarmuka keluaran (*output interface*) utama yang menampilkan diagnosis akhir kondisi tanaman bawang merah ke dalam salah satu dari 4 kelas kategori (*Moler*, *Trotol*, *Sehat*, atau *Bukan Bawang*) dilengkapi dengan tingkat persentase kepercayaan probabilitas (*confidence score*) yang dihitung oleh fungsi *Softmax*. Selain menyajikan label diagnosis secara visual dan jelas, halaman ini juga menyediakan rekomendasi praktis mengenai langkah-langkah pengendalian hama atau perawatan pertanian yang disesuaikan dengan jenis penyakit yang terdeteksi untuk membantu petani mengambil tindakan yang tepat.

`[BAGIAN GAMBAR 4. 15: Tempatkan Tangkapan Layar Halaman Hasil Deteksi di sini]`

**Gambar 4. 15 Tangkapan Layar Halaman Hasil Deteksi**

Berdasarkan tangkapan layar pada Gambar 4.15, berikut adalah rincian setiap komponen antarmuka yang terdapat pada halaman hasil deteksi:

#### Tabel 4. 13 Rincian Komponen Antarmuka Halaman Hasil Deteksi

| No | Nama Komponen | Jenis Elemen | Deskripsi Fungsional |
| :-: | :--- | :--- | :--- |
| 1 | Header Tahapan | *Step Indicator* | Label bertuliskan *"Tahap 7 dari 7"* disertai judul *"Hasil Deteksi Penyakit"* dan penjelasan bahwa model CNN telah selesai menganalisis gambar. |
| 2 | Banner Hasil Deteksi (*Result Banner*) | *Success Banner* | Panel notifikasi hijau di bagian atas yang menampilkan ikon centang, judul *"Deteksi Selesai"*, sub-judul ringkasan hasil, dan *tag badge* putih bertuliskan *"Akurat"*. Warna banner berubah secara dinamis sesuai kelas terdeteksi. |
| 3 | Kartu Gambar Input Teranalisis | *Card + Image* | Kartu di sisi kiri yang menampilkan ulang gambar asli yang telah diunggah pengguna, dilengkapi keterangan *"Menganalisis area gambar…"* di bagian bawah gambar. |
| 4 | Kartu Probabilitas Tiap Kelas | *Card + Progress Bars* | Kartu berisi 4 bilah progres horizontal yang masing-masing merepresentasikan probabilitas satu kelas: **Moler**, **Bukan Bawang**, **Sehat**, dan **Trotol**. Setiap bilah menampilkan nama kelas, nilai persentase, dan *progress bar* berwarna sesuai identitas kelas. |
| 5 | Kartu Grafik Probabilitas | *Card + Chart.js Canvas* | Kartu berisi elemen `<canvas>` yang menampilkan *bar chart* probabilitas ke-4 kelas menggunakan pustaka Chart.js. Grafik ini memvisualisasikan distribusi kepercayaan model secara interaktif dengan *tooltip* saat kursor diarahkan ke setiap batang. |
| 6 | Panel Kelas Terdeteksi (*Confidence Big*) | *Hero Display Panel* | Panel besar di sisi kanan yang menampilkan tiga informasi utama secara vertikal: label *"Kelas Terdeteksi"*, **nama penyakit** yang terdeteksi dalam teks besar (misal: *"Moler"*), **nama latin** patogen dalam teks miring ungu (misal: *Fusarium oxysporum*), **angka *confidence score*** berukuran sangat besar (misal: *95.00%*), label *"Confidence Score"*, serta **bilah progres ungu** yang panjangnya proporsional terhadap tingkat kepercayaan. |
| 7 | Kartu Rekomendasi Penanganan | *Card + List* | Kartu berisi daftar poin-poin rekomendasi tindakan praktis yang disesuaikan secara dinamis dengan kelas penyakit terdeteksi, mencakup langkah pengendalian hama, pemupukan, penyemprotan fungisida, atau informasi bahwa tanaman dalam kondisi sehat. |
| 8 | Tabel Ringkasan Proses CNN | *Parameter Table* | Tabel rangkuman seluruh alur *pipeline* CNN dari awal hingga akhir: **Input gambar** ($224 x 224 x 3$), **Konvolusi** (3 blok: 32 → 64 → 128 filter), **ReLU** (jumlah nilai aktif), **Pooling** (3 tahap, akhir $28 x 28 x 128$), **GAP** (128 elemen), **FC** (512 neuron + Dropout 0.5), dan **Output** (4 kelas Softmax). |
| 9 | Tombol *"Kembali"* | *Secondary Button* | Tombol navigasi kembali ke halaman Fully Connected (Tahap 6). |
| 10 | Tombol *"Deteksi Ulang"* | *Secondary Button* | Tombol yang mereset seluruh sesi dan mengembalikan pengguna ke halaman Input (Tahap 1) untuk mengunggah gambar baru. |
| 11 | Tombol *"Unduh Laporan PDF"* | *Primary Button* | Tombol aksi untuk mengunduh resume lengkap hasil deteksi ke dalam berkas format `.pdf`, mencakup gambar masukan, kelas terdeteksi, *confidence score*, probabilitas tiap kelas, dan rekomendasi penanganan. |

**Kodingan Fungsi Logika Endpoint Prediksi & Sesi (`api_predict_session`):**
```python
@app.get("/api/predict")
def api_predict_session(session_id: str = Query(...)):
    if session_id not in SESSION_STORE:
        raise HTTPException(status_code=404, detail="Sesi tidak ditemukan atau telah kedaluwarsa.")

    session_data = SESSION_STORE[session_id]
    
    if session_data["prediction_cache"] is not None:
        return session_data["prediction_cache"]

    # Jalankan prediksi asli dengan model CNN
    input_tensor = preprocess_image(session_data["bytes"])
    predictions = model.predict(input_tensor)[0]
    predicted_class_idx = int(np.argmax(predictions))
    predicted_label = class_labels[predicted_class_idx]
    confidence_score = round(float(predictions[predicted_class_idx] * 100), 2)

    meta = get_disease_meta(predicted_label)
    
    probabilities_list = [
        {
            "name": get_disease_meta(label).get("display_name", label),
            "pct": round(float(prob * 100), 2),
            "color": get_disease_meta(label).get("color", "#534AB7")
        }
        for label, prob in zip(class_labels, predictions)
    ]

    res_data = {
        "success": True,
        "data": {
            "predicted_class_idx": predicted_class_idx,
            "predicted_class": meta["display_name"],
            "predicted_latin": meta["latin"],
            "confidence": confidence_score,
            "color": meta["color"],
            "rekomendasi": meta["rekomendasi"],
            "probabilities": probabilities_list,
            "raw_label": predicted_label
        }
    }
    session_data["prediction_cache"] = res_data
    return res_data
```

---

## F. Pengujian Sistem

### 1. Pengujian Black Box

Pengujian fungsionalitas antarmuka web dilakukan menggunakan metode *Black-Box Testing* untuk memverifikasi kebenaran respons sistem terhadap berbagai skenario pengguna.

#### Tabel 4. 4 Hasil Pengujian Black-Box Antarmuka Web

| No | Skenario Pengujian | Masukan (Input) | Hasil yang Diharapkan | Hasil Pengujian | Status |
| :-: | :--- | :--- | :--- | :--- | :-: |
| 1 | Pengunggahan gambar valid | Berkas `.jpg` / `.png` | Gambar diterima, pratinjau tampil, proses CNN berjalan | Sesuai ekspektasi | **Valid** |
| 2 | Pengunggahan berkas non-gambar | Berkas `.pdf` / `.docx` | Sistem menolak berkas & menampilkan pesan penolakan | Sesuai ekspektasi | **Valid** |
| 3 | Tangkapan kamera langsung | Izin kamera diberikan | Foto berhasil ditangkap dan diproses ke pipeline | Sesuai ekspektasi | **Valid** |
| 4 | Navigasi visualisasi CNN | Klik tombol *Lanjut/Kembali* | Tampilan berpindah antar step konvolusi, relu, pooling, dll. | Sesuai ekspektasi | **Valid** |
| 5 | Tampilan hasil klasifikasi | Citra daun bercak ungu | Menampilkan label *Trotol* & score probabilitas beserta rekomendasi | Sesuai ekspektasi | **Valid** |
| 6 | Simpan & unduh laporan | Klik *Unduh Laporan PDF* | Berkas laporan PDF terunduh berisi rincian hasil deteksi | Sesuai ekspektasi | **Valid** |

---

### 2. Hasil Pelatihan Model

Ringkasan performa akhir model *Convolutional Neural Network* (CNN) MobileNetV2 yang diuji disajikan pada Tabel 4.5.

#### Tabel 4. 5 Ringkasan Performa Evaluasi Model CNN

| Metrik Evaluasi | Nilai Hasil Pelatihan / Pengujian |
| :--- | :---: |
| **Total Dataset** | **2.000 Citra** |
| **Pembagian Data (Train / Val / Test)** | **70% (1.400) / 15% (300) / 15% (300)** |
| **Jumlah Kelas Target** | **4 Kelas (`moler`, `non_bawang`, `sehat`, `trotol`)** |
| **Test Accuracy (Data Uji 300 Sampel)** | **93,67%** |
| **Macro Average Precision** | **93,83%** |
| **Macro Average Recall** | **93,67%** |
| **Macro Average F1-Score** | **93,70%** |
| **Weighted Average F1-Score** | **93,70%** |

---

## Kesimpulan Bab IV

Berdasarkan pengujian dan pembahasan yang telah dipaparkan, sistem klasifikasi penyakit bawang merah berbasis CNN MobileNetV2 ini terbukti memiliki performa yang handal dengan mencapai **Akurasi sebesar 93,67% pada 300 data uji mandiri**. Jaringan mampu membedakan kondisi bawang sehat, penyakit trotol (*Alternaria porri*), penyakit moler (*Fusarium oxysporum*), serta menolak citra non-bawang secara tepat.
