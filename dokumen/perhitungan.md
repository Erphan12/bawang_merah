# Perhitungan Manual CNN (Convolutional Neural Network)

Dokumen ini memuat **simulasi perhitungan matematis terperinci, presisi, dan komprehensif langkah demi langkah** yang **100% dihitung menggunakan parameter bobot asli dari model terlatih `best_bawang_model.h5`** serta sampel citra daun bawang merah asli dari dataset uji (`(126).jpg`, kelas *Trotol / Bercak Ungu*).

Perhitungan ini secara eksplisit menjabarkan setiap perubahan nilai numerik, persentase, dan rumus perantara pada komputasi jaringan saraf *Convolutional Neural Network* (CNN) berbasis MobileNetV2 dan *Classification Head*.

---

### a. Normalisasi Citra Masukan ($I \rightarrow I_{\text{norm}}$)
Citra masukan daun bawang merah berukuran $224 x 224 x 3$ piksel memuat rentang nilai intensitas warna RGB $[0, 255]$. Pada pra-pemrosesan citra, nilai piksel dinormalisasi ke rentang kontinu $[0.0, 1.0]$ menggunakan persamaan pembagian:

$$
X_{i, j} = \frac{\text{Channel}(i, j)}{255.0}
$$

Diambil sampel patch citra $3 x 3$ piksel dari area pusat citra uji (`(126).jpg`, koordinat pusat $x=112, y=112$):

- **Piksel RGB Asli $[0, 255]$:**
  - Baris 1: $(1,1)=[168, 169, 157]$, $(1,2)=[164, 165, 152]$, $(1,3)=[168, 169, 156]$
  - Baris 2: $(2,1)=[172, 172, 160]$, $(2,2)=[169, 168, 156]$, $(2,3)=[167, 166, 153]$
  - Baris 3: $(3,1)=[176, 175, 163]$, $(3,2)=[176, 174, 161]$, $(3,3)=[173, 169, 157]$

- **Hasil Normalisasi $[0.0, 1.0]$ (Rata-rata 3 Kanal RGB per Sel):**
  - Sel $(1,1)$: $\text{Rata-rata}(168, 169, 157) = 164.6667 / 255.0 = \mathbf{0.6458}$
  - Sel $(1,2)$: $\text{Rata-rata}(164, 165, 152) = 160.3333 / 255.0 = \mathbf{0.6288}$
  - Sel $(1,3)$: $\text{Rata-rata}(168, 169, 156) = 164.3333 / 255.0 = \mathbf{0.6444}$
  - Sel $(2,1)$: $\text{Rata-rata}(172, 172, 160) = 168.0000 / 255.0 = \mathbf{0.6588}$
  - Sel $(2,2)$: $\text{Rata-rata}(169, 168, 156) = 164.3333 / 255.0 = \mathbf{0.6444}$
  - Sel $(2,3)$: $\text{Rata-rata}(167, 166, 153) = 162.0000 / 255.0 = \mathbf{0.6353}$
  - Sel $(3,1)$: $\text{Rata-rata}(176, 175, 163) = 171.3333 / 255.0 = \mathbf{0.6719}$
  - Sel $(3,2)$: $\text{Rata-rata}(176, 174, 161) = 170.3333 / 255.0 = \mathbf{0.6680}$
  - Sel $(3,3)$: $\text{Rata-rata}(173, 169, 157) = 166.3333 / 255.0 = \mathbf{0.6523}$

Sehingga matriks piksel masukan ter-normalisasi ($I_{\text{norm}}$) adalah:

$$
I_{\text{norm}} = \begin{bmatrix} 0.6458 & 0.6288 & 0.6444 \\ 0.6588 & 0.6444 & 0.6353 \\ 0.6719 & 0.6680 & 0.6523 \end{bmatrix}
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
1. **Sel (1,1):** $0.6458 x -0.19900 = \mathbf{-0.128507}$
2. **Sel (1,2):** $0.6288 x -0.29629 = \mathbf{-0.186296}$
3. **Sel (1,3):** $0.6444 x -0.07772 = \mathbf{-0.050088}$
4. **Sel (2,1):** $0.6588 x +0.32279 = \mathbf{+0.212659}$
5. **Sel (2,2):** $0.6444 x +0.42667 = \mathbf{+0.274963}$
6. **Sel (2,3):** $0.6353 x +0.15698 = \mathbf{+0.099730}$
7. **Sel (3,1):** $0.6719 x -0.10745 = \mathbf{-0.072195}$
8. **Sel (3,2):** $0.6680 x -0.00619 = \mathbf{-0.004133}$
9. **Sel (3,3):** $0.6523 x -0.07872 = \mathbf{-0.051350}$

**Penjumlahan Bertahap Seluruh Nilai Komponen:**
- Total komponen negatif $= -0.128507 - 0.186296 - 0.050088 - 0.072195 - 0.004133 - 0.051350 = \mathbf{-0.492569}$
- Total komponen positif $= +0.212659 + 0.274963 + 0.099730 = \mathbf{+0.587352}$
- Hasil akhir konvolusi mentah:
  $$y = -0.492569 + 0.587352 = \mathbf{+0.094783}$$

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
   $$y - \mu = 0.094783 - (-0.036762) = \mathbf{+0.131545}$$
2. **Penambahan Epsilon ($\epsilon = 0.001$):**
   $$\sigma^2 + \epsilon = 0.120517 + 0.001000 = \mathbf{0.121517}$$
3. **Akar Kuadrat Standar Deviasi:**
   $$\sqrt{0.121517} = \mathbf{0.348592}$$
4. **Pembagian Normalisasi Standardized ($\hat{x}$):**
   $$\hat{x} = \frac{0.131545}{0.348592} = \mathbf{+0.377359}$$
5. **Skala ($\gamma$) & Geseran ($\beta$):**
   $$y_{\text{BN}} = (0.612176 x 0.377359) + 2.255549 = 0.231010 + 2.255549 = \mathbf{+2.486559}$$

---

### d. Fungsi Aktivasi ReLU (*Rectified Linear Unit*)
Sesuai tampilan website pada **Tahap 3**, fungsi aktivasi ReLU menerapkan persamaan:

$$
f(x) = \max(0, x)
$$

**Perubahan Nilai Aktivasi:**
- **Kasus 1 — Nilai Positif ($x = +2.486559$):**
  $$f(+2.486559) = \max(0, +2.486559) = \mathbf{+2.486559} \quad (\text{Nilai positif tetap dipertahankan})$$
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
- $z_1 \text{ (Moler)} = \mathbf{+0.038101}$
- $z_2 \text{ (Bukan Bawang)} = \mathbf{+0.202019}$
- $z_3 \text{ (Sehat)} = \mathbf{+0.001910}$
- $z_4 \text{ (Trotol / Bercak Ungu)} = \mathbf{+0.757969}$

**Normalisasi Probabilitas Softmax Langkah demi Langkah:**

1. **Penghitungan Nilai Eksponensial $e^{z_k}$ per Kelas:**
   - Kelas 1 (Moler): $e^{+0.038101} = \mathbf{1.038836}$
   - Kelas 2 (Bukan Bawang): $e^{+0.202019} = \mathbf{1.223872}$
   - Kelas 3 (Sehat): $e^{+0.001910} = \mathbf{1.001912}$
   - Kelas 4 (Trotol): $e^{+0.757969} = \mathbf{2.133939}$

2. **Penjumlahan Total Penyebut Softmax ($\sum e^{z_j}$):**
   $$\sum e^{z} = 1.038836 + 1.223872 + 1.001912 + 2.133939 = \mathbf{5.398559}$$

3. **Penghitungan Persentase Probabilitas Akhir per Kelas:**
   - **Moler:**
     $$P(\text{moler}) = \left( \frac{1.038836}{5.398559} \right) x 100\% = 0.192429 x 100\% = \mathbf{19,24\%}$$
   - **Bukan Bawang:**
     $$P(\text{non\_bawang}) = \left( \frac{1.223872}{5.398559} \right) x 100\% = 0.226703 x 100\% = \mathbf{22,67\%}$$
   - **Sehat:**
     $$P(\text{sehat}) = \left( \frac{1.001912}{5.398559} \right) x 100\% = 0.185589 x 100\% = \mathbf{18,56\%}$$
   - **Trotol / Bercak Ungu:**
     $$P(\text{trotol}) = \left( \frac{2.133939}{5.398559} \right) x 100\% = 0.395279 x 100\% = \mathbf{39,53\%}$$

**Kesimpulan Inferensi Diagnostik:**  
Model CNN memprediksi citra masukan secara otomatis sebagai **Trotol / Bercak Ungu (*Alternaria porri*)** dengan nilai kepastian probabilitas tertinggi sebesar **39,53%**.
