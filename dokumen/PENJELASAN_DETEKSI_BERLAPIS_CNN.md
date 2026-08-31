# MEKANISME DETEKSI BERLAPIS CNN PADA KLASIFIKASI PENYAKIT BAWANG MERAH

Dokumen ini menjelaskan secara ilmiah, komprehensif, dan sistematis bagaimana program dan model CNN (*Convolutional Neural Network*) berbasis MobileNetV2 memindai (*scan*) citra masukan dan memprosesnya secara berlapis (*Hierarchical Feature Extraction*) hingga menghasilkan kesimpulan diagnosis penyakit bawang merah.

---

## 1. Konsep Utama: Ekstraksi Fitur Berlapis (*Hierarchical Feature Extraction*)

Jaringan saraf konvolusi (CNN) tidak menganalisis foto secara instan dalam satu tahap tunggal, melainkan melalui **pemindaian spasial bertingkat (*layered scanning*)** dari informasi visual paling sederhana menuju pemahaman pola semantik yang global dan kompleks.

**Diagram Alur Pemrosesan Berlapis:**

```
[Citra Masukan (224x224x3 Piksel)]
    |
    v
1. Pemindaian Piksel (Scanning Kernel 3x3)
    |
    v
2. Lapisan Awal (Fitur Rendah: Garis, Tepi & Warna Hijau)
    |
    v
3. Lapisan Menengah (Fitur Tekstur: Tabung Daun & Bercak Lesi)
    |
    v
4. Lapisan Mendalam (Fitur Kompleks: Konsep Morfologi Penyakit Utuh)
    |
    v
5. Global Average Pooling (GAP: Vektor Fitur 1280-D)
    |
    v
6. Dense Layer & Softmax (Pengambilan Keputusan Diagnosis)
```

---

## 2. Rincian Peran Tiap Lapisan dalam Mendeteksi Penyakit

### a. Tahap Pemindaian Piksel Masukan (*Input & Normalization*)
- **Operasi:** Matriks citra berukuran 224x224x3 dinormalisasi ke rentang [0.0, 1.0] dengan rumus: `nilai_piksel / 255.0`.
- **Mekanisme:** Jendela kernel konvolusi (3x3) bergeser (*sliding window / scan*) melintasi seluruh piksel gambar dengan langkah pergeseran (*stride*).

### b. Lapisan Awal / *Low-Level Features* (Conv1 - Block 1)
- **Fungsi:** Mendeteksi elemen visual paling primitif.
- **Fitur yang Diekstrak:**
  - Garis tepi vertikal, horizontal, dan diagonal daun.
  - Gradien intensitas terang-gelap (*kontras pencahayaan*).
  - Spektrum warna dasar hijau klorofil daun bawang merah (*Allium cepa*).

### c. Lapisan Menengah / *Mid-Level Features* (Block 2 - Block 7)
- **Fungsi:** Menggabungkan garis dan warna menjadi bentuk geometris dan tekstur lokal.
- **Fitur yang Diekstrak:**
  - Bentuk tabung silinder berongga khas daun bawang merah.
  - Kelengkungan garis helai daun (daun lurus vs daun meliuk terpuntir).
  - Tekstur anomali bercak/lingkaran lokal di permukaan daun.

### d. Lapisan Mendalam / *High-Level Features* (Block 8 - Conv Head 1280)
- **Fungsi:** Memahami konteks visual penyakit secara utuh dan menyeluruh.
- **Fitur yang Diekstrak:**
  - **Pola Trotol:** Pola bercak cekung (*lesi*) oval keputihan dengan tepi berwarna ungu/cokelat (*Alternaria porri*).
  - **Pola Moler:** Pola helai daun meliuk-liuk abnormal, rebah, dan menguning mulai dari pangkal (*Fusarium oxysporum*).
  - **Pola Sehat:** Helai daun hijau mulus merata dan tegak tanpa lesi/cacat.
  - **Pola Bukan Bawang:** Objek luar yang tidak memiliki karakteristik tanaman bawang merah.

---

## 3. Mekanisme Pengambilan Keputusan Program

### a. Lapisan Klasifikasi Akhir (Dense Layer & Softmax)
1. **Global Average Pooling (GAP):** Merangkum seluruh informasi spasial dari 1.280 kanal menjadi vektor 1D berdimensi 1.280 elemen.
2. **Dense Layer (128 Neuron):** Melakukan kombinasi linier bobot fitur terpenting.
3. **Dense Output Layer (4 Neuron):** Menghasilkan 4 nilai logit mentah (**z_1, z_2, z_3, z_4**) untuk masing-masing kelas target.
4. **Fungsi Aktivasi Softmax:** Mengonversi logit menjadi distribusi persentase probabilitas total 100%:

$$P(y_k) = \frac{e^{z_k}}{\sum_{j=1}^{4} e^{z_j}} \times 100\%$$

### b. Pemetaan Indeks Kelas (`class_indices.json`)
Program membaca kamus indeks kelas:

| Indeks | Label Kelas | Nama Tampilan |
|:------:|:------------|:--------------|
| 0 | `moler` | Moler / Layu Fusarium |
| 1 | `non_bawang` | Objek Bukan Bawang |
| 2 | `sehat` | Bawang Sehat |
| 3 | `trotol` | Trotol / Bercak Ungu |

Fungsi `np.argmax(predictions)` mengambil indeks dengan probabilitas tertinggi:
- Jika Indeks `3` memiliki nilai terbesar -> Program menyimpulkan **"Trotol / Bercak Ungu"**.

### c. Sistem Validasi Keamanan (*Rejection Guard*)
Sebelum kesimpulan dikirim ke antarmuka pengguna, program menjalankan 4 pemeriksaan keamanan:
1. Apakah model secara eksplisit memprediksi kelas `non_bawang`?
2. Apakah skor keyakinan kelas tertinggi di bawah 50%?
3. Apakah selisih antara kelas tertinggi dan kedua tertinggi terlalu tipis (< 12%) dan skor < 65%?
4. Apakah probabilitas kelas "Bukan Bawang" cukup signifikan (>= 25%)?

Jika salah satu kondisi terpenuhi -> Program secara otomatis menolak dan menetapkan status **"Objek Bukan Bawang"** agar tidak terjadi salah diagnosis.

---

## 4. Studi Kasus Nyata: Foto Sampel Uji Pengguna

Pada pengujian foto sampel daun lapangan yang diunggah pengguna:

### Hasil Deteksi Model:

| No | Kelas Penyakit | Indeks | Probabilitas |
|:--:|:---------------|:------:|:------------:|
| 1 | **Trotol / Bercak Ungu** | 3 | **93,05%** |
| 2 | Sehat | 2 | 3,40% |
| 3 | Moler / Layu Fusarium | 0 | 3,37% |
| 4 | Objek Bukan Bawang | 1 | 0,18% |

### Kesimpulan Komputasi:
- Total probabilitas **bawang merah** = 93,05% + 3,40% + 3,37% = **99,82%**
- Probabilitas **bukan bawang** = **0,18%**
- Karena Neuron Indeks 3 teraktivasi paling dominan (93,05%) akibat kecocokan pola bercak lesi trotol, program dengan akurat menyimpulkan bahwa tanaman terinfeksi **Trotol / Bercak Ungu (*Alternaria porri*)** dengan tingkat keyakinan **Akurat**.

---

## 5. Panduan Menjawab Pertanyaan Dosen Penguji Saat Sidang

> **Dosen Penguji:** *"Bagaimana alur komputasi CNN hingga program Anda bisa menyimpulkan penyakit bawang ini?"*

> **Jawaban Mahasiswa:**
> *"Sistem memproses gambar melalui pendekatan ekstraksi fitur hierarkis bertingkat. Pertama, citra dipindai oleh filter konvolusi berukuran 3x3. Lapisan awal mengekstrak garis tepi dan warna klorofil bawang; lapisan menengah mengenali tekstur bentuk tabung daun dan lesi bercak; sedangkan lapisan akhir merangkai seluruh pola tersebut menjadi konsep klinis penyakit. Skor dari 1.280 fitur dirangkum melalui Global Average Pooling dan diklasifikasikan oleh fungsi Softmax pada 4 neuron keluaran. Indeks dengan aktivasi tertinggi (93,05% pada kelas Trotol) secara otomatis dipetakan oleh program menjadi kesimpulan diagnosis dan rekomendasi penanganan fungisida yang tepat."*

> **Dosen Penguji:** *"Bagaimana program tahu bahwa ini adalah bawang merah dan bukan tanaman lain?"*

> **Jawaban Mahasiswa:**
> *"Model telah dilatih dengan ribuan contoh foto berlabel dari 4 kelas. Saat foto baru diunggah, model mengeluarkan probabilitas untuk semua kelas. Pada sampel ini, total probabilitas ketiga kelas bawang (Trotol 93,05% + Sehat 3,40% + Moler 3,37%) mencapai 99,82%, sedangkan kelas bukan bawang hanya 0,18%. Selain itu, sistem memiliki mekanisme Rejection Guard yang secara otomatis menolak gambar jika keyakinan model di bawah 50% atau jika probabilitas kelas bukan bawang terlalu tinggi."*
