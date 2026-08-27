# Daftar Komponen dan Elemen Tampilan Frontend Per Halaman HTML
**Sistem Deteksi & Visualisasi Penyakit Bawang Merah**

---

## 1. Halaman Beranda / Utama (`index.html` / `beranda.html`)

| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Top Bar / Header Utama | Elemen Navigasi & Header | Bilah bagian atas yang memuat logo aplikasi, judul sistem, serta tombol navigasi menuju halaman deteksi dan login. |
| 2 | Logo & Identitas Aplikasi | Elemen Branding & Visual | Ikon SVG berlatar warna tematik beserta teks nama sistem (*Klasifikasi Penyakit Bawang Merah*). |
| 3 | Hero Banner / Section Utama | Elemen Tampilan Informasi | Area utama sambutan yang berisi judul besar, deskripsi singkat aplikasi, dan tombol aksi utama untuk mulai deteksi. |
| 4 | Tombol Mulai Deteksi | Elemen Interaktif (Primary Button) | Tombol pemicu aksi utama yang mengarahkan pengguna/petani langsung ke halaman deteksi (`deteksi.html`). |
| 5 | Kartu Ringkasan Fitur Utama | Elemen Layout (Content Cards) | Kartu-kartu visual yang memperkenalkan keunggulan sistem (misal: *Deteksi Cepat*, *Visualisasi CNN Interaktif*, *Laporan Akurat*). |
| 6 | Footer / Catatan Bawah | Elemen Informasi & Navigasi | Bagian dasar halaman yang memuat hak cipta dan tautan informasi pelengkap. |

---

## 2. Halaman Deteksi Utama & Pipeline (`deteksi.html`)

| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Top Bar Navigasi Deteksi | Elemen Navigasi & Header | Header khusus halaman deteksi dengan tombol kembali ke beranda dan indikator status mode. |
| 2 | Pipeline Stepper (7 Layer CNN) | Elemen Navigasi Tahapan | Baris tab interaktif 7 langkah (Input → Conv → ReLU → Pooling → Flatten → FC → Output) untuk menelusuri alur CNN. |
| 3 | Frame Sub-Halaman CNN (`pages/01` - `07`) | Elemen Kontainer / Frame | Frame dinamis yang memuat modul simulasi matematika interaktif dari `pages/01_input.html` s.d. `pages/07_output.html`. |

---

## 3. Halaman Login Admin (`login.html`)

| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Kartu Form Login (Login Card) | Elemen Layout / Kontainer | Kartu di tengah layar yang memuat seluruh kontrol masukan autentikasi admin. |
| 2 | Header & Subtitle Form Login | Elemen Tampilan Teks | Judul *Login Admin* dan petunjuk bahwa halaman ini khusus untuk pengelola sistem. |
| 3 | Field Input Username | Elemen Formulir (Input Text) | Kotak isian teks untuk memasukkan nama pengguna admin. |
| 4 | Field Input Password | Elemen Formulir (Input Password) | Kotak isian kata sandi berkarakter tersembunyi untuk otentikasi admin. |
| 5 | Tombol Masuk (Submit Button) | Elemen Interaktif (Primary Button) | Tombol aksi untuk mengirimkan data kredensial login ke server. |
| 6 | Kotak Pesan Error Login | Elemen Alert / Umpan Balik | Kotak pesan peringatan berwarna merah yang muncul jika username atau password yang dimasukkan salah. |
| 7 | Tautan Kembali ke Beranda | Elemen Navigasi Teks | Tautan teks di bagian bawah form untuk kembali ke halaman utama petani. |

---

## 4. Halaman Panel Admin & Training (`admin.html`)

| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Top Bar Panel Admin | Elemen Navigasi & Header | Bilah atas khusus admin berisi judul panel, tombol *Lihat Riwayat*, *Kembali ke Situs*, dan *Keluar (Logout)*. |
| 2 | Kartu Status Model Aktif | Elemen Tampilan Informasi | Kartu ringkasan yang menampilkan jumlah kelas aktif, sumber dataset, dan waktu terakhir model dilatih. |
| 3 | Area Upload Dataset `.zip` | Elemen Input Berkas | Zone seret-lepas berkas `.zip` dataset berstruktur folder per kelas penyakit. |
| 4 | Grid Ringkasan Jumlah Sampel Kelas | Elemen Visualisasi Data | Grid petak yang menampilkan nama kelas penyakit beserta jumlah sampel gambar yang terdeteksi dari file `.zip`. |
| 5 | Tombol Latih Model (Start Training) | Elemen Interaktif (Primary Button) | Tombol eksekusi utama untuk memulai proses *training* model CNN di latar belakang server. |
| 6 | Batang Progres Pelatihan (Progress Bar) | Elemen Indikator Progres | Batang animasi berwarna dengan teks persentase (0%–100%) dan status tahap pelatihan yang sedang berjalan. |
| 7 | Form Ubah Akun Admin | Elemen Formulir Interaktif | Formulir tersembunyi (collapsible) berisi isian password lama, username baru, password baru, dan konfirmasi password. |
| 8 | Tabel Riwayat Prediksi SQLite | Elemen Tabel Data | Tabel yang memuat log riwayat prediksi petani (Waktu, Nama File, Hasil Prediksi, Confidence, Mode, dan Ukuran File). |
| 9 | Tombol Kelola Riwayat | Elemen Interaktif (Secondary) | Tombol *Muat Ulang* data riwayat dan tombol *Hapus Riwayat* dari database. |

---

## 5. Halaman Laporan Evaluasi Model (`report.html`)

| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Top Bar Laporan | Elemen Navigasi & Header | Bilah navigasi laporan dengan tombol kembali ke panel admin dan tombol unduh/cetak laporan. |
| 2 | Kartu Ringkasan Akurasi & Loss | Elemen Tampilan Informasi (Stat Box) | Kartu yang menyoroti angka persentase Akurasi Pengujian (*Test Accuracy*) dan Nilai *Loss* Akhir. |
| 3 | Grafik Kurva Pelatihan (Accuracy & Loss) | Elemen Visualisasi Grafik | Grafik garis interaktif Chart.js yang menampilkan perbandingan kurva *Training vs Validation Accuracy & Loss*. |
| 4 | Tabel Confusion Matrix Evaluasi | Elemen Tabel Data | Tabel perbandingan matriks antara kelas asli (*actual*) dan hasil prediksi (*predicted*) dari sampel pengujian. |
| 5 | Tabel Metrik Per Kelas | Elemen Tabel Data | Tabel rincian nilai *Precision*, *Recall*, dan *F1-Score* untuk setiap kelas penyakit bawang merah. |
| 6 | Kartu Rincian Parameter Teknis | Elemen Tampilan Informasi | Kartu daftar konfigurasi hiperparameter yang digunakan (ukuran gambar, jumlah epoch, batch size, learning rate, optimizer). |

---

## 6. Khusus Sub-Halaman Visualisasi CNN (`pages/01_input.html` s.d. `pages/07_output.html`)

### 6.1. Sub-Halaman 01: Unggah & Pra-proses Citra Input (`pages/01_input.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 1 | Elemen Header Teks | Memuat penanda langkah ("Tahap 1 dari 7"), judul utama ("Unggah Gambar Daun"), dan petunjuk pengambilan gambar. |
| 2 | Area Drag & Drop Berkas | Elemen Input Berkas | Kotak interaktif berkontur garis putus-putus untuk mengunggah gambar daun bawang (JPG, PNG, JPEG, maks 10 MB). |
| 3 | Tombol Ambil Foto Langsung | Elemen Interaktif (Button) | Tombol sekunder berikon kamera untuk mengaktifkan pengambil foto via webcam HP/laptop. |
| 4 | Kotak Petunjuk Pencahayaan | Elemen Notice / Alert | Kotak informasi kebiruan yang memberikan saran agar menggunakan pencahayaan yang cukup dan fokus pada daun. |
| 5 | Card Pratinjau Gambar | Elemen Layout / Kontainer | Kartu wadah untuk menampilkan gambar sampel yang dipilih beserta status keabsahan file ("✓ Gambar Valid"). |
| 6 | Stat Box Metadata Gambar | Elemen Tampilan Informasi | Tiga kotak statistik ringkas yang menampilkan Nama File, Ukuran File (KB/MB), dan Resolusi Citra (piksel). |
| 7 | Progress Bar Pra-proses Otomatis | Elemen Indikator Progres | Batang indikator penanda selesainya tahap *Resizing* ke $224 x 224$ dan normalisasi piksel $[0, 1]$. |
| 8 | Notice Kesiapan Gambar | Elemen Notice Success | Kotak hijau mengonfirmasi gambar siap diproses oleh model CNN. |
| 9 | Card Pratinjau Kosong (Empty State) | Elemen State Tampilan | Tampilan placeholder berikon gambar yang muncul sebelum pengguna memilih/mengunggah gambar. |
| 10 | Tombol Proses Gambar | Elemen Interaktif (Primary Button) | Tombol aksi utama di kanan bawah untuk memulai kalkulasi dan lanjut ke tahap konvolusi (`pages/02_conv.html`). |

---

### 6.2. Sub-Halaman 02: Lapisan Konvolusi / Conv1 (`pages/02_conv.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 2 & Navigasi | Elemen Header & Button | Penanda "Tahap 2 dari 7", judul "Lapisan Konvolusi (Conv1)", serta tombol navigasi *Kembali* dan *Lanjut*. |
| 2 | Card Demo Cara Kerja Conv1 | Elemen Visualisasi Interaktif | Tampilan rumus perkalian elemen $y = \sum (X_{i,j} x W_{i,j})$ dan petunjuk tanpa bias (`use_bias=False`). |
| 3 | Visualisator Patch $3 x 3$ & Bobot W | Elemen Visualisasi Data | Grid petak mini perbandingan patch piksel citra ($X$) dikalikan bobot filter real-time ($W$) menghasilkan nilai output mentah ($y$). |
| 4 | Kotak Penjelasan Sumber Data | Elemen Notice Info | Informasi silsilah koordinat sampel piksel asli $(110-112, 110-112)$ dan pembacaan bobot asli file `.h5`. |
| 5 | Tabel Perhitungan Empiris Dot Product | Elemen Tabel Data | Tabel rincian kalkulasi 9 posisi spasial $(i,j)$ perkalian sel $X x W$ hingga perkalian kumulatif total dot product $+0.05028$. |
| 6 | Akordion 8 Operasi Matematika | Elemen Interactive Dropdown | Akordion lipat yang menampilkan rincian ekspansif rumus → substitusi angka → hasil perhitungan secara eksplisit. |
| 7 | Akordion Peta Asal-Usul & Silsilah Data | Elemen Interactive Dropdown | Akordion rincian lacak silsilah angka dari piksel fisik HP/laptop hingga perhitungan FLOPs ($86.7\text{ MFLOPs}$). |
| 8 | Galeri Feature Maps Hasil Konvolusi | Elemen Visualisasi Grafik | Grid 32 petak gambar mini (*feature maps*) hasil ekstraksi ciri mentah filter Conv1. |
| 9 | Warning Notice Nilai Negatif | Elemen Alert / Warning | Peringatan berwarna kuning bahwa feature map mentah masih mengandung nilai negatif yang akan diolah oleh ReLU. |

---

### 6.3. Sub-Halaman 03: Fungsi Aktivasi ReLU (`pages/03_relu.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 3 | Elemen Header Teks | Penanda "Tahap 3 dari 7" dan penjelasan fungsi aktivasi $f(x) = \max(0, x)$ untuk menambah non-linearitas. |
| 2 | Display Rumus & Komparasi ReLU | Elemen Visualisasi Formula | Kotak perbandingan nilai negatif yang dibuang menjadi $0.00$ (merah) dan nilai positif yang dipertahankan utuh (teal). |
| 3 | Tabel Perhitungan Empiris ReLU | Elemen Tabel Data | Tabel perbandingan 9 piksel sampel: nilai output Conv1 ($x$), fungsi $\max(0, x)$, dan nilai akhir $f(x)$. |
| 4 | Notice Transparansi Filter #1 | Elemen Notice Info | Penjelasan transparansi bahwa data tabel bersumber langsung dari hasil nyata Filter #1 Konvolusi sebelumnya. |
| 5 | Akordion Aturan Perhitungan & Silsilah | Elemen Interactive Dropdown | Lipatan informasi logika pemutusan nilai $x \le 0 \to 0$ dan penerusan $x > 0 \to x$. |
| 6 | Tabel Spesifikasi Parameter ReLU | Elemen Tabel Parameter | Tabel ringkas metrik ReLU ($0$ parameter terlatih, ukuran tensor tetap $224 x 224 x 32$). |
| 7 | Feature Maps Setelah ReLU | Elemen Visualisasi Grafik | Galeri 32 feature maps di mana area gelap/abu-abu menandakan nilai nol (fitur non-aktif). |
| 8 | Kotak Penjelasan Fungsi Feature Maps | Elemen Notice Success | Penjelasan visualisasi pembersihan *noise*, penyorotan fitur aktif, dan kesiapan masukan ke Pooling. |

---

### 6.4. Sub-Halaman 04: Max Pooling Layer (`pages/04_pooling.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 4 | Elemen Header Teks | Penanda "Tahap 4 dari 7" dan penjelasan reduksi dimensi spasial $50\%$ pada 3 tahap pooling (Pool1, Pool2, Pool3). |
| 2 | Visualisator Max Pooling $2 x 2$ | Elemen Visualisasi Data | Grid perbandingan sebelum (4x4 aktivasi) dan sesudah (2x2 output) yang mengambil nilai tertinggi tiap jendela $2 x 2$. |
| 3 | Tabel Perhitungan Empiris Max Pooling | Elemen Tabel Data | Tabel rincian 4 jendela $2 x 2$ non-overlapping beserta nilai masukan dan nilai maksimum terambil. |
| 4 | Akordion Rincian Operasi & Silsilah Data | Elemen Interactive Dropdown | Lipatan rincian proses windowing, selection max, downsampling, dan silsilah data pooling. |
| 5 | Tabel Parameter 3 Tahap Pooling | Elemen Tabel Parameter | Spesifikasi ukuran tensor dari Pool1 ($112 x 112 x 32$), Pool2 ($56 x 56 x 64$), hingga Pool3 ($28 x 28 x 128$). |
| 6 | Feature Maps Setelah Pooling | Elemen Visualisasi Grafik | Tampilan feature maps hasil penyusutan ukuran dengan indikator persentase reduksi dimensi spasial $75\%$. |

---

### 6.5. Sub-Halaman 05: Global Average Pooling / GAP (`pages/05_flatten.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 5 | Elemen Header Teks | Penanda "Tahap 5 dari 7" dan penjelasan perataan tensor 3D ($7 x 7 x 1280$) menjadi vektor 1D ($1.280$ elemen). |
| 2 | Display Rumus & Panduan Cara Pakai GAP | Elemen Visualisasi Formula | Rumus $y = \frac{1}{49} \sum X_{7 x 7}$ dan rincian 3 langkah kalkulasi (Kumpulkan 49 piksel $\to$ Penjumlahan $\to$ Pembagian N=49). |
| 3 | Diagram 3D Tensor Masuk | Elemen Visualisasi 3D | Visualisasi tumpukan 1.280 feature maps ukuran $7 x 7$ piksel. |
| 4 | Tabel Perhitungan Empiris GAP | Elemen Tabel Data | Tabel rincian perata-rataan channel: total penjumlahan 49 sel dibagi 49 menghasilkan nilai skalar tunggal. |
| 5 | Akordion Operasi Matematika & Silsilah GAP | Elemen Interactive Dropdown | Rincian ekspansif perhitungan rata-rata spasial dan silsilah transformasi vektor. |
| 6 | Tabel Statistik Vektor GAP | Elemen Tabel Parameter | Tabel ringkasan metrik vektor ($1.280$ elemen, min, max, rata-rata, dan tingkat *sparsity* $29\%$). |
| 7 | Sample Bar Visualisasi Vektor 1D | Elemen Visualisasi Vektor | Visualisasi pita warna sampel 60 nilai pertama dari $1.280$ total nilai vektor GAP. |
| 8 | Skema Diagram Transformasi Dimensi | Elemen Diagram Visual | Diagram alur perubahan dimensi $7 x 7 x 1280 \to \text{GAP} \to 1280 x 1$. |

---

### 6.6. Sub-Halaman 06: Fully Connected / Dense Layer (`pages/06_fc.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 6 | Elemen Header Teks | Penanda "Tahap 6 dari 7" dan penjelasan lapisan Dense (128 neuron) dengan aktivasi ReLU dan Softmax. |
| 2 | Arsitektur & Parameter FC | Elemen Layout & Tabel | Tampilan skema arsitektur penghubung 1.280 elemen input ke 128 neuron Dense dan 4 logit output. |
| 3 | Visualisasi Grid Neuron Aktif | Elemen Visualisasi Grid | Grid petak 200 sampel neuron: warna ungu menandakan neuron aktif ($>0$) dan abu-abu non-aktif ($0$). |
| 4 | Display Rumus Logit & Softmax | Elemen Visualisasi Formula | Rumus gabungan perkalian bobot $z_k = \sum (X_i x W_{i,k}) + b_k$ dan normalisasi eksponensial $P(y_k) = \frac{e^{z_k}}{\sum e^{z_j}}$. |
| 5 | Panduan 3 Langkah Perhitungan Klasifikasi | Elemen Step-by-Step Info | Penjelasan urutan: Hitung Logit Mentah $z_k \to$ Ubah ke Eksponensial $e^{z_k} \to$ Probabilitas Softmax $(\%)$. |
| 6 | Tabel Perhitungan Matriks Dense & Softmax | Elemen Tabel Data | Tabel kalkulasi 4 kelas penyakit: nilai logit $z_k$, eksponensial $e^{z_k}$, dan persentase probabilitas akhir. |
| 7 | Akordion Rincian Matriks & Silsilah Keputusan | Elemen Interactive Dropdown | Lipatan penjelasan detail perkalian matriks bobot Dense dan rincian asal-usul pengambilan keputusan model. |

---

### 6.7. Sub-Halaman 07: Hasil Deteksi & Diagnosis (`pages/07_output.html`)
| No. | Nama Komponen | Jenis Elemen | Deskripsi |
| :---: | :--- | :--- | :--- |
| 1 | Header Langkah 7 | Elemen Header Teks | Penanda "Tahap 7 dari 7" ("Hasil Deteksi Penyakit") dan status penyelesaian analisis CNN. |
| 2 | Banner Hasil Diagnosa (Result Banner) | Elemen Banner Notifikasi | Banner kebiruan/teal penanda deteksi selesai dengan tag label "Akurat". |
| 3 | Preview Gambar Input Teranalisis | Elemen Tampilan Gambar | Frame foto sampel daun yang telah dianalisis lengkap dengan catatan status area teranalisis. |
| 4 | Bar Probabilitas Tiap Kelas Penyakit | Elemen Visualisasi Progres | Progress bar horizontal berwarna per kelas penyakit (*Bercak Ungu*, *Moler*, *Sehat*, dll.). |
| 5 | Canvas Grafik Probabilitas Output | Elemen Visualisasi Chart | Chart canvas yang menampilkan grafik perbandingan probabilitas antar kelas. |
| 6 | Card Kepercayaan Terbesar (Big Confidence) | Elemen Highlight Informasi | Kartu dengan teks ukuran besar yang menyoroti Nama Kelas Terdeteksi, Nama Latin, dan Angka Persentase Confidence. |
| 7 | Kartu Rekomendasi Penanganan | Elemen Tampilan Informasi | Kartu daftar langkah penanganan dan solusi pengobatan yang disarankan berdasarkan jenis penyakit yang terdeteksi. |
| 8 | Tabel Ringkasan Proses CNN | Elemen Tabel Parameter | Tabel komparatif alur komputasi dari input $224 x 224 x 3$, 3 blok konvolusi, pooling, GAP, FC, hingga Softmax output. |
| 9 | Tombol Deteksi Ulang & Unduh PDF | Elemen Interaktif (Button) | Tombol sekunder *Deteksi Ulang* (`resetAll()`) dan tombol utama *Unduh Laporan PDF* (`unduhLaporanPdf()`). |
