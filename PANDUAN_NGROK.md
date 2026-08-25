# 🚀 Panduan Mandiri: Menjalankan Website Online via Ngrok

Panduan ini dibuat agar Anda dapat menjalankan server website Klasifikasi Penyakit Bawang Merah secara mandiri kapan saja dari komputer Anda tanpa bantuan AI.

---

## 📋 Prasyarat Lengkap

Sebelum memulai, pastikan:
1. Python sudah terinstall di komputer.
2. Ngrok sudah terinstall (bisa dipanggil via terminal).

---

## 🛠️ Langkah-Langkah Menjalankan Website (2 Terminal)

Untuk menjalankan website agar bisa diakses publik via internet, Anda membutuhkan **2 jendela Terminal / PowerShell**.

---

### 🔹 TERMINAL 1: Jalankan Server FastAPI (Backend & Web)

1. Buka **PowerShell** atau **Command Prompt**.
2. Masuk ke folder proyek Anda:
   ```powershell
   cd d:\TUTUP\bawang_merah
   ```
3. Jalankan perintah Uvicorn berikut:
   ```powershell
   uvicorn src.main_api:app --reload --host 127.0.0.1 --port 8000
   ```
4. **Ciri berhasil:**
   Akan muncul tulisan:
   `INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`
   
> ⚠️ **Penting:** Jendela Terminal 1 ini **JANGAN DITUTUP** selama Anda ingin website tetap aktif!

---

### 🔹 TERMINAL 2: Jalankan Ngrok Tunnel (Akses Online Public)

1. Buka **PowerShell** atau **Command Prompt** baru (Jendela kedua).
2. Masuk ke folder proyek Anda:
   ```powershell
   cd d:\TUTUP\bawang_merah
   ```
3. Jalankan perintah Ngrok berikut:
   ```powershell
   ngrok http 8000
   ```
4. **Ciri berhasil:**
   Layar terminal akan berubah menampilkan status Ngrok seperti ini:
   ```text
   Session Status        online
   Account               NamaAnda (Plan: Free)
   Forwarding            https://xxxx-xxxx-xxxx.ngrok-free.dev -> http://localhost:8000
   ```

---

## 🌐 Cara Mengakses Website

1. Salin/Copy link HTTPS yang ada di sebelah tulisan **`Forwarding`** (contoh: `https://xxxx-xxxx-xxxx.ngrok-free.dev`).
2. Bagikan link tersebut ke pengguna lain / buka di HP Anda.
3. **Catatan Pertama Kali Akses:** 
   Saat pertama kali membuka link ngrok di browser, akan muncul halaman peringatan dari Ngrok. Klik tombol **"Visit Site"** untuk masuk ke dalam website.

---

## 📊 Cara Cek Link Ngrok via Browser (Alternatif)

Jika Anda lupa link Ngrok yang sedang berjalan:
1. Buka browser di komputer Anda.
2. Akses alamat: **`http://localhost:4040`**
3. Anda bisa melihat **Public URL** dan memantau semua trafik lalu lintas gambar yang diunggah pengguna secara *real-time*.

---

## 🛑 Cara Mematikan / Menghentikan Server

Jika sudah selesai digunakan:
1. Buka Terminal 1, tekan **`CTRL + C`** di keyboard.
2. Buka Terminal 2, tekan **`CTRL + C`** di keyboard.
3. Tutup kedua jendela terminal.

---

## ❓ FAQ & Troubleshooting (Penyelesaian Masalah)

### 1. Kenapa Link Ngrok Berubah Saat Dijalankan Ulang?
Akun Ngrok versi gratis (*Free Account*) akan menghasilkan URL acak yang baru setiap kali Anda mematikan dan menjalankan ulang perintah `ngrok http 8000`. Jika ingin URL tetap (static domain), Anda bisa mendaftar domain gratis di dashboard Ngrok dan menjalankannya dengan:
```powershell
ngrok http --url=domain-kamu.ngrok.dev 8000
```

### 2. Peringatan "Redirection / CORS Error"?
Semua file frontend sudah diperbaiki menggunakan `window.location.origin`, sehingga website akan secara otomatis menyesuaikan link API dengan domain Ngrok yang aktif tanpa perlu mengubah isi kode lagi!
