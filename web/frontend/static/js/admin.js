// ============================================================
// admin.js — logika halaman admin.html (upload dataset & training)
// Sengaja TIDAK pernah menampilkan epoch/batch size mentah ke UI ini;
// hanya persentase & pesan umum. Detail teknis lengkap ada di report.html
// (di dalam <details>, tersembunyi secara default).
// ============================================================
const metaApi = document.querySelector('meta[name="api-base"]');
const API_BASE = metaApi ? metaApi.content : 'http://localhost:5000';

let pollTimer = null;

// Halaman ini hanya untuk admin yang sudah login -- lempar ke login.html
// kalau belum ada token tersimpan (lihat admin-auth.js).
if (requireAdminAuth()) {
  window.addEventListener('DOMContentLoaded', () => {
    loadStatusSummary();
    bindDragDrop();
    loadAdminUsername();
  });
}

// ============================================================
// Riwayat prediksi — dibaca langsung dari predictions.db (SQLite)
// lewat endpoint /api/history dan /api/history/stats.
// Section ini sengaja disembunyikan secara default dan baru dimuat
// saat admin menekan tombol "Lihat Riwayat Prediksi" di topbar,
// supaya halaman admin tidak langsung berat/penuh saat dibuka.
// ============================================================
let historyLoadedOnce = false;

function toggleHistorySection() {
  const section = document.getElementById('historySection');
  const btn = document.getElementById('btnToggleHistory');
  const isHidden = section.style.display === 'none';

  if (isHidden) {
    section.style.display = 'block';
    btn.innerHTML = `
      <svg viewBox="0 0 24 24"><polyline points="18 15 12 9 6 15"/></svg>
      Sembunyikan Riwayat
    `;
    if (!historyLoadedOnce) {
      historyLoadedOnce = true;
      loadHistory();
    }
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    section.style.display = 'none';
    btn.innerHTML = `
      <svg viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.1-3-3-3.6 3.6"/></svg>
      Lihat Riwayat Prediksi
    `;
  }
}
async function loadHistory() {
  const tbody = document.getElementById('historyTableBody');
  const emptyBox = document.getElementById('historyEmpty');
  tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-secondary);padding:16px;">Memuat riwayat...</td></tr>';
  emptyBox.style.display = 'none';

  try {
    const [histRes, statsRes] = await Promise.all([
      adminFetch(`${API_BASE}/api/history?limit=100`),
      adminFetch(`${API_BASE}/api/history/stats`),
    ]);
    const hist = await histRes.json();
    const stats = await statsRes.json();

    if (!hist.success) throw new Error(hist.error || 'Gagal memuat riwayat');

    // --- Statistik ringkas ---
    document.getElementById('histTotal').textContent = hist.count;
    const distBox = document.getElementById('histDistribution');
    distBox.innerHTML = '';
    if (stats.success) {
      stats.distribution.forEach(({ predicted_class, total }) => {
        const box = document.createElement('div');
        box.className = 'stat-box';
        box.innerHTML = `<div class="stat-val">${total}</div><div class="stat-lbl">${predicted_class}</div>`;
        distBox.appendChild(box);
      });
    }

    // --- Tabel riwayat ---
    if (hist.history.length === 0) {
      tbody.innerHTML = '';
      emptyBox.style.display = 'flex';
      return;
    }

    tbody.innerHTML = hist.history.map(row => `
      <tr>
        <td>${row.id}</td>
        <td>${formatDate(row.created_at)}</td>
        <td>${row.filename ?? '-'}</td>
        <td>${row.predicted_class ?? '-'}</td>
        <td>${row.confidence != null ? Number(row.confidence).toFixed(1) + '%' : '-'}</td>
        <td>${row.is_simulation ? 'Simulasi' : (row.mode ?? 'Model')}</td>
        <td>${row.file_size_kb != null ? row.file_size_kb + ' KB' : '-'}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error(err);
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--red-600, #c0392b);padding:16px;">Gagal memuat riwayat: ${err.message}</td></tr>`;
  }
}

function formatDate(iso) {
  if (!iso) return '-';
  try {
    return new Date(iso).toLocaleString('id-ID');
  } catch {
    return iso;
  }
}

async function clearHistory() {
  if (!confirm('Hapus semua riwayat prediksi? Tindakan ini tidak bisa dibatalkan.')) return;
  try {
    const res = await adminFetch(`${API_BASE}/api/admin/history/clear`, { method: 'DELETE' });
    const data = await res.json();
    if (!data.success) {
      alert(data.error || 'Gagal menghapus riwayat.');
      return;
    }
    loadHistory();
  } catch (err) {
    alert('Tidak dapat terhubung ke server.');
  }
}

async function loadStatusSummary() {
  try {
    const res = await fetch(`${API_BASE}/api/classes`);
    const data = await res.json();
    if (!data.success) return;
    document.getElementById('sumClasses').textContent = data.classes.join(', ');
    document.getElementById('sumSource').textContent = data.source === 'trained' ? 'Model terlatih' : 'Default (belum dilatih)';
    document.getElementById('sumTrainedAt').textContent = data.trained_at
      ? new Date(data.trained_at).toLocaleString('id-ID')
      : '—';
    if (data.source === 'trained') {
      document.getElementById('reportLinkCard').style.display = 'block';
    }
  } catch (e) {
    console.warn('Gagal memuat status:', e);
  }
}

function bindDragDrop() {
  const zone = document.getElementById('zipZone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag');
    const file = e.dataTransfer.files[0];
    if (file) uploadZip(file);
  });
}

function handleZipSelect(e) {
  const file = e.target.files[0];
  if (file) uploadZip(file);
}

async function uploadZip(file) {
  if (!file.name.toLowerCase().endsWith('.zip')) {
    alert('File harus berformat .zip'); return;
  }
  const zone = document.getElementById('zipZone');
  zone.querySelector('.upload-title').textContent = 'Mengunggah & memeriksa dataset...';

  try {
    const form = new FormData();
    form.append('file', file);
    const res = await adminFetch(`${API_BASE}/api/admin/dataset/upload`, { method: 'POST', body: form });
    const data = await res.json();

    if (!data.success) {
      zone.querySelector('.upload-title').textContent = 'Seret & lepas dataset.zip di sini';
      alert(data.error || 'Upload dataset gagal.');
      return;
    }

    zone.querySelector('.upload-title').textContent = `✓ ${file.name} berhasil diunggah`;

    const wrap = document.getElementById('datasetSummary');
    const box = document.getElementById('classCounts');
    box.innerHTML = '';
    Object.entries(data.class_counts).forEach(([name, count]) => {
      const el = document.createElement('div');
      el.className = 'class-count-box';
      el.innerHTML = `<div class="cc-name">${name}</div><div class="cc-val">${count} gambar</div>`;
      box.appendChild(el);
    });

    // --- Ringkasan kompresi otomatis (dari model/preprocess.py) ---
    const compBox = document.getElementById('compressionSummary');
    if (compBox && data.compression) {
      const c = data.compression;
      compBox.style.display = 'flex';
      if (c.compressed > 0 && c.before_mb > 0) {
        const saved_pct = Math.round((1 - (c.after_mb / c.before_mb)) * 100);
        compBox.innerHTML = `
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          <div class="notice-text">
            Gambar dikompres otomatis: ${c.compressed} dari ${c.total_files} file
            (${c.before_mb} MB &rarr; ${c.after_mb} MB, hemat ~${saved_pct}%)${c.skipped > 0 ? `. ${c.skipped} file dilewati (rusak/bukan gambar).` : ''}
          </div>`;
      } else {
        compBox.innerHTML = `
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          <div class="notice-text">${c.total_files} file diperiksa, tidak ada yang perlu dikompres lagi.</div>`;
      }
    }

    wrap.style.display = 'block';
  } catch (err) {
    console.error(err);
    alert('Tidak dapat terhubung ke server. Pastikan backend Flask menyala.');
  }
}

async function startTraining() {
  const btn = document.getElementById('btnTrain');
  btn.disabled = true;
  document.getElementById('trainError').style.display = 'none';
  document.getElementById('trainDone').style.display = 'none';

  try {
    const res = await adminFetch(`${API_BASE}/api/admin/train/start`, { method: 'POST' });
    const data = await res.json();
    if (!data.success) {
      showTrainError(data.error || 'Gagal memulai pelatihan.');
      btn.disabled = false;
      return;
    }
    document.getElementById('trainProgress').style.display = 'block';
    pollStatus();
  } catch (err) {
    showTrainError('Tidak dapat terhubung ke server.');
    btn.disabled = false;
  }
}

function pollStatus() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await adminFetch(`${API_BASE}/api/admin/train/status`);
      const s = await res.json();

      document.getElementById('trainMsg').innerHTML = `<span class="spinner"></span>${s.message}`;
      document.getElementById('trainPct').textContent = s.progress_pct + '%';
      document.getElementById('trainFill').style.width = s.progress_pct + '%';

      if (s.status === 'done') {
        clearInterval(pollTimer);
        document.getElementById('btnTrain').disabled = false;
        document.getElementById('trainDoneMsg').textContent = s.message + ' ';
        document.getElementById('trainDone').style.display = 'flex';
        document.getElementById('reportLinkCard').style.display = 'block';
        loadStatusSummary();
      } else if (s.status === 'error') {
        clearInterval(pollTimer);
        document.getElementById('btnTrain').disabled = false;
        showTrainError(s.error || 'Pelatihan gagal karena kesalahan tak terduga.');
      }
    } catch (e) {
      console.warn('Polling status gagal:', e);
    }
  }, 2000);
}

function showTrainError(msg) {
  document.getElementById('trainErrorMsg').textContent = msg;
  document.getElementById('trainError').style.display = 'flex';
}

// ============================================================
// AKUN ADMIN — ubah username & password (hanya admin yang login yang
// bisa membukanya, dan tetap wajib memasukkan password lama yang benar).
// ============================================================
async function loadAdminUsername() {
  try {
    const res = await adminFetch(`${API_BASE}/api/admin/session`);
    const data = await res.json();
    if (data.success) {
      document.getElementById('accCurrentUsername').textContent = data.username;
    }
  } catch (e) {
    // adminFetch sudah menangani redirect kalau token tidak valid
  }
}

function toggleAccountForm() {
  const box = document.getElementById('accountFormBox');
  box.style.display = box.style.display === 'none' ? 'block' : 'none';
  document.getElementById('accountError').style.display = 'none';
  document.getElementById('accountSuccess').style.display = 'none';
}

async function submitAccountForm(event) {
  event.preventDefault();
  const currentPassword = document.getElementById('accCurrentPassword').value;
  const newUsername = document.getElementById('accNewUsername').value.trim();
  const newPassword = document.getElementById('accNewPassword').value;
  const newPasswordConfirm = document.getElementById('accNewPasswordConfirm').value;
  const errBox = document.getElementById('accountError');
  const okBox = document.getElementById('accountSuccess');
  errBox.style.display = 'none';
  okBox.style.display = 'none';

  if (!newUsername && !newPassword) {
    errBox.querySelector('.notice-text').textContent = 'Isi username baru dan/atau password baru terlebih dahulu.';
    errBox.style.display = 'flex';
    return false;
  }
  if (newPassword && newPassword !== newPasswordConfirm) {
    errBox.querySelector('.notice-text').textContent = 'Konfirmasi password baru tidak cocok.';
    errBox.style.display = 'flex';
    return false;
  }

  const btn = document.getElementById('accSubmitBtn');
  btn.disabled = true;
  try {
    const res = await adminFetch(`${API_BASE}/api/admin/account`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        current_password: currentPassword,
        new_username: newUsername,
        new_password: newPassword,
      }),
    });
    const data = await res.json();
    if (!data.success) {
      errBox.querySelector('.notice-text').textContent = data.error || 'Gagal memperbarui akun.';
      errBox.style.display = 'flex';
      btn.disabled = false;
      return false;
    }
    // Server mencabut semua token setelah kredensial diganti -- jadi admin
    // (termasuk sesi ini) wajib login ulang memakai kredensial baru.
    okBox.style.display = 'flex';
    setTimeout(() => { adminLogout(); }, 1500);
  } catch (e) {
    btn.disabled = false;
  }
  return false;
}
