// ============================================================
// KONFIGURASI API & STATE GLOBAL
// ============================================================
const metaApi = document.querySelector('meta[name="api-base"]');
const API_BASE = (metaApi && metaApi.content && !metaApi.content.includes('localhost')) 
  ? metaApi.content 
  : window.location.origin;

let currentPage = 0;
let sessionId = null;       // ID sesi dari backend setelah upload
let prediksiData = null;       // Hasil /api/predict
let layerStats = null;       // Hasil /api/layer-stats
let fmapData = null;       // Hasil /api/feature-maps
let classNames = [];         // Daftar kelas aktif (dari /api/classes) — dinamis, bukan hardcode
let uploadedImageDataUrl = null; // Foto yang diupload, disimpan supaya bisa dipakai lagi di halaman hasil (07_output)

// Promise gabungan dari 3 fetch background (predict, layer-stats, feature-maps).
// Diisi saat upload selesai. goTo() akan menunggu promise ini SEBELUM merender
// halaman tahap 1-6, supaya halaman tidak pernah menampilkan data ilustrasi/acak
// hanya karena fetch backend belum selesai (lihat fix race condition di goTo()).
let dataReadyPromise = null;

async function loadClassNames() {
  try {
    const res = await fetch(`${API_BASE}/api/classes`);
    const data = await res.json();
    if (data.success) classNames = data.classes;
  } catch (err) { console.warn('Gagal memuat daftar kelas:', err); }
}

// Global error handling
window.addEventListener('error', (e) => {
  console.error('Global error:', e);
  showError('app-content', `Terjadi kesalahan: ${e.message}`);
});
window.addEventListener('unhandledrejection', (e) => {
  console.error('Unhandled promise rejection:', e.reason);
  showError('app-content', `Terjadi kesalahan: ${e.reason}`);
});

// ============================================================
// FILE PATHS UNTUK FETCH
// ============================================================
const pageFiles = [
  'pages/01_input.html',
  'pages/02_conv.html',
  'pages/03_relu.html',
  'pages/04_pooling.html',
  'pages/05_flatten.html',
  'pages/06_fc.html',
  'pages/07_output.html'
];

// ============================================================
// HELPER UI
// ============================================================
function showToast(msg, type = 'info') {
  const existing = document.getElementById('toast');
  if (existing) existing.remove();
  const colors = { info: '#534AB7', success: '#1D9E75', error: '#ef4444', loading: '#d48a0c' };
  const t = document.createElement('div');
  t.id = 'toast';
  t.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:${colors[type] || colors.info};color:white;
    padding:12px 18px;border-radius:10px;font-size:13px;font-weight:500;
    box-shadow:0 4px 16px rgba(0,0,0,.2);max-width:320px;line-height:1.4;
    display:flex;align-items:center;gap:8px;
    animation:slideIn .25s ease;
  `;
  const icons = { info: 'ℹ', success: '✓', error: '✕', loading: '⏳' };
  t.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${msg}</span>`;
  document.body.appendChild(t);
  if (type !== 'loading') setTimeout(() => t.remove(), 3500);
  return t;
}

function showError(elId, msg) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.innerHTML = `
    <div style="padding:16px;background:var(--red-50);border-radius:8px;
      color:var(--red-700);font-size:12px;text-align:center;">
      ⚠ ${msg}
    </div>`;
}

// ============================================================
// NAVIGASI DENGAN FETCH API
// ============================================================
// Overlay ringan di atas #app-content selagi menunggu data ASLI dari backend
// (predict / layer-stats / feature-maps). Dipakai oleh goTo() supaya tahap 1-6
// tidak pernah dirender dengan data ilustrasi/acak hanya karena fetch belum selesai.
function showStageLoading(container, show) {
  let overlay = document.getElementById('stageLoadingOverlay');
  if (show) {
    if (!overlay) {
      overlay = document.createElement('div');
      overlay.id = 'stageLoadingOverlay';
      overlay.style.cssText = `
        position:fixed; inset:0; z-index:9999;
        background:rgba(255,255,255,0.85);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:10px; font-size:13px; color:#3C3489; font-weight:600;
      `;
      overlay.innerHTML = `
        <div style="width:28px;height:28px;border:3px solid #E5E3FD;border-top-color:#3C3489;border-radius:50%;animation:stageSpin 0.8s linear infinite;"></div>
        <div>Memuat data asli dari model...</div>
        <style>@keyframes stageSpin{to{transform:rotate(360deg)}}</style>
      `;
      document.body.appendChild(overlay);
    }
  } else if (overlay) {
    overlay.remove();
  }
}

async function goTo(n) {
  if (n > 0 && !sessionId) {
    showToast('Upload gambar terlebih dahulu.', 'error');
    return;
  }
  if (n === 6 && !prediksiData) {
    showToast('Prediksi belum selesai. Tunggu sebentar.', 'error');
    return;
  }

  // A. Update Visual Progress Bar
  for (let i = 0; i < 7; i++) {
    const dot = document.getElementById('sd' + i);
    const step = dot.closest('.step');
    dot.className = i < n ? 'step-dot done' : i === n ? 'step-dot active' : 'step-dot idle';
    dot.innerHTML = i < n ? '✓' : (i + 1).toString();

    if (step) {
      step.classList.toggle('active', i === n);
      step.classList.toggle('done', i < n);
    }
    if (i < 6) {
      const line = document.getElementById('sl' + i);
      if (line) line.className = 'step-line' + (i < n ? ' done' : '');
    }
  }

  currentPage = n;
  window.scrollTo({ top: 0, behavior: 'smooth' });

  // B. Proses Fetching HTML Dinamis
  const contentContainer = document.getElementById('app-content');
  try {
    const pageUrl = `${API_BASE}/${pageFiles[n]}`;
    const response = await fetch(pageUrl, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Gagal memuat ${pageFiles[n]} (Status: ${response.status})`);

    contentContainer.innerHTML = await response.text();

    // C. Untuk tahap 1-6, tunggu data ASLI dari backend (predict, layer-stats,
    // feature-maps) selesai dulu sebelum inisialisasi chart/feature map.
    // Ini mencegah halaman sempat menampilkan ilustrasi/acak lalu "berubah
    // sendiri" jadi data asli begitu pengguna kembali ke halaman yang sama.
    if (n > 0 && dataReadyPromise) {
      showStageLoading(contentContainer, true);
      try { await dataReadyPromise; } catch (e) { /* tetap lanjut, fallback ilustrasi kalau memang gagal */ }
      showStageLoading(contentContainer, false);
    }

    // D. Inisialisasi Chart/Grafik & Binding Event (SETELAH HTML dirender & data siap)
    if (n === 0) initInputEvents();
    if (n === 1 && typeof initConv === 'function') initConv();
    if (n === 2 && typeof initRelu === 'function') initRelu();
    if (n === 3 && typeof initPool === 'function') initPool();
    if (n === 4 && typeof initFlat === 'function') initFlat();
    if (n === 5 && typeof initFC === 'function') initFC();
    if (n === 6 && typeof initOutput === 'function') initOutput();

  } catch (error) {
    console.error(error);
    showError('app-content', `Error: ${error.message}. Pastikan Local Server berjalan.`);
  }
}

function resetAll() {
  if (sessionId) {
    fetch(`${API_BASE}/api/reset?session_id=${sessionId}`, { method: 'DELETE' }).catch(() => { });
  }
  sessionId = null;
  prediksiData = null;
  layerStats = null;
  fmapData = null;
  dataReadyPromise = null;
  uploadedImageDataUrl = null;
  goTo(0);
}

// ============================================================
// UPLOAD & EVENT BINDING HALAMAN INPUT (n = 0)
// ============================================================
function initInputEvents() {
  const zone = document.getElementById('uploadZone');
  if (!zone) return;

  // Binding Drag & Drop
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.style.borderColor = '#534AB7';
    zone.style.background = '#F3F2FE';
  });
  zone.addEventListener('dragleave', () => {
    zone.style.borderColor = '';
    zone.style.background = '';
  });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.style.borderColor = '';
    zone.style.background = '';
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
}

function triggerFileInput() {
  const fileInput = document.getElementById('fileInput');
  if (fileInput) fileInput.click();
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadFile(file);
}

async function uploadFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
  if (!allowed.includes(file.type)) {
    showToast('Format tidak didukung. Gunakan JPG atau PNG.', 'error'); return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('Ukuran file melebihi 10 MB.', 'error'); return;
  }

  const toast = showToast('⏳ Mengunggah gambar...', 'loading');

  try {
    const form = new FormData();
    form.append('file', file);

    // Pastikan request menggunakan mode 'cors'
    const res = await fetch(`${API_BASE}/api/upload`, {
      method: 'POST',
      body: form,
      mode: 'cors'
    });

    if (!res.ok) throw new Error(`Server merespons dengan status ${res.status}`);

    const data = await res.json();
    toast.remove();

    if (!data.success) { showToast(data.error || 'Upload gagal.', 'error'); return; }

    // Reset data cache sesi lama agar tidak tercampur dengan gambar baru
    prediksiData = null;
    layerStats = null;
    fmapData = null;

    sessionId = data.session_id;
    showToast('Gambar berhasil diunggah!', 'success');
    tampilkanPreview(file, data);

    // Smooth scroll ke tombol proses untuk kemudahan UX pengguna
    setTimeout(() => {
      const btn = document.getElementById('btnProses');
      if (btn) btn.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);

    // Jalankan prediksi terlebih dahulu agar cache tersimpan, baru ambil statistik layer & feature maps
    dataReadyPromise = (async () => {
      await jalankanPrediksi();
      await Promise.all([ambilLayerStats(), ambilFeatureMaps()]);
    })();

  } catch (err) {
    toast.remove();
    showToast('Gagal terhubung ke server backend FastAPI/Uvicorn.', 'error');
    console.error('Upload error:', err);
  }
}

function tampilkanPreview(file, data) {
  const zone = document.getElementById('uploadZone');
  if (zone) {
    zone.className = 'upload-zone has-file';
    zone.innerHTML = `
      <div style="color:var(--teal-700);font-weight:600;font-size:14px;">
        ✓ ${data.filename} berhasil diunggah
      </div>
      <div style="font-size:12px;color:var(--teal-500);margin-top:4px;">
        ${data.file_size_kb} KB · ${data.original_size}
      </div>`;
  }

  const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };

  const reader = new FileReader();
  reader.onload = e => {
    uploadedImageDataUrl = e.target.result; // dipakai lagi nanti di kartu "Gambar Input Teranalisis" (07_output)
    const preview = document.getElementById('imgPreviewEl');
    if (preview) {
      preview.src = e.target.result;
      preview.style.display = 'block';
      const placeholder = document.getElementById('imgPlaceholder');
      if (placeholder) placeholder.style.display = 'none';
    }

    // Hitung resolusi asli gambar langsung di browser (tidak bergantung pada backend)
    const imgProbe = new Image();
    imgProbe.onload = () => {
      const resolusi = `${imgProbe.naturalWidth}x${imgProbe.naturalHeight}`;
      setEl('metaResolution', resolusi);

      const zoneInfo = document.querySelector('#uploadZone div[style*="teal-500"]');
      if (zoneInfo && (!data.original_size || data.original_size === 'unknown')) {
        zoneInfo.textContent = `${data.file_size_kb} KB · ${resolusi}`;
      }
    };
    imgProbe.onerror = () => {
      setEl('metaResolution', data.original_size || '—');
    };
    imgProbe.src = e.target.result;
  };
  reader.readAsDataURL(file);

  setEl('metaFilename', data.filename);
  setEl('metaSize', data.file_size_kb + ' KB');
  setEl('metaResolution', (data.original_size && data.original_size !== 'unknown') ? data.original_size : '…');

  if (document.getElementById('previewCard')) document.getElementById('previewCard').style.display = 'block';
  if (document.getElementById('emptyPreview')) document.getElementById('emptyPreview').style.display = 'none';
  if (document.getElementById('btnProses')) document.getElementById('btnProses').style.display = 'flex';
}

// ============================================================
// FUNGSI KOMUNIKASI BACKEND
// ============================================================
async function jalankanPrediksi() {
  if (!sessionId) return;
  try {
    const res = await fetch(`${API_BASE}/api/predict?session_id=${sessionId}`);
    const data = await res.json();
    if (data.success) {
      prediksiData = data;
    }
  } catch (err) { console.warn('Prediksi error:', err); }
}

async function ambilLayerStats() {
  if (!sessionId) return;
  try {
    const res = await fetch(`${API_BASE}/api/layer-stats?session_id=${sessionId}`);
    const data = await res.json();
    if (data.success) layerStats = data.data;   // unwrap {success, data} -> data asli
  } catch (err) { console.warn('Layer stats error:', err); }
}

async function ambilFeatureMaps() {
  if (!sessionId) return;
  try {
    const res = await fetch(`${API_BASE}/api/feature-maps?session_id=${sessionId}&n_filters=32`);
    const data = await res.json();
    if (data.success) fmapData = data.data;   // unwrap {success, data} -> data asli
  } catch (err) { console.warn('Feature maps error:', err); }
}

// ============================================================
// INISIALISASI SAAT PERTAMA KALI DIBUKA
// ============================================================
window.addEventListener('DOMContentLoaded', () => {
  loadClassNames();
  goTo(0);
});