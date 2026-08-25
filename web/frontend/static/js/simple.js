// ============================================================
// simple.js — logika halaman deteksi.html (versi sederhana untuk petani)
//
// Beda dengan main.js/pipeline.js (dipakai index.html versi 7-tahap):
// halaman ini HANYA punya 2 langkah yang terlihat pengguna — unggah/
// ambil foto, lalu hasil deteksi. Tidak ada tahap konvolusi, ReLU,
// pooling, dst. yang ditampilkan. index.html versi detail TETAP ada
// dan tidak diubah alurnya sama sekali; ini cuma pintu masuk kedua.
// ============================================================

const metaApi = document.querySelector('meta[name="api-base"]');
const API_BASE = metaApi ? metaApi.content : 'http://localhost:5000';

let sessionId = null;
let uploadedImageDataUrl = null;

function showToast(msg, type = 'info') {
  const existing = document.getElementById('toast');
  if (existing) existing.remove();
  const colors = { info:'#534AB7', success:'#1D9E75', error:'#ef4444' };
  const t = document.createElement('div');
  t.id = 'toast';
  t.style.cssText = `
    position:fixed;bottom:24px;right:24px;z-index:9999;
    background:${colors[type]||colors.info};color:white;
    padding:12px 18px;border-radius:10px;font-size:13px;font-weight:500;
    box-shadow:0 4px 16px rgba(0,0,0,.2);max-width:320px;line-height:1.4;
  `;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3500);
}

function showScreen(name) {
  document.getElementById('uploadState').style.display  = name === 'upload'  ? 'block' : 'none';
  document.getElementById('loadingState').style.display = name === 'loading' ? 'block' : 'none';
  document.getElementById('resultState').style.display  = name === 'result'  ? 'block' : 'none';
}

function triggerFileInput() {
  const el = document.getElementById('fileInput');
  if (el) el.click();
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) uploadFile(file);
}

// Drag & drop
window.addEventListener('DOMContentLoaded', () => {
  const zone = document.getElementById('uploadZone');
  if (!zone) return;
  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.style.borderColor = '#534AB7';
    zone.style.background  = '#F3F2FE';
  });
  zone.addEventListener('dragleave', () => {
    zone.style.borderColor = '';
    zone.style.background  = '';
  });
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.style.borderColor = '';
    zone.style.background  = '';
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
});

async function uploadFile(file) {
  const allowed = ['image/jpeg', 'image/png', 'image/jpg'];
  if (!allowed.includes(file.type)) {
    showToast('Format tidak didukung. Gunakan JPG atau PNG.', 'error'); return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('Ukuran foto melebihi 10 MB.', 'error'); return;
  }

  // Tampilkan preview segera sambil upload berjalan
  const reader = new FileReader();
  reader.onload = e => {
    uploadedImageDataUrl = e.target.result;
    const preview = document.getElementById('imgPreviewEl');
    if (preview) { preview.src = e.target.result; preview.style.display = 'block'; }
    const pc = document.getElementById('previewCard'); if (pc) pc.style.display = 'block';
    const ep = document.getElementById('emptyPreview'); if (ep) ep.style.display = 'none';
  };
  reader.readAsDataURL(file);

  showScreen('loading');

  try {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form, mode: 'cors' });
    if (!res.ok) throw new Error(`Server merespons dengan status ${res.status}`);
    const data = await res.json();
    if (!data.success) { showToast(data.error || 'Upload gagal.', 'error'); showScreen('upload'); return; }

    sessionId = data.session_id;
    await jalankanPrediksi();
  } catch (err) {
    console.error('Upload error:', err);
    showToast('Tidak dapat terhubung ke server. Pastikan backend Flask menyala.', 'error');
    showScreen('upload');
  }
}

async function jalankanPrediksi() {
  if (!sessionId) return;
  try {
    const res  = await fetch(`${API_BASE}/api/predict?session_id=${sessionId}`);
    const data = await res.json();
    if (data.success) {
      renderResult(data.data);
      showScreen('result');
    } else {
      showToast(data.error || 'Prediksi gagal.', 'error');
      showScreen('upload');
    }
  } catch (err) {
    console.warn('Prediksi error:', err);
    showToast('Gagal memproses foto. Coba lagi.', 'error');
    showScreen('upload');
  }
}

function renderResult(d) {
  const outImg = document.getElementById('outputPreviewImg');
  if (outImg && uploadedImageDataUrl) outImg.src = uploadedImageDataUrl;

  const banner = document.getElementById('resultBanner');
  const bTitle = document.getElementById('bannerTitle');
  const bSub   = document.getElementById('bannerSub');
  if (d.rejected) {
    banner.style.background = '#fef2f2';
    banner.style.border     = '1px solid #fecaca';
    bTitle.textContent = 'Gambar Tidak Dikenali';
    bTitle.style.color = '#991b1b';
    bSub.textContent = d.rejection_message || 'Gambar tidak terdeteksi sebagai daun/umbi bawang. Coba unggah ulang foto yang lebih jelas.';
    bSub.style.color = '#b91c1c';
  } else if (d.is_simulation) {
    banner.style.background = '#fffbeb';
    banner.style.border     = '1px solid #fde68a';
    bTitle.textContent = 'Deteksi Selesai (Mode Simulasi)';
    bTitle.style.color = '#92400e';
    bSub.textContent = 'Model belum dilatih dengan dataset asli — hasil ini masih perkiraan sementara.';
    bSub.style.color = '#b45309';
  } else {
    banner.style.background = 'var(--teal-50)';
    banner.style.border     = '1px solid var(--teal-100)';
    bTitle.textContent = 'Deteksi Selesai';
    bTitle.style.color = 'var(--teal-700)';
    bSub.textContent = 'Berikut hasil analisis foto daun yang diunggah.';
    bSub.style.color = 'var(--teal-500)';
  }

  const outClass = document.getElementById('outClass');
  const outLatin = document.getElementById('outLatin');
  const outConf  = document.getElementById('outConfidence');
  const outFill  = document.getElementById('outConfFill');
  if (outClass) {
    outClass.textContent = d.rejected ? 'Tidak dikenali' : (d.predicted_class || '—');
    outClass.style.color = d.rejected ? '#dc2626' : (d.color || '');
  }
  if (outLatin) outLatin.textContent = d.rejected ? '—' : (d.predicted_latin || '—');
  if (outConf)  outConf.textContent  = (d.confidence != null ? d.confidence + '%' : '—');
  if (outFill)  outFill.style.width  = (d.confidence != null ? d.confidence : 0) + '%';

  const recoEl = document.getElementById('recoList');
  if (recoEl && d.rejected) {
    recoEl.innerHTML = `<div style="font-size:13px;color:var(--text-secondary);">
      Unggah ulang foto yang menampilkan daun atau umbi bawang dengan jelas, pencahayaan cukup, dan tidak buram.
    </div>`;
  } else if (recoEl && Array.isArray(d.rekomendasi)) {
    recoEl.innerHTML = '';
    d.rekomendasi.forEach(text => {
      const item = document.createElement('div');
      item.className = 'reco-item';
      item.style.background = 'var(--bg)';
      item.innerHTML = `
        <div class="reco-icon" style="background:${d.color || '#534AB7'};">
          <svg viewBox="0 0 24 24" style="stroke:white;fill:none;stroke-width:2.5;"><polyline points="20 6 9 17 4 12"/></svg>
        </div>
        <div class="reco-text">${text}</div>`;
      recoEl.appendChild(item);
    });
  }
}

function resetSimple() {
  if (sessionId) fetch(`${API_BASE}/api/reset?session_id=${sessionId}`, { method: 'DELETE' }).catch(() => {});
  sessionId = null;
  uploadedImageDataUrl = null;

  const zone = document.getElementById('uploadZone');
  if (zone) zone.className = 'upload-zone';
  const preview = document.getElementById('imgPreviewEl');
  if (preview) { preview.src = ''; preview.style.display = 'none'; }
  const pc = document.getElementById('previewCard'); if (pc) pc.style.display = 'none';
  const ep = document.getElementById('emptyPreview'); if (ep) ep.style.display = 'block';
  const fi = document.getElementById('fileInput'); if (fi) fi.value = '';

  showScreen('upload');
}

window.addEventListener('DOMContentLoaded', () => showScreen('upload'));
