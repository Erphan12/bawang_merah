// ============================================================
// report.js — logika halaman report.html (dokumentasi hasil pelatihan)
// ============================================================
const metaApi = document.querySelector('meta[name="api-base"]');
const API_BASE = metaApi ? metaApi.content : 'http://localhost:5000';

// Halaman ini hanya untuk admin yang sudah login (lihat admin-auth.js).
if (requireAdminAuth()) {
  window.addEventListener('DOMContentLoaded', loadReport);
}

async function loadReport() {
  try {
    const res = await adminFetch(`${API_BASE}/api/admin/train/report`);
    const data = await res.json();
    if (!data.success) {
      document.getElementById('emptyState').style.display = 'flex';
      return;
    }
    render(data.report);
  } catch (e) {
    document.getElementById('emptyState').style.display = 'flex';
    console.warn('Gagal memuat laporan:', e);
  }
}

function render(r) {
  document.getElementById('reportBody').style.display = 'block';

  // ── Ringkasan ──
  document.getElementById('rAcc').textContent = (r.evaluation.overall_accuracy * 100).toFixed(1) + '%';
  document.getElementById('rTotal').textContent = r.dataset_stats.total_images.toLocaleString('id');
  document.getElementById('rClasses').textContent = r.classes.length;
  document.getElementById('rTrainedAt').textContent = 'Dilatih pada ' + new Date(r.trained_at).toLocaleString('id-ID');

  // ── Konfigurasi pelatihan yang benar-benar dipakai (otomatis) ──
  const hpTop = r.hyperparameters;
  document.getElementById('rEpochs').textContent = `${hpTop.epochs_ran} / ${hpTop.max_epochs} maks`;
  document.getElementById('rBatch').textContent  = hpTop.batch_size;
  document.getElementById('rLr').textContent     = hpTop.learning_rate;
  document.getElementById('rHpReasoning').textContent = hpTop.reasoning;

  // ── Kurva akurasi & loss ──
  const epochs = r.history.loss.map((_, i) => 'Epoch ' + (i + 1));
  new Chart(document.getElementById('chartAcc'), {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: 'Akurasi Latih', data: r.history.accuracy, borderColor: '#534AB7', backgroundColor: 'transparent', tension: .3 },
        { label: 'Akurasi Validasi', data: r.history.val_accuracy, borderColor: '#1D9E75', backgroundColor: 'transparent', tension: .3 },
      ],
    },
    options: { scales: { y: { min: 0, max: 1 } }, plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } } },
  });
  new Chart(document.getElementById('chartLoss'), {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: 'Loss Latih', data: r.history.loss, borderColor: '#d97706', backgroundColor: 'transparent', tension: .3 },
        { label: 'Loss Validasi', data: r.history.val_loss, borderColor: '#dc2626', backgroundColor: 'transparent', tension: .3 },
      ],
    },
    options: { plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } } },
  });

  // ── Confusion matrix ──
  renderConfusionMatrix(r.evaluation.confusion_matrix, r.evaluation.confusion_matrix_labels);

  // ── Precision/Recall/F1 ──
  const table = document.getElementById('perClassTable');
  table.innerHTML = '<tr><td><strong>Kelas</strong></td><td><strong>Precision</strong></td><td><strong>Recall</strong></td><td><strong>F1-Score</strong></td><td><strong>Jumlah data uji</strong></td></tr>';
  r.evaluation.per_class.forEach(p => {
    table.innerHTML += `<tr><td>${p.class}</td><td>${p.precision}</td><td>${p.recall}</td><td>${p.f1}</td><td>${p.support}</td></tr>`;
  });

  // ── Distribusi dataset ──
  const splitLabels = r.classes;
  const trainData = splitLabels.map(c => r.dataset_stats.split.train[c] || 0);
  const valData = splitLabels.map(c => r.dataset_stats.split.val[c] || 0);
  const testData = splitLabels.map(c => r.dataset_stats.split.test[c] || 0);
  new Chart(document.getElementById('chartDist'), {
    type: 'bar',
    data: {
      labels: splitLabels,
      datasets: [
        { label: 'Train', data: trainData, backgroundColor: '#534AB7' },
        { label: 'Validation', data: valData, backgroundColor: '#1D9E75' },
        { label: 'Test', data: testData, backgroundColor: '#d97706' },
      ],
    },
    options: { plugins: { legend: { position: 'bottom', labels: { font: { size: 11 } } } }, scales: { x: { stacked: true }, y: { stacked: true } } },
  });

  // ── Detail teknis (tersembunyi di <details>) ──
  const hp = r.hyperparameters;
  const tech = document.getElementById('techTable');
  tech.innerHTML = `
    <tr><td>Epoch berjalan (dari maks. ${hp.max_epochs}, berhenti otomatis)</td><td>${hp.epochs_ran}</td></tr>
    <tr><td>Batch size</td><td>${hp.batch_size}</td></tr>
    <tr><td>Learning rate awal</td><td>${hp.learning_rate}</td></tr>
    <tr><td>Early stopping patience</td><td>${hp.early_stopping_patience} epoch</td></tr>
    <tr><td>Total gambar dataset</td><td>${r.dataset_stats.total_images}</td></tr>
  `;
  document.getElementById('techReasoning').textContent = hp.reasoning + ' ' + hp.note;
}

function renderConfusionMatrix(cm, labels) {
  const wrap = document.getElementById('confMatrixWrap');
  const max = Math.max(1, ...cm.flat());
  let html = '<table class="cm-table"><tr><td class="cm-corner"></td>';
  labels.forEach(l => { html += `<th>Prediksi: ${l}</th>`; });
  html += '</tr>';
  cm.forEach((row, i) => {
    html += `<tr><td class="cm-axis-label">Aktual: ${labels[i]}</td>`;
    row.forEach(val => {
      const t = val / max;
      const bg = `rgba(83,74,183,${0.15 + t * 0.7})`;
      html += `<td class="cm-cell" style="background:${bg};">${val}</td>`;
    });
    html += '</tr>';
  });
  html += '</table>';
  wrap.innerHTML = html;
}
