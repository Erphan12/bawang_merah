// ============================================================
// report-pdf.js — bikin & unduh laporan PDF hasil deteksi (halaman 7)
// Dipakai oleh tombol "Unduh Laporan PDF" di 07_output.html.
// Client-side pakai jsPDF (tidak butuh endpoint backend baru).
// ============================================================

function _hexToRgb(hex) {
  if (!hex) return [83, 74, 183];
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  const n = parseInt(h, 16);
  if (Number.isNaN(n)) return [83, 74, 183];
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

async function unduhLaporanPdf() {
  const btn = document.getElementById('btnUnduhPdf');
  const btnOriginal = btn ? btn.innerHTML : null;

  if (!prediksiData || !prediksiData.data) {
    alert('Belum ada hasil deteksi untuk diunduh. Selesaikan proses deteksi terlebih dahulu.');
    return;
  }
  if (!window.jspdf || !window.jspdf.jsPDF) {
    alert('Gagal memuat komponen pembuat PDF. Periksa koneksi internet lalu coba lagi.');
    return;
  }

  if (btn) { btn.disabled = true; btn.innerHTML = 'Menyiapkan PDF…'; }

  try {
    const { jsPDF } = window.jspdf;
    const doc  = new jsPDF({ unit: 'mm', format: 'a4' });
    const d    = prediksiData.data;
    const pageW = doc.internal.pageSize.getWidth();
    const pageH = doc.internal.pageSize.getHeight();
    const mX = 15;
    let y = 18;

    const ensureSpace = (need) => {
      if (y + need > pageH - 15) { doc.addPage(); y = 18; }
    };

    // ── Header ──
    doc.setFont('helvetica', 'bold'); doc.setFontSize(15);
    doc.text('Laporan Hasil Deteksi Penyakit Bawang Merah', mX, y);
    y += 6;
    doc.setFont('helvetica', 'normal'); doc.setFontSize(9); doc.setTextColor(110);
    doc.text('Dibuat otomatis pada ' + new Date().toLocaleString('id-ID'), mX, y);
    doc.setTextColor(0);
    y += 9;

    // ── Gambar + ringkasan hasil ──
    let infoTop = y;
    let imgBottom = y;
    if (typeof uploadedImageDataUrl !== 'undefined' && uploadedImageDataUrl) {
      try {
        const props = doc.getImageProperties(uploadedImageDataUrl);
        const imgW = 55, imgH = (props.height / props.width) * imgW;
        doc.addImage(uploadedImageDataUrl, props.fileType || 'JPEG', mX, y, imgW, imgH);
        imgBottom = y + imgH;
      } catch (e) {
        console.warn('Gagal menyisipkan gambar input ke PDF:', e);
      }
    }

    const infoX = mX + 60;
    let iy = infoTop + 2;
    doc.setFont('helvetica', 'bold'); doc.setFontSize(10); doc.setTextColor(90);
    doc.text('KELAS TERDETEKSI', infoX, iy); iy += 6;
    doc.setFontSize(14); doc.setTextColor(0);
    doc.text(d.rejected ? 'Tidak dikenali' : (d.predicted_class || '-'), infoX, iy); iy += 6;

    if (!d.rejected && d.predicted_latin) {
      doc.setFont('helvetica', 'italic'); doc.setFontSize(10); doc.setTextColor(90);
      doc.text(d.predicted_latin, infoX, iy); iy += 6;
    }
    doc.setFont('helvetica', 'normal'); doc.setFontSize(10); doc.setTextColor(0);
    doc.text('Confidence score: ' + (d.confidence != null ? d.confidence + '%' : '-'), infoX, iy); iy += 7;

    const noteWidth = pageW - infoX - mX;
    if (d.rejected) {
      doc.setTextColor(185, 28, 28);
      const lines = doc.splitTextToSize(
        d.rejection_message || 'Gambar tidak terdeteksi sebagai daun/umbi bawang merah. Hasil ini tidak dapat dijadikan acuan diagnosis.',
        noteWidth
      );
      doc.text(lines, infoX, iy); iy += lines.length * 4.5;
      doc.setTextColor(0);
    } else if (d.is_simulation) {
      doc.setTextColor(180, 120, 0);
      const lines = doc.splitTextToSize(
        'Mode simulasi — ' + (d.disclaimer || 'model belum dilatih dengan dataset asli, hasil ini bersifat ilustratif.'),
        noteWidth
      );
      doc.text(lines, infoX, iy); iy += lines.length * 4.5;
      doc.setTextColor(0);
    }

    y = Math.max(imgBottom, iy) + 8;
    doc.setDrawColor(225); doc.line(mX, y, pageW - mX, y);
    y += 8;

    // ── Probabilitas tiap kelas ──
    if (Array.isArray(d.probabilities) && d.probabilities.length) {
      ensureSpace(14);
      doc.setFont('helvetica', 'bold'); doc.setFontSize(12); doc.setTextColor(0);
      doc.text('Probabilitas Tiap Kelas', mX, y); y += 7;
      doc.setFont('helvetica', 'normal'); doc.setFontSize(9.5);

      d.probabilities.slice().sort((a, b) => b.pct - a.pct).forEach(p => {
        ensureSpace(9);
        doc.setTextColor(0);
        doc.text(String(p.name), mX, y);
        doc.text(`${p.pct}%`, pageW - mX, y, { align: 'right' });
        y += 3.5;
        const barW = pageW - mX * 2;
        doc.setFillColor(230, 230, 230);
        doc.rect(mX, y, barW, 2.2, 'F');
        const pct = Math.max(0, Math.min(100, Number(p.pct) || 0));
        const [r, g, b] = _hexToRgb(p.color);
        doc.setFillColor(r, g, b);
        doc.rect(mX, y, (barW * pct) / 100, 2.2, 'F');
        y += 6.5;
      });
      y += 3;
    }

    // ── Rekomendasi penanganan ──
    ensureSpace(14);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(12); doc.setTextColor(0);
    doc.text('Rekomendasi Penanganan', mX, y); y += 7;
    doc.setFont('helvetica', 'normal'); doc.setFontSize(9.5);

    const recoLines = d.rejected
      ? ['Unggah ulang foto yang menampilkan daun atau umbi bawang dengan jelas, pencahayaan cukup, dan tidak buram.']
      : (Array.isArray(d.rekomendasi) ? d.rekomendasi : []);

    if (recoLines.length === 0) {
      doc.text('Tidak ada rekomendasi khusus untuk hasil ini.', mX, y);
      y += 6;
    } else {
      recoLines.forEach(text => {
        const lines = doc.splitTextToSize('• ' + text, pageW - mX * 2);
        ensureSpace(lines.length * 4.6 + 2);
        doc.text(lines, mX, y);
        y += lines.length * 4.6 + 2;
      });
    }
    y += 3;

    // ── Ringkasan proses CNN ──
    ensureSpace(45);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(12); doc.setTextColor(0);
    doc.text('Ringkasan Proses CNN', mX, y); y += 7;
    doc.setFontSize(9.5);

    const rows = [
      ['Input gambar', '224 × 224 × 3'],
      ['Konvolusi', '3 blok — 32 → 64 → 128 filter, kernel 3×3'],
      ['ReLU', d.relu_active_pct != null ? d.relu_active_pct + '% nilai aktif' : '-'],
      ['Pooling', '3 tahap — akhir 28 × 28 × 128'],
      ['Global Average Pooling', '128 elemen'],
      ['Fully connected', '512 neuron (Dense) + Dropout 0.5'],
      ['Output', (Array.isArray(d.probabilities) ? d.probabilities.length : '-') + ' kelas — Softmax'],
    ];
    rows.forEach(([k, v]) => {
      ensureSpace(6);
      doc.setFont('helvetica', 'bold'); doc.text(k, mX, y);
      doc.setFont('helvetica', 'normal'); doc.text(String(v), mX + 48, y);
      y += 5.5;
    });
    y += 3;

    // ── Footer disclaimer di tiap halaman ──
    const totalPages = doc.internal.getNumberOfPages();
    const footerNote = d.rejected
      ? 'Catatan: gambar tidak dikenali sebagai daun/umbi bawang merah — hasil ini TIDAK dapat dijadikan acuan diagnosis.'
      : (d.is_simulation
          ? 'Catatan: hasil ini dari MODE SIMULASI (model belum dilatih dengan dataset asli), bukan diagnosis final.'
          : 'Laporan ini dihasilkan otomatis oleh sistem klasifikasi penyakit bawang merah berbasis CNN.');
    for (let p = 1; p <= totalPages; p++) {
      doc.setPage(p);
      doc.setFontSize(8); doc.setTextColor(140);
      doc.text(doc.splitTextToSize(footerNote, pageW - mX * 2), mX, pageH - 10);
      doc.text(`${p} / ${totalPages}`, pageW - mX, pageH - 10, { align: 'right' });
    }

    const safeClass = (d.predicted_class || 'hasil-deteksi').toLowerCase().replace(/[^a-z0-9]+/g, '-');
    doc.save(`laporan-deteksi-bawang-${safeClass}-${Date.now()}.pdf`);
  } catch (err) {
    console.error('Gagal membuat laporan PDF:', err);
    alert('Gagal membuat laporan PDF. Silakan coba lagi.');
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = btnOriginal; }
  }
}
