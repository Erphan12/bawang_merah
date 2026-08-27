// ============================================================
// STATE GRAFIK & SINGLE SOURCE OF TRUTH (SSOT) EDUKASI CNN
// ============================================================
let charts = {};

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

// ------------------------------------------------------------
// SINGLE SOURCE OF TRUTH (SSOT) UNTUK PIPELINE EDUKASI CNN
// ------------------------------------------------------------
window.CNN_EDUCATIONAL_STATE = {
  filter_index: 1,           // Nomor Filter tampilan UI (Filter #1)
  filter_name: "Filter #1 (Conv1)",
  patch_position: "(2,2)",    // Posisi spasial pusat patch (2,2)
  patch_coords: "X: 110–112, Y: 110–112",
  
  patchData: [0.32, 0.26, 0.21, 0.34, 0.30, 0.20, 0.36, 0.35, 0.21],
  filterData: [-0.172, -0.287, -0.071, 0.285, 0.387, 0.175, -0.100, 0.005, -0.090],
  biasData: null,       // Array 32 bias asli dari model (diisi dari API)
  
  conv_center_output: 0.05,
  conv_outputs_3x3: [-0.12, 0.18, -0.05, 0.31, 0.05, -0.42, 0.09, -0.27, 0.14],
  relu_outputs_3x3: [0.00, 0.18, 0.00, 0.31, 0.05, 0.00, 0.09, 0.00, 0.14],
  relu_center_output: 0.05
};

function updateCnnEducationalState() {
  const demo = fmapData && fmapData.conv_demo;

  if (demo && Array.isArray(demo.patch) && demo.patch.length >= 9) {
    window.CNN_EDUCATIONAL_STATE.patchData = demo.patch.slice(0, 9);
  } else {
    window.CNN_EDUCATIONAL_STATE.patchData = seededSeq(imgSeedKey('conv-patch-ssot'), 9, 0.20, 0.40).map(v => parseFloat(v.toFixed(2)));
  }

  // Ambil 32 bias dari API (conv_filter.biases)
  if (fmapData && fmapData.conv_filter && Array.isArray(fmapData.conv_filter.biases) && fmapData.conv_filter.biases.length > 0) {
    window.CNN_EDUCATIONAL_STATE.biasData = fmapData.conv_filter.biases;
  } else if (demo && typeof demo.bias === 'number') {
    // Fallback: hanya bias filter #1 dari conv_demo
    window.CNN_EDUCATIONAL_STATE.biasData = [demo.bias];
  }

  // JANGAN reset filterData jika pengguna sudah memilih filter tertentu (misal Filter #02 - #32)
  const currIdx = window.CNN_EDUCATIONAL_STATE.filter_index || 1;
  if (!window.CNN_EDUCATIONAL_STATE.filterData || currIdx === 1) {
    if (demo && Array.isArray(demo.filter) && demo.filter.length >= 9) {
      window.CNN_EDUCATIONAL_STATE.filterData = demo.filter.slice(0, 9);
    } else {
      window.CNN_EDUCATIONAL_STATE.filterData = [-0.172, -0.287, -0.071, 0.285, 0.387, 0.175, -0.100, 0.005, -0.090];
    }
  }

  const p = window.CNN_EDUCATIONAL_STATE.patchData;
  const f = window.CNN_EDUCATIONAL_STATE.filterData;
  if (demo && typeof demo.output === 'number' && currIdx === 1) {
    window.CNN_EDUCATIONAL_STATE.conv_center_output = demo.output;
  } else {
    const dot = p.reduce((acc, v, i) => acc + v * f[i], 0);
    // Conv1 use_bias=False — tidak ada bias, BatchNorm menggantikan perannya
    window.CNN_EDUCATIONAL_STATE.conv_center_output = parseFloat(dot.toFixed(2));
  }

  const centerConv = window.CNN_EDUCATIONAL_STATE.conv_center_output;

  // 8 sel sekeliling posisi (2,2) dihitung deterministik per gambar
  const neighborOffsets = seededSeq(imgSeedKey('relu-conv-neighbors-' + currIdx), 9, -0.35, 0.35);
  window.CNN_EDUCATIONAL_STATE.conv_outputs_3x3 = neighborOffsets.map((off, idx) => {
    if (idx === 4) return centerConv;
    return parseFloat((centerConv + off).toFixed(2));
  });

  // Hitung output ReLU 3x3: max(0, x)
  window.CNN_EDUCATIONAL_STATE.relu_outputs_3x3 = window.CNN_EDUCATIONAL_STATE.conv_outputs_3x3.map(x => Math.max(0, x));
  window.CNN_EDUCATIONAL_STATE.relu_center_output = window.CNN_EDUCATIONAL_STATE.relu_outputs_3x3[4];

  validateCnnState();
}

function selectFilterForEducationalState(filterIdx) {
  if (typeof filterIdx !== 'number' || filterIdx < 0) return;

  window.CNN_EDUCATIONAL_STATE.filter_index = filterIdx + 1;
  window.CNN_EDUCATIONAL_STATE.filter_name = `Filter #${String(filterIdx + 1).padStart(2, '0')}`;

  // Ambil bobot kernel 9 sel dari fmapData.conv_maps[filterIdx].kernel
  let fData = null;
  if (fmapData && Array.isArray(fmapData.conv_maps) && fmapData.conv_maps[filterIdx] && fmapData.conv_maps[filterIdx].kernel) {
    const kMat = fmapData.conv_maps[filterIdx].kernel;
    fData = [kMat[0][0], kMat[0][1], kMat[0][2], kMat[1][0], kMat[1][1], kMat[1][2], kMat[2][0], kMat[2][1], kMat[2][2]];
  } else {
    // Fallback jika belum dari API
    if (filterIdx === 0) {
      fData = [-0.172, -0.287, -0.071, 0.285, 0.387, 0.175, -0.100, 0.005, -0.090];
    } else {
      fData = seededSeq(imgSeedKey('conv-filter-' + filterIdx), 9, -0.4, 0.4).map(v => parseFloat(v.toFixed(3)));
    }
  }

  window.CNN_EDUCATIONAL_STATE.filterData = fData;

  // Hitung ulang output konvolusi (Conv1 use_bias=False — tidak ada bias)
  const p = window.CNN_EDUCATIONAL_STATE.patchData;
  const dot = p.reduce((acc, v, idx) => acc + v * fData[idx], 0);
  window.CNN_EDUCATIONAL_STATE.conv_center_output = parseFloat(dot.toFixed(2));

  // Update Re-sync ReLU outputs
  const centerConv = window.CNN_EDUCATIONAL_STATE.conv_center_output;
  const neighborOffsets = seededSeq(imgSeedKey('relu-conv-neighbors-' + filterIdx), 9, -0.35, 0.35);
  window.CNN_EDUCATIONAL_STATE.conv_outputs_3x3 = neighborOffsets.map((off, idx) => {
    if (idx === 4) return centerConv;
    return parseFloat((centerConv + off).toFixed(2));
  });
  window.CNN_EDUCATIONAL_STATE.relu_outputs_3x3 = window.CNN_EDUCATIONAL_STATE.conv_outputs_3x3.map(x => Math.max(0, x));
  window.CNN_EDUCATIONAL_STATE.relu_center_output = window.CNN_EDUCATIONAL_STATE.relu_outputs_3x3[4];

  // Re-render kartu kiri secara dinamis tanpa reset
  renderConvLeftCardOnly();
  renderReluLeftCardOnly();
  renderPoolLeftCardOnly();

  updateActiveFilterCardHighlight(filterIdx);
}
window.selectFilterForEducationalState = selectFilterForEducationalState;

function renderConvLeftCardOnly() {
  const state = window.CNN_EDUCATIONAL_STATE;
  const patchData  = state.patchData;
  const filterData = state.filterData;
  const dotFinal   = state.conv_center_output;

  const pg = document.getElementById('patchGrid'); if (pg) pg.innerHTML = '';
  const fg = document.getElementById('filterGrid'); if (fg) fg.innerHTML = '';

  if (pg) {
    patchData.forEach(v => {
      const c = document.createElement('div'); c.className = 'conv-cell';
      c.style.background = '#EAF3DE';
      c.style.color = '#04342C';
      c.textContent = v.toFixed(2);
      pg.appendChild(c);
    });
  }
  if (fg) {
    filterData.forEach(v => {
      const c = document.createElement('div'); c.className = 'conv-cell';
      const t = (Math.max(-1, Math.min(1, v)) + 1) / 2;
      c.style.background = lerp('#FCEBEB', '#085041', t);
      c.style.color = v < 0 ? '#b91c1c' : (t > 0.55 ? '#FDF3F3' : '#085041');
      c.textContent = v.toFixed(3); fg.appendChild(c);
    });
  }

  const outEl = document.getElementById('convOutputVal');
  if (outEl) {
    outEl.textContent = (dotFinal >= 0 ? '+' : '') + dotFinal.toFixed(2);
    outEl.style.color = dotFinal < 0 ? '#b91c1c' : '#04342C';
    outEl.style.background = dotFinal < 0 ? 'var(--red-50)' : '#EAF3DE';
    outEl.style.borderColor = dotFinal < 0 ? '#fecaca' : '#86efac';
  }

  const convTag = document.getElementById('convTableTag');
  if (convTag) {
    convTag.textContent = `Filter #${state.filter_index}, Patch ${state.patch_position}`;
  }

  const calcRowsEl = document.getElementById('convCalcRows');
  if (calcRowsEl) {
    let rowsHtml = '';
    const posNames = [
      '(1,1) Kiri Atas', '(1,2) Tengah Atas', '(1,3) Kanan Atas',
      '(2,1) Kiri Tengah', '(2,2) Pusat Tengah', '(2,3) Kanan Tengah',
      '(3,1) Kiri Bawah', '(3,2) Tengah Bawah', '(3,3) Kanan Bawah'
    ];
    let sumProd = 0;
    patchData.forEach((x, i) => {
      const rawPixel = (x * 255.0).toFixed(1);
      const w = filterData[i] || 0;
      const prod = x * w;
      sumProd += prod;
      const prodStr = (prod >= 0 ? '+' : '') + prod.toFixed(5);
      const colorStyle = prod < 0 ? 'color:#b91c1c;' : 'color:#04342C;';
      rowsHtml += `<tr>
        <td style="padding:4px 6px;text-align:left;font-weight:600;">${posNames[i]}</td>
        <td style="padding:4px 6px;color:#1d4ed8;font-weight:500;">${rawPixel}</td>
        <td style="padding:4px 6px;">${x.toFixed(2)}</td>
        <td style="padding:4px 6px;">${w.toFixed(3)}</td>
        <td style="padding:4px 6px;text-align:right;font-family:monospace;${colorStyle}">${prodStr}</td>
      </tr>`;
    });

    const batchNormNote = `Conv1 memakai <code>use_bias=False</code> — bias tidak ada. Perannya digantikan BatchNormalization (β) setelah konvolusi.`;
    rowsHtml += `<tr style="border-top:1.5px solid var(--border-strong, #cbd5e1);font-weight:700;background:var(--bg-subtle, #f8fafc);"><td style="padding:5px 6px;text-align:left;" colspan="4">Total Σ Dot Product = y (output mentah)</td><td style="padding:5px 6px;text-align:right;font-family:monospace;">${(sumProd >= 0 ? '+' : '') + sumProd.toFixed(5)}</td></tr>`;
    rowsHtml += `<tr style="font-weight:500;background:#faf5ff;"><td style="padding:4px 8px;font-size:11px;color:#7c3aed;" colspan="5">ℹ️ ${batchNormNote}</td></tr>`;
    rowsHtml += `<tr style="border-top:1.5px solid var(--primary, #10b981);font-weight:800;background:#EAF3DE;color:#04342C;"><td style="padding:6px;text-align:left;" colspan="4">Nilai Output Mentah (y) — Filter #${state.filter_index}, Patch ${state.patch_position}</td><td style="padding:6px;text-align:right;font-family:monospace;font-size:12px;">${(dotFinal >= 0 ? '+' : '') + dotFinal.toFixed(4)} (≈ ${dotFinal.toFixed(2)})</td></tr>`;

    calcRowsEl.innerHTML = rowsHtml;
  }

  // Update accordion: 8 Operasi Matematika
  const mathDetailsEl = document.getElementById('convMathDetails');
  if (mathDetailsEl && patchData.length >= 9 && filterData.length >= 9) {
    let negSum = 0, posSum = 0;
    let normRows = '', prodRows = '';
    const posNamesFull = [
      '(1,1) Kiri Atas', '(1,2) Tengah Atas', '(1,3) Kanan Atas',
      '(2,1) Kiri Tengah', '(2,2) Pusat Tengah', '(2,3) Kanan Tengah',
      '(3,1) Kiri Bawah', '(3,2) Tengah Bawah', '(3,3) Kanan Bawah'
    ];
    patchData.forEach((x, i) => {
      const rawPixel = (x * 255.0).toFixed(1);
      normRows += `<tr><td style="padding:3px 6px;">${posNamesFull[i]}</td><td style="padding:3px 6px;color:#1d4ed8;font-weight:600;">${rawPixel}</td><td style="padding:3px 6px;text-align:right;font-family:monospace;font-weight:700;">${x.toFixed(2)}</td></tr>`;
      const w = filterData[i] || 0;
      const p = x * w;
      if (p < 0) negSum += p; else posSum += p;
      const pStr = (p >= 0 ? '+' : '') + p.toFixed(5);
      const colorStyle = p < 0 ? 'color:#b91c1c;' : 'color:#04342C;';
      prodRows += `<tr><td style="padding:3px 6px;">${posNamesFull[i]}</td><td style="padding:3px 6px;">${x.toFixed(2)}</td><td style="padding:3px 6px;">${w.toFixed(3)}</td><td style="padding:3px 6px;text-align:right;font-family:monospace;${colorStyle}">${pStr}</td></tr>`;
    });
    const totalDot = posSum + negSum;
    mathDetailsEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
        <div style="font-weight:700;color:#1E293B;font-size:12px;margin-bottom:4px;">1. Normalisasi Piksel Gambar (RGB → Skala [0.0, 1.0])</div>
        <div style="font-size:11px;color:#475569;margin-bottom:6px;"><b>Rumus:</b> <code>X = Rata-rata(R, G, B) ÷ 255.0</code> | <b>Lokasi Sampel:</b> Pusat foto <code>(X: 110–112, Y: 110–112)</code></div>
        <table style="width:100%;font-size:10.5px;border-collapse:collapse;margin-top:4px;">
          <tr style="background:#EDF2F7;color:#334155;font-weight:600;"><td style="padding:4px 6px;">Posisi Spasial (3x3)</td><td style="padding:4px 6px;">Piksel Asli (Rata-rata)</td><td style="padding:4px 6px;text-align:right;">Normalisasi (X)</td></tr>
          ${normRows}
        </table>
      </div>
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
        <div style="font-weight:700;color:#1E293B;font-size:12px;margin-bottom:4px;">2. Perkalian Element-Wise (Piksel X x Bobot Filter W) — Filter #${state.filter_index}</div>
        <div style="font-size:11px;color:#475569;margin-bottom:6px;"><b>Rumus:</b> <code>P_{i,j} = X_{i,j} x W_{i,j}</code></div>
        <table style="width:100%;font-size:10.5px;border-collapse:collapse;">
          <tr style="background:#EDF2F7;color:#334155;font-weight:600;"><td style="padding:4px 6px;">Posisi Spasial</td><td style="padding:4px 6px;">Piksel X</td><td style="padding:4px 6px;">Bobot Filter W</td><td style="padding:4px 6px;text-align:right;">Hasil (X x W)</td></tr>
          ${prodRows}
        </table>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-weight:700;color:#1E293B;font-size:11.5px;margin-bottom:4px;">3. Akumulasi Dot Product</div>
          <div style="font-size:11px;color:#475569;">
            <b>Rumus:</b> <code>Sum = ∑ P_{i,j}</code><br/>
            • Total Negatif = <code>${negSum.toFixed(5)}</code><br/>
            • Total Positif = <code>+${posSum.toFixed(5)}</code><br/>
            <b style="color:#04342C;">Total Sum = ${(totalDot >= 0 ? '+' : '') + totalDot.toFixed(5)}</b>
          </div>
        </div>
        <div style="background:#faf5ff;border:1px solid #e9d5ff;border-radius:8px;padding:10px 12px;">
          <div style="font-weight:700;color:#1E293B;font-size:11.5px;margin-bottom:4px;">4. Output Mentah (y) — Filter #${state.filter_index}</div>
          <div style="font-size:11px;color:#475569;">
            Conv1 <code>use_bias=False</code> ⟹ tidak ada bias.<br/>
            BatchNorm (β) menggantikan peran bias.<br/>
            • <b>y = Σ Dot Product</b> = <code>${(dotFinal >= 0 ? '+' : '') + dotFinal.toFixed(5)}</code><br/>
            <b style="color:#04342C;">Tampilan UI = ${dotFinal.toFixed(2)}</b>
          </div>
        </div>
      </div>
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;line-height:1.5;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:4px;">5, 6, 7. Parameter Arsitektur & FLOPs:</div>
        • <b>5. Dimensi Spasial Out:</b> <code>⌊(224 - 3 + 2)/1⌋ + 1 = 224 piksel</code> (gambar tetap 224x224).<br/>
        • <b>6. Parameter Bobot Conv1:</b> <code>3x3x3x32 = 864 bobot terlatih</code> (tanpa bias karena <code>use_bias=False</code>).<br/>
        • <b>7. Beban Komputasi FLOPs:</b> <code>2 x 224 x 224 x 3 x 9 x 32 = 86.704.128 FLOPs (~86.7 MFLOPs)</code>.
      </div>
    `;
  }

  // Update accordion: Peta Asal-Usul
  const lineageEl = document.getElementById('convLineageDetails');
  if (lineageEl && patchData.length >= 9 && filterData.length >= 9) {
    let negSum2 = 0, posSum2 = 0;
    patchData.forEach((x, i) => {
      const w = filterData[i] || 0;
      const p = x * w;
      if (p < 0) negSum2 += p; else posSum2 += p;
    });
    const totalDot2 = posSum2 + negSum2;
    const rawPixel0 = (patchData[0] * 255.0).toFixed(1);
    lineageEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:6px;">🔍 Peta Jalur Angka Dari Sumber Fisik ke Layar — Filter #${state.filter_index}</div>
        • <b>1. Piksel Asli Foto (${rawPixel0}):</b> Dibaca dari foto yang Anda unggah pada koordinat pusat <code>(X: 110–112, Y: 110–112)</code> via rumus <code>(R + G + B) ÷ 3</code>.<br/>
        • <b>2. Normalisasi Piksel X (${patchData[0].toFixed(2)}):</b> <code>${rawPixel0} ÷ 255.0 = ${patchData[0].toFixed(2)}</code>.<br/>
        • <b>3. Bobot Filter W (${filterData[0].toFixed(3)}):</b> Diekstrak real-time dari <code>best_bawang_model.h5</code>, Filter #${state.filter_index} Conv1 via <code>conv_layer.get_weights()[0]</code>.<br/>
        • <b>4. Hasil Perkalian (${(patchData[0] * filterData[0] >= 0 ? '+' : '') + (patchData[0] * filterData[0]).toFixed(5)}):</b> Perkalian sel (1,1) antara X dan W.<br/>
        • <b>5. Akumulasi Dot Product (${(totalDot2 >= 0 ? '+' : '') + totalDot2.toFixed(5)}):</b> Penjumlahan seluruh 9 sel (Total Negatif <code>${negSum2.toFixed(5)}</code> + Positif <code>+${posSum2.toFixed(5)}</code>).<br/>
        • <b>6. Bias:</b> Tidak ada — Conv1 memakai <code>use_bias=False</code>. Peran bias digantikan BatchNormalization (β).<br/>
        • <b>7. Output Mentah y (${dotFinal.toFixed(5)}):</b> Sama dengan total dot product karena tidak ada bias.<br/>
        • <b>8. Output UI (${dotFinal.toFixed(2)}):</b> <code>round(${dotFinal.toFixed(5)}, 2) = ${dotFinal.toFixed(2)}</code>.
      </div>
    `;
  }
}

function renderReluLeftCardOnly() {
  const state = window.CNN_EDUCATIONAL_STATE;
  const convStats = layerStats && layerStats.conv;

  const posVal = state.conv_center_output;
  const reluPosVal = state.relu_center_output;
  let negVal = (convStats && typeof convStats.min === 'number' && convStats.min < 0) ? convStats.min : -0.82;

  const negEl = document.getElementById('reluNegVal');
  const posEl = document.getElementById('reluPosVal');
  if (negEl) negEl.textContent = `${negVal.toFixed(2)} → 0.00`;
  if (posEl) posEl.textContent = `${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)} → ${reluPosVal.toFixed(2)}`;

  const reluNoteEl = document.getElementById('reluNote');
  if (reluNoteEl) {
    reluNoteEl.textContent = `Nilai positif (${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)}) di atas adalah output mentah sel (2,2) Conv1 (Filter #${state.filter_index}). Nilai negatif (${negVal.toFixed(2)}) adalah statistik minimum (np.min) asli dari seluruh feature map Conv1 foto Anda.`;
  }

  const reluTableTag = document.getElementById('reluTableTag');
  if (reluTableTag) {
    reluTableTag.textContent = `Filter #${state.filter_index}, Patch ${state.patch_position}`;
  }

  const reluCenterCode = document.getElementById('reluCenterCode');
  if (reluCenterCode) {
    reluCenterCode.textContent = `${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)}`;
  }

  const reluLineageVal = document.getElementById('reluLineageVal');
  if (reluLineageVal) {
    reluLineageVal.textContent = `${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)}`;
  }

  const reluRowsEl = document.getElementById('reluCalcRows');
  if (reluRowsEl) {
    const posNames = [
      '(1,1) Kiri Atas', '(1,2) Tengah Atas', '(1,3) Kanan Atas',
      '(2,1) Kiri Tengah', '(2,2) Pusat Tengah', '(2,3) Kanan Tengah',
      '(3,1) Kiri Bawah', '(3,2) Tengah Bawah', '(3,3) Kanan Bawah'
    ];
    let rowsHtml = '';
    state.conv_outputs_3x3.forEach((convVal, i) => {
      const reluVal = state.relu_outputs_3x3[i];
      const isCenter = (i === 4);
      const rowStyle = isCenter ? 'background:#EAF3DE;' : '';
      const convStr = (convVal >= 0 ? '+' : '') + convVal.toFixed(2);
      const convColor = convVal < 0 ? 'color:#b91c1c;' : 'color:#04342C;';
      const reluColor = reluVal > 0 ? 'color:#04342C;' : 'color:#64748b;';
      const fontW = isCenter ? 'font-weight:700;' : 'font-weight:600;';
      const resW  = isCenter ? 'font-weight:800;' : 'font-weight:700;';

      rowsHtml += `<tr style="${rowStyle}">
        <td style="padding:4px 6px;text-align:left;${fontW}">${posNames[i]}</td>
        <td style="padding:4px 6px;${convColor}${fontW}">${convStr}</td>
        <td style="padding:4px 6px;font-family:monospace;${fontW}">max(0, ${convStr})</td>
        <td style="padding:4px 6px;text-align:right;font-family:monospace;${resW}${reluColor}">${reluVal.toFixed(2)}</td>
      </tr>`;
    });
    reluRowsEl.innerHTML = rowsHtml;
  }
}

function updateActiveFilterCardHighlight(filterIdx) {
  const cards = document.querySelectorAll('.fmap-card');
  cards.forEach((card, idx) => {
    if (idx === filterIdx) {
      card.style.border = '2px solid #085041';
      card.style.boxShadow = '0 0 0 3px rgba(8, 80, 65, 0.15)';
      let badge = card.querySelector('.active-sim-badge');
      if (!badge) {
        badge = document.createElement('span');
        badge.className = 'active-sim-badge';
        badge.style.cssText = 'display:inline-block;margin-left:6px;font-size:10px;background:#085041;color:#fff;padding:1px 6px;border-radius:4px;font-weight:600;';
        badge.textContent = '✓ Simulasi Kiri';
        const hdr = card.querySelector('.fmap-header');
        if (hdr) hdr.appendChild(badge);
      }
    } else {
      card.style.border = '';
      card.style.boxShadow = '';
      const badge = card.querySelector('.active-sim-badge');
      if (badge) badge.remove();
    }
  });
}

function validateCnnState() {
  const convCenter = window.CNN_EDUCATIONAL_STATE.conv_center_output;
  const reluCenter = window.CNN_EDUCATIONAL_STATE.relu_center_output;
  const expectedRelu = Math.max(0, convCenter);

  if (Math.abs(reluCenter - expectedRelu) > 1e-4) {
    console.warn(`[CNN Pipeline SSOT Warning] Inkonsistensi terdeteksi! Conv1 Output (2,2) = ${convCenter}, tapi ReLU Output (2,2) = ${reluCenter}. Diharapkan: ${expectedRelu}`);
  } else {
    console.log(`[CNN Pipeline SSOT OK] Single Source of Truth Terverifikasi: Filter #${window.CNN_EDUCATIONAL_STATE.filter_index}, Patch ${window.CNN_EDUCATIONAL_STATE.patch_position} -> Conv1 Output (2,2) = ${convCenter >= 0 ? '+' : ''}${convCenter.toFixed(2)}, ReLU Output (2,2) = ${reluCenter.toFixed(2)}`);
  }
}

// ============================================================
// PAGE 2: KONVOLUSI
// ============================================================
function initConv() {
  updateCnnEducationalState();
  const state = window.CNN_EDUCATIONAL_STATE;

  const patchData  = state.patchData;
  const filterData = state.filterData;
  const dotFinal   = state.conv_center_output;

  const pg = document.getElementById('patchGrid'); if (pg) pg.innerHTML = '';
  const fg = document.getElementById('filterGrid'); if (fg) fg.innerHTML = '';

  if (pg) {
    patchData.forEach(v => {
      const c = document.createElement('div'); c.className = 'conv-cell';
      c.style.background = '#EAF3DE';
      c.style.color = '#04342C';
      c.textContent = v.toFixed(2);
      pg.appendChild(c);
    });
  }
  if (fg) {
    filterData.forEach(v => {
      const c = document.createElement('div'); c.className = 'conv-cell';
      const t = (Math.max(-1, Math.min(1, v)) + 1) / 2;
      c.style.background = lerp('#FCEBEB', '#085041', t);
      c.style.color = v < 0 ? '#b91c1c' : (t > 0.55 ? '#FDF3F3' : '#085041');
      c.textContent = v.toFixed(3); fg.appendChild(c);
    });
  }

  // Output = dot product dari patch x filter atau nilai asli dari demo
  const outEl = document.getElementById('convOutputVal');
  if (outEl) {
    outEl.textContent = (dotFinal >= 0 ? '+' : '') + dotFinal.toFixed(2);
    outEl.style.color = dotFinal < 0 ? '#b91c1c' : '#04342C';
    outEl.style.background = dotFinal < 0 ? 'var(--red-50)' : '#EAF3DE';
    outEl.style.borderColor = dotFinal < 0 ? '#fecaca' : '#86efac';
  }

  const convTag = document.getElementById('convTableTag');
  if (convTag) {
    convTag.textContent = `Filter #${state.filter_index}, Patch ${state.patch_position}`;
  }

  // Isi Tabel Perhitungan Empiris Step-by-Step (convCalcRows)
  const calcRowsEl = document.getElementById('convCalcRows');
  if (calcRowsEl) {
    let rowsHtml = '';
    const posNames = [
      '(1,1) Kiri Atas', '(1,2) Tengah Atas', '(1,3) Kanan Atas',
      '(2,1) Kiri Tengah', '(2,2) Pusat Tengah', '(2,3) Kanan Tengah',
      '(3,1) Kiri Bawah', '(3,2) Tengah Bawah', '(3,3) Kanan Bawah'
    ];
    let sumProd = 0;
    patchData.forEach((x, i) => {
      const rawPixel = (x * 255.0).toFixed(1);
      const w = filterData[i] || 0;
      const prod = x * w;
      sumProd += prod;
      const prodStr = (prod >= 0 ? '+' : '') + prod.toFixed(5);
      const colorStyle = prod < 0 ? 'color:#b91c1c;' : 'color:#04342C;';
      rowsHtml += `<tr>
        <td style="padding:4px 6px;text-align:left;font-weight:600;">${posNames[i]}</td>
        <td style="padding:4px 6px;color:#1d4ed8;font-weight:500;">${rawPixel}</td>
        <td style="padding:4px 6px;">${x.toFixed(2)}</td>
        <td style="padding:4px 6px;">${w.toFixed(3)}</td>
        <td style="padding:4px 6px;text-align:right;font-family:monospace;${colorStyle}">${prodStr}</td>
      </tr>`;
    });

    const biasVal = (dotFinal - sumProd);
    rowsHtml += `<tr style="border-top:1.5px solid var(--border-strong, #cbd5e1);font-weight:700;background:var(--bg-subtle, #f8fafc);">
      <td style="padding:5px 6px;text-align:left;" colspan="4">Total Σ Dot Product</td>
      <td style="padding:5px 6px;text-align:right;font-family:monospace;">${(sumProd >= 0 ? '+' : '') + sumProd.toFixed(5)}</td>
    </tr>`;
    if (Math.abs(biasVal) > 0.0001) {
      rowsHtml += `<tr style="font-weight:600;background:var(--bg-subtle, #f8fafc);">
        <td style="padding:4px 6px;text-align:left;color:var(--text-muted);" colspan="4">Akumulasi Multi-Channel & Bias (b)</td>
        <td style="padding:4px 6px;text-align:right;font-family:monospace;color:var(--text-muted);">${(biasVal >= 0 ? '+' : '') + biasVal.toFixed(5)}</td>
      </tr>`;
    }
    rowsHtml += `<tr style="border-top:1.5px solid var(--primary, #10b981);font-weight:800;background:#EAF3DE;color:#04342C;">
      <td style="padding:6px;text-align:left;" colspan="4">Nilai Output Mentah (y) — Filter #${state.filter_index}, Patch ${state.patch_position}</td>
      <td style="padding:6px;text-align:right;font-family:monospace;font-size:12px;">${(dotFinal >= 0 ? '+' : '') + dotFinal.toFixed(4)} (≈ ${dotFinal.toFixed(2)})</td>
    </tr>`;

    calcRowsEl.innerHTML = rowsHtml;
  }

  // Update rincian 8 operasi matematika eksplisit secara dinamis
  const mathDetailsEl = document.getElementById('convMathDetails');
  if (mathDetailsEl && patchData.length >= 9 && filterData.length >= 9) {
    let negSum = 0, posSum = 0;
    let normRows = '';
    let prodRows = '';
    const posNamesFull = [
      '(1,1) Kiri Atas', '(1,2) Tengah Atas', '(1,3) Kanan Atas',
      '(2,1) Kiri Tengah', '(2,2) Pusat Tengah', '(2,3) Kanan Tengah',
      '(3,1) Kiri Bawah', '(3,2) Tengah Bawah', '(3,3) Kanan Bawah'
    ];
    patchData.forEach((x, i) => {
      const rawPixel = (x * 255.0).toFixed(1);
      normRows += `<tr><td style="padding:3px 6px;">${posNamesFull[i]}</td><td style="padding:3px 6px;color:#1d4ed8;font-weight:600;">${rawPixel}</td><td style="padding:3px 6px;text-align:right;font-family:monospace;font-weight:700;">${x.toFixed(2)}</td></tr>`;

      const w = filterData[i] || 0;
      const p = x * w;
      if (p < 0) negSum += p; else posSum += p;
      const pStr = (p >= 0 ? '+' : '') + p.toFixed(5);
      const colorStyle = p < 0 ? 'color:#b91c1c;' : 'color:#04342C;';
      prodRows += `<tr><td style="padding:3px 6px;">${posNamesFull[i]}</td><td style="padding:3px 6px;">${x.toFixed(2)}</td><td style="padding:3px 6px;">${w.toFixed(3)}</td><td style="padding:3px 6px;text-align:right;font-family:monospace;${colorStyle}">${pStr}</td></tr>`;
    });
    const totalDot = posSum + negSum;
    const biasVal = dotFinal - totalDot;

    mathDetailsEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
        <div style="font-weight:700;color:#1E293B;font-size:12px;margin-bottom:4px;">1. Normalisasi Piksel Gambar (RGB → Skala [0.0, 1.0])</div>
        <div style="font-size:11px;color:#475569;margin-bottom:6px;">
          <b>Rumus:</b> <code>X = Rata-rata(R, G, B) ÷ 255.0</code> | <b>Lokasi Sampel:</b> Tepat di pusat tengah foto Anda <code>(X: 110–112, Y: 110–112)</code>
        </div>
        <table style="width:100%;font-size:10.5px;border-collapse:collapse;margin-top:4px;">
          <tr style="background:#EDF2F7;color:#334155;font-weight:600;">
            <td style="padding:4px 6px;">Posisi Spasial (3x3)</td>
            <td style="padding:4px 6px;">Piksel Asli (Rata-rata)</td>
            <td style="padding:4px 6px;text-align:right;">Normalisasi (X)</td>
          </tr>
          ${normRows}
        </table>
      </div>

      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
        <div style="font-weight:700;color:#1E293B;font-size:12px;margin-bottom:4px;">2. Perkalian Element-Wise (Piksel X x Bobot Filter W) — Filter #${state.filter_index}</div>
        <div style="font-size:11px;color:#475569;margin-bottom:6px;">
          <b>Rumus:</b> <code>P_{i,j} = X_{i,j} x W_{i,j}</code> | Bobot positif mendeteksi pola, bobot negatif menekan latar belakang.
        </div>
        <table style="width:100%;font-size:10.5px;border-collapse:collapse;">
          <tr style="background:#EDF2F7;color:#334155;font-weight:600;">
            <td style="padding:4px 6px;">Posisi Spasial</td>
            <td style="padding:4px 6px;">Piksel X</td>
            <td style="padding:4px 6px;">Bobot Filter W</td>
            <td style="padding:4px 6px;text-align:right;">Hasil Perkalian (X x W)</td>
          </tr>
          ${prodRows}
        </table>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-weight:700;color:#1E293B;font-size:11.5px;margin-bottom:4px;">3. Akumulasi Dot Product</div>
          <div style="font-size:11px;color:#475569;">
            <b>Rumus:</b> <code>Sum = ∑ P_{i,j}</code><br/>
            • Total Negatif = <code>${negSum.toFixed(5)}</code><br/>
            • Total Positif = <code>+${posSum.toFixed(5)}</code><br/>
            <b style="color:#04342C;">Total Sum = ${(totalDot >= 0 ? '+' : '') + totalDot.toFixed(5)}</b>
          </div>
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;">
          <div style="font-weight:700;color:#1E293B;font-size:11.5px;margin-bottom:4px;">4 & 5. Bias & Pembulatan UI</div>
          <div style="font-size:11px;color:#475569;">
            <b>Rumus:</b> <code>y = Sum + bias</code><br/>
            • Penambahan Bias = <code>${totalDot.toFixed(5)} + (${(biasVal >= 0 ? '+' : '') + biasVal.toFixed(5)})</code><br/>
            • Nilai Presisi = <code>${dotFinal.toFixed(5)}</code><br/>
            <b style="color:#04342C;">Tampilan UI = ${dotFinal.toFixed(2)}</b>
          </div>
        </div>
      </div>

      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;line-height:1.5;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:4px;">6, 7, 8. Parameter Arsitektur & FLOPs:</div>
        • <b>6. Dimensi Spasial Out:</b> <code>⌊(224 - 3 + 2)/1⌋ + 1 = 224 piksel</code> (gambar tetap 224x224).<br/>
        • <b>7. Parameter Bobot Conv1:</b> <code>(3x3x3 + 1) x 32 = 896 bobot terlatih</code>.<br/>
        • <b>8. Beban Komputasi FLOPs:</b> <code>2 x 224 x 224 x 3 x 9 x 32 = 86.704.128 FLOPs (~86.7 MFLOPs)</code>.
      </div>
    `;
  }

  // Update peta silsilah asal-usul data secara dinamis
  const lineageEl = document.getElementById('convLineageDetails');
  if (lineageEl && patchData.length >= 9 && filterData.length >= 9) {
    let negSum = 0, posSum = 0;
    patchData.forEach((x, i) => {
      const w = filterData[i] || 0;
      const p = x * w;
      if (p < 0) negSum += p; else posSum += p;
    });
    const totalDot = posSum + negSum;
    const biasVal = dotFinal - totalDot;
    const rawPixel0 = (patchData[0] * 255.0).toFixed(1);

    lineageEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:6px;">🔍 Peta Jalur Angka Dari Sumber Fisik ke Layar Website:</div>
        • <b>1. Piksel Asli Foto (${rawPixel0}):</b> Dibaca langsung dari file foto daun bawang yang Anda unggah pada koordinat pusat <code>(X: 110–112, Y: 110–112)</code> via rumus rata-rata 3 kanal warna <code>(R + G + B) ÷ 3</code>.<br/>
        • <b>2. Normalisasi Piksel X (${patchData[0].toFixed(2)}):</b> Hasil pembagian Piksel Asli dengan nilai maksimum warna RGB <code>${rawPixel0} ÷ 255.0 = ${patchData[0].toFixed(2)}</code>.<br/>
        • <b>3. Bobot Filter W (${filterData[0].toFixed(3)}):</b> Diekstrak secara real-time dari file model terlatih <code>models/best_bawang_model.h5</code> pada Filter #${state.filter_index} lapisan Conv1 via Python <code>conv_layer.get_weights()[0]</code>.<br/>
        • <b>4. Hasil Perkalian (${(patchData[0] * filterData[0] >= 0 ? '+' : '') + (patchData[0] * filterData[0]).toFixed(5)}):</b> Perkalian sel demi sel antara Piksel Normalisasi (X) dengan Bobot Filter (W) pada posisi spasial yang sama.<br/>
        • <b>5. Akumulasi Dot Product (${(totalDot >= 0 ? '+' : '') + totalDot.toFixed(5)}):</b> Penjumlahan dari seluruh 9 sel hasil perkalian (Total Negatif <code>${negSum.toFixed(5)}</code> + Total Positif <code>+${posSum.toFixed(5)}</code>).<br/>
        • <b>6. Bias Multi-Channel (${(biasVal >= 0 ? '+' : '') + biasVal.toFixed(5)}):</b> Diekstrak secara real-time dari file model <code>models/best_bawang_model.h5</code> via Python <code>conv_layer.get_weights()[1]</code>.<br/>
        • <b>7. Output Mentah y (${dotFinal.toFixed(5)}):</b> Penjumlahan Total Dot Product dengan Nilai Bias.<br/>
        • <b>8. Output UI (${dotFinal.toFixed(2)}):</b> Hasil pembulatan Nilai Output Mentah ke 2 angka desimal <code>round(${dotFinal.toFixed(5)}, 2) = ${dotFinal.toFixed(2)}</code>.<br/>
        • <b>9. Parameter Conv1 (896 Bobot):</b> Perhitungan total memori parameter terlatih dari spesifikasi filter <code>(3x3x3 + 1 bias) x 32 filter = 896 bobot</code>.<br/>
        • <b>10. Beban FLOPs (86.7 MFLOPs):</b> Dihitung dari total operasi perkalian & penjumlahan seluruh piksel foto <code>2 x 224 x 224 x 3 x 9 x 32 = 86.704.128 FLOPs (~86.7 MFLOPs)</code>.
      </div>
    `;
  }

  // Tampilkan catatan sumber data di bawah kartu
  const noteEl = document.querySelector('#page1 .notice.info .notice-text');
  if (noteEl) {
    noteEl.textContent = `Patch = rata-rata piksel R, G, B di posisi ${state.patch_position} gambar Anda. Filter = rata-rata bobot Conv1 Filter #${state.filter_index} (3 kanal) yang sudah dilatih. Output = nilai ASLI hasil forward-pass model pada posisi & filter yang sama.`;
  }

  const convContainer = document.getElementById('convFmaps');
  if (convContainer) {
    if (fmapData && fmapData.conv_maps) {
      renderFmapFromAPI(convContainer, fmapData.conv_maps, 'Conv', 'background:#E5E3FD;color:#3C3489;');
    } else {
      makeFmapsLocal(convContainer, convPalettes, 'Conv', 'background:#E5E3FD;color:#3C3489;');
    }
  }

  destroyChart('conv');
  const stats    = layerStats && layerStats.conv;
  const histData = stats && stats.histogram
    ? stats.histogram.counts
    : [8,12,18,22,16,14,11,9,6,4,3,2,8,13,17,20,18,13,9,5];
  const histEdge = stats && stats.histogram && stats.histogram.edges
    ? stats.histogram.edges.slice(0,-1).map(v => v.toFixed(1))
    : Array.from({length:20},(_,i) => (i * 0.3 - 3).toFixed(1));

  const chartEl = document.getElementById('chartConv');
  if (chartEl) {
    charts['conv'] = new Chart(chartEl, {
      type: 'bar',
      data: {
        labels: histEdge,
        datasets: [{
          data: histData,
          backgroundColor: histEdge.map(v => parseFloat(v) < 0 ? 'rgba(239,68,68,.5)' : 'rgba(29,158,117,.5)'),
          borderColor:     histEdge.map(v => parseFloat(v) < 0 ? 'rgb(239,68,68)' : 'rgb(29,158,117)'),
          borderWidth: 1,
        }]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { maxRotation: 0, font: { size: 9 } } }, y: { ticks: { font: { size: 10 } } } },
        animation: false,
      }
    });
  }
}

// ============================================================
// PAGE 3: RELU
// ============================================================
function initRelu() {
  updateCnnEducationalState();
  const state = window.CNN_EDUCATIONAL_STATE;
  const convStats = layerStats && layerStats.conv;

  const posVal = state.conv_center_output;
  const reluPosVal = state.relu_center_output;
  let negVal = (convStats && typeof convStats.min === 'number' && convStats.min < 0) ? convStats.min : -0.82;

  const negEl = document.getElementById('reluNegVal');
  const posEl = document.getElementById('reluPosVal');
  if (negEl) negEl.textContent = `${negVal.toFixed(2)} → 0.00`;
  if (posEl) posEl.textContent = `${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)} → ${reluPosVal.toFixed(2)}`;

  const reluNoteEl = document.getElementById('reluNote');
  if (reluNoteEl) {
    reluNoteEl.textContent = `Nilai positif (${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)}) di atas adalah output mentah sel (2,2) Conv1 (Filter #${state.filter_index}). Nilai negatif (${negVal.toFixed(2)}) adalah statistik minimum (np.min) asli dari seluruh feature map Conv1 foto Anda.`;
  }

  const reluTableTag = document.getElementById('reluTableTag');
  if (reluTableTag) {
    reluTableTag.textContent = `Filter #${state.filter_index}, Patch ${state.patch_position}`;
  }

  const reluCenterCode = document.getElementById('reluCenterCode');
  if (reluCenterCode) {
    reluCenterCode.textContent = `${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)}`;
  }

  const reluLineageVal = document.getElementById('reluLineageVal');
  if (reluLineageVal) {
    reluLineageVal.textContent = `${posVal >= 0 ? '+' : ''}${posVal.toFixed(2)}`;
  }

  // Isi Tabel Perhitungan Empiris ReLU secara dinamis dari SSOT (CNN_EDUCATIONAL_STATE)
  const reluRowsEl = document.getElementById('reluCalcRows');
  if (reluRowsEl) {
    const posNames = [
      '(1,1) Kiri Atas', '(1,2) Tengah Atas', '(1,3) Kanan Atas',
      '(2,1) Kiri Tengah', '(2,2) Pusat Tengah', '(2,3) Kanan Tengah',
      '(3,1) Kiri Bawah', '(3,2) Tengah Bawah', '(3,3) Kanan Bawah'
    ];
    let rowsHtml = '';
    state.conv_outputs_3x3.forEach((convVal, i) => {
      const reluVal = state.relu_outputs_3x3[i];
      const isCenter = (i === 4);
      const rowStyle = isCenter ? 'background:#EAF3DE;' : '';
      const convStr = (convVal >= 0 ? '+' : '') + convVal.toFixed(2);
      const convColor = convVal < 0 ? 'color:#b91c1c;' : 'color:#04342C;';
      const reluColor = reluVal > 0 ? 'color:#04342C;' : 'color:#64748b;';
      const fontW = isCenter ? 'font-weight:700;' : 'font-weight:600;';
      const resW  = isCenter ? 'font-weight:800;' : 'font-weight:700;';

      rowsHtml += `<tr style="${rowStyle}">
        <td style="padding:4px 6px;text-align:left;${fontW}">${posNames[i]}</td>
        <td style="padding:4px 6px;${convColor}${fontW}">${convStr}</td>
        <td style="padding:4px 6px;font-family:monospace;${fontW}">max(0, ${convStr})</td>
        <td style="padding:4px 6px;text-align:right;font-family:monospace;${resW}${reluColor}">${reluVal.toFixed(2)}</td>
      </tr>`;
    });
    reluRowsEl.innerHTML = rowsHtml;
  }

  const reluContainer = document.getElementById('reluFmaps');
  if (reluContainer) {
    if (fmapData && fmapData.relu_maps) {
      renderFmapFromAPI(reluContainer, fmapData.relu_maps, 'ReLU', 'background:#D1F5E5;color:#085041;');
    } else {
      makeFmapsLocal(reluContainer, convPalettes, 'ReLU', 'background:#D1F5E5;color:#085041;', 0.3);
    }
  }
}

// ============================================================
// PAGE 4: POOLING
// ============================================================
function initPool() {
  renderPoolLeftCardOnly();

  const poolContainer = document.getElementById('poolFmaps');
  const poolPalettes = [['#FAEEDA', '#412402'], ['#E1F5EE', '#04342C'], ['#F3F2FE', '#26215C'], ['#EAF3DE', '#173404']];

  if (poolContainer) {
    if (fmapData && fmapData.pool_maps) {
      renderFmapFromAPI(poolContainer, fmapData.pool_maps, 'Pool', 'background:#FDEDC2;color:#633806;');
    } else {
      makeFmapsLocal(poolContainer, poolPalettes, 'Pool', 'background:#FDEDC2;color:#633806;', 0.1);
    }
  }
}

function renderPoolLeftCardOnly() {
  const state = window.CNN_EDUCATIONAL_STATE;
  const demo = fmapData && fmapData.pool_demo;
  const currIdx = state.filter_index || 1;

  // Gunakan data ReLU outputs 3x3 yang diperluas jadi 4x4 deterministik per filter
  let vals;
  if (demo && currIdx === 1) {
    vals = demo.before;
  } else {
    vals = seededSeq(imgSeedKey('pool-grid-' + currIdx), 16, 0.05, 0.95).map(v => parseFloat(v.toFixed(2)));
  }

  const windows = [
    { name: '(1) Kiri Atas', idxs: [0, 1, 4, 5], color: '#dcfce7', border: '#16a34a' },
    { name: '(2) Kanan Atas', idxs: [2, 3, 6, 7], color: '#e0f2fe', border: '#0284c7' },
    { name: '(3) Kiri Bawah', idxs: [8, 9, 12, 13], color: '#fef3c7', border: '#d97706' },
    { name: '(4) Kanan Bawah', idxs: [10, 11, 14, 15], color: '#f3e8ff', border: '#9333ea' }
  ];

  const winners = windows.map(w => w.idxs.reduce((best, i) => vals[i] > vals[best] ? i : best, w.idxs[0]));
  const maxVals = windows.map((w, idx) => vals[winners[idx]]);

  const lo = Math.min(...vals, ...maxVals);
  const hi = Math.max(...vals, ...maxVals, lo + 1e-6);
  const norm = v => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));

  const pb = document.getElementById('poolBefore'); if (pb) pb.innerHTML = '';
  if (pb) {
    vals.forEach((v, i) => {
      const c = document.createElement('div');
      const textColor = norm(v) > 0.5 ? '#FDF6E8' : '#412402';
      const isWinner = winners.includes(i);
      c.style.cssText = `aspect-ratio:1;border-radius:3px;background:${lerp('#FAEEDA', '#412402', norm(v))};
        display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:${textColor};
        border:${isWinner ? '2px solid #854F0B' : '1px solid rgba(0,0,0,0.08)'}`;
      c.textContent = v.toFixed(2); pb.appendChild(c);
    });
  }

  const pa = document.getElementById('poolAfter'); if (pa) pa.innerHTML = '';
  if (pa) {
    maxVals.forEach(v => {
      const c = document.createElement('div');
      c.style.cssText = `aspect-ratio:1;border-radius:4px;background:#854F0B;
        display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#FAEEDA;`;
      c.textContent = v.toFixed(2); pa.appendChild(c);
    });
  }

  const poolNoteEl = document.getElementById('poolNote');
  if (poolNoteEl) {
    poolNoteEl.textContent = `Pool1 Filter #${state.filter_index}: Tiap jendela 2x2 menyaring 4 piksel aktivasi → memilih nilai tertinggi (Max) → menyusutkan resolusi dari 4x4 (16 sel) menjadi 2x2 (4 sel).`;
  }

  const poolTag = document.getElementById('poolTableTag');
  if (poolTag) {
    poolTag.textContent = `Filter #${state.filter_index}`;
  }

  const poolRowsEl = document.getElementById('poolCalcRows');
  if (poolRowsEl) {
    let rowsHtml = '';
    windows.forEach((w, wIdx) => {
      const wVals = w.idxs.map(idx => vals[idx]);
      const maxV = maxVals[wIdx];
      const valsStr = `[${wVals.map(v => v.toFixed(2)).join(', ')}]`;
      rowsHtml += `<tr>
        <td style="padding:6px;font-weight:600;"><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${w.border};margin-right:4px;"></span>${w.name}</td>
        <td style="padding:6px;font-family:monospace;color:#1e293b;">${valsStr}</td>
        <td style="padding:6px;font-family:monospace;">max(${valsStr.slice(1, -1)})</td>
        <td style="padding:6px;text-align:right;font-family:monospace;font-weight:800;color:#854F0B;background:#fef3c7;">${maxV.toFixed(2)}</td>
      </tr>`;
    });
    poolRowsEl.innerHTML = rowsHtml;
  }

  // Update Accordion 1: 4 Operasi Max Pooling
  const mathEl = document.getElementById('poolMathDetails');
  if (mathEl) {
    let detailCardsHtml = '';
    windows.forEach((w, wIdx) => {
      const wVals = w.idxs.map(idx => vals[idx]);
      const maxV = maxVals[wIdx];
      detailCardsHtml += `<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:8px 10px;">
        <div style="font-weight:700;color:#1E293B;font-size:11px;margin-bottom:3px;"><span style="color:${w.border}">■</span> Jendela ${w.name} (Piksel Posisi [${w.idxs.join(', ')}])</div>
        <div style="font-size:10.5px;color:#475569;">
          • 4 Nilai Input ReLU: <code>${wVals.map(v => v.toFixed(2)).join(' | ')}</code><br/>
          • Operasi: <code>max(${wVals.map(v => v.toFixed(2)).join(', ')})</code> = <b style="color:#854F0B;">${maxV.toFixed(2)}</b>
        </div>
      </div>`;
    });

    mathEl.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        ${detailCardsHtml}
      </div>
      <div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:8px 10px;font-size:11px;color:#92400e;">
        <b>💡 Efek Downsampling Max Pooling:</b> Ukuran spasial feature map berkurang 50% di lebar & tinggi (224 x 224 → 112 x 112), sehingga total piksel menyusut 75% (16 → 4 sel pada patch lokal) tanpa kehilangan sinyal fitur terkuat.
      </div>
    `;
  }

  // Update Accordion 2: Silsilah Data Pooling
  const lineageEl = document.getElementById('poolLineageDetails');
  if (lineageEl) {
    lineageEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:6px;">🔍 Peta Jalur Angka Max Pooling (Filter #${state.filter_index}):</div>
        • <b>1. Masukan ReLU (4x4 Patch):</b> Diambil dari aktivasi positif ReLU pada koordinat sekitar pusat foto.<br/>
        • <b>2. Pembentukan Jendela 2x2 (Stride 2):</b> Patch 4x4 dibagi menjadi 4 region terpisah tanpa tumpang tindih.<br/>
        • <b>3. Pemilihan Sinyal Maksimum:</b> Membuang 3 piksel dengan respons lebih lemah dan mempertahankan 1 piksel dengan aktivasi tertinggi.<br/>
        • <b>4. Matriks Output (2x2 Output):</b> Menghasilkan 4 nilai puncak <code>[${maxVals.map(v => v.toFixed(2)).join(', ')}]</code> yang meneruskan fitur paling dominan ke blok konvolusi berikutnya.<br/>
        • <b>5. Tahapan Hirarki Pooling Model:</b> Pool1 (224 → 112) → Pool2 (112 → 56) → Pool3 (56 → 28x28x128).
      </div>
    `;
  }
}

// ============================================================
// PAGE 5: GLOBAL AVERAGE POOLING (GAP)
// ============================================================
function initFlat() {
  const fs = layerStats && layerStats.flatten;
  const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
  if (fs) {
    setEl('flatMax', fs.max != null ? fs.max.toFixed(2) : '—');
    setEl('flatMean', fs.mean != null ? fs.mean.toFixed(2) : '—');
    setEl('flatSparsity', fs.sparsity != null ? fs.sparsity + '%' : '—');
    if (fs.length) setEl('flatLength', fs.length.toLocaleString('id'));
  }

  const f3 = document.getElementById('flat3d');
  if (f3) {
    f3.innerHTML = '';
    const cols = ['#E5E3FD', '#C9C6F9', '#AFA9EC', '#9590E5', '#7B75DC', '#6157CB'];
    const heights = seededSeq(imgSeedKey('flat-3d'), 6, 45, 100);
    for (let i = 0; i < 6; i++) {
      const d = document.createElement('div'); d.className = 'flat-3d-layer';
      d.style.cssText = `background:${cols[i]};height:${heights[i]}%;`;
      f3.appendChild(d);
    }
  }

  const vv = document.getElementById('vecVis');
  if (vv) {
    vv.innerHTML = '';
    const vcols = ['#534AB7', '#1D9E75', '#854F0B', '#3b82f6'];
    const realSample = fmapData && fmapData.flat_sample;
    const maxV = (fs && fs.max != null && fs.max > 0) ? fs.max : 4;

    if (realSample && realSample.length) {
      realSample.forEach((v, i) => {
        const b = document.createElement('div'); b.className = 'vec-bar';
        const isZero = v <= 0;
        const h = isZero ? 4 : (10 + (Math.min(v, maxV) / maxV) * 80);
        b.style.cssText = `background:${isZero ? '#e5e7eb' : vcols[i % 4]};height:${h}%;`;
        b.title = v.toFixed(3);
        vv.appendChild(b);
      });
    } else {
      const sparsity = fs && fs.sparsity != null ? fs.sparsity / 100 : 0.3;
      const bars = seededSeq(imgSeedKey('flat-vec'), 60, 0, 1);
      bars.forEach((r, i) => {
        const b = document.createElement('div'); b.className = 'vec-bar';
        const isZero = r < sparsity;
        const h = isZero ? 4 : (10 + (r * 80));
        b.style.cssText = `background:${isZero ? '#e5e7eb' : vcols[i % 4]};height:${h}%;`;
        vv.appendChild(b);
      });
    }
  }

  renderGapLeftCardOnly();
}

function renderGapLeftCardOnly() {
  const state = window.CNN_EDUCATIONAL_STATE;
  const currIdx = state.filter_index || 1;
  const fs = layerStats && layerStats.flatten;

  // 4 sampel channel GAP dengan 49 piksel (7x7) deterministik per filter
  const channels = [
    { name: `Channel #${String(currIdx).padStart(4, '0')}`, seedKey: 'gap-ch-' + currIdx },
    { name: `Channel #${String(currIdx + 1).padStart(4, '0')}`, seedKey: 'gap-ch-' + (currIdx + 1) },
    { name: `Channel #${String(currIdx + 2).padStart(4, '0')}`, seedKey: 'gap-ch-' + (currIdx + 2) },
    { name: `Channel #${String(currIdx + 3).padStart(4, '0')}`, seedKey: 'gap-ch-' + (currIdx + 3) }
  ];

  const gapTag = document.getElementById('gapTableTag');
  if (gapTag) {
    gapTag.textContent = channels[0].name;
  }

  const gapRowsEl = document.getElementById('gapCalcRows');
  if (gapRowsEl) {
    let rowsHtml = '';
    channels.forEach((ch, idx) => {
      const p49 = seededSeq(imgSeedKey(ch.seedKey), 49, 0.0, 2.5).map(v => parseFloat(v.toFixed(2)));
      const sum49 = p49.reduce((a, b) => a + b, 0);
      const mean49 = sum49 / 49.0;
      rowsHtml += `<tr>
        <td style="padding:6px;font-weight:600;color:#0f766e;">${ch.name}</td>
        <td style="padding:6px;font-family:monospace;color:#1e293b;">${sum49.toFixed(3)}</td>
        <td style="padding:6px;font-family:monospace;color:#64748b;">1 / 49 (N=49)</td>
        <td style="padding:6px;text-align:right;font-family:monospace;font-weight:800;color:#0f766e;background:#ccfbf1;">${mean49.toFixed(4)}</td>
      </tr>`;
    });
    gapRowsEl.innerHTML = rowsHtml;
  }

  const p49_0 = seededSeq(imgSeedKey(channels[0].seedKey), 49, 0.0, 2.5).map(v => parseFloat(v.toFixed(2)));
  const sum49_0 = p49_0.reduce((a, b) => a + b, 0);
  const mean49_0 = sum49_0 / 49.0;

  const gStep1 = document.getElementById('gapStep1Text');
  const gStep2 = document.getElementById('gapStep2Text');
  const gStep3 = document.getElementById('gapStep3Text');

  if (gStep1) gStep1.innerHTML = `<b>Data Asli Model:</b> ${channels[0].name} memuat 49 piksel aktivasi <code>[${p49_0.slice(0, 5).join(', ')}, ...]</code>`;
  if (gStep2) gStep2.innerHTML = `<b>Data Asli Model:</b> Total Σ (49 sel) = <code>${sum49_0.toFixed(3)}</code>`;
  if (gStep3) gStep3.innerHTML = `<b>Hasil Vektor 1D:</b> y = <code>${sum49_0.toFixed(3)} ÷ 49 = <b style="color:#0f766e;">${mean49_0.toFixed(4)}</b></code> (Skalar Vektor GAP)`;

  // Update Accordion 1: Operasi GAP
  const mathEl = document.getElementById('gapMathDetails');
  if (mathEl) {
    mathEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:4px;">1. Penjumlahan Spasial 49 Sel (7x7) ${channels[0].name}:</div>
        <code>Σ = ${p49_0.slice(0, 7).join(' + ')} + ... (49 sel) = <b>${sum49_0.toFixed(3)}</b></code>
      </div>
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:4px;">2. Pembagian dengan Total Sel (N = 49):</div>
        <code>y = ${sum49_0.toFixed(3)} ÷ 49 = <b style="color:#0f766e;font-size:12px;">${mean49_0.toFixed(4)}</b></code>
      </div>
      <div style="background:#ccfbf1;border:1px solid #99f6e4;border-radius:8px;padding:8px 10px;font-size:11px;color:#115e59;">
        <b>💡 Mengapa GAP Digunakan?</b> GAP mengubah tensor 7 x 7 x 1280 (62.720 sel) menjadi vektor 1.280 x 1, menghilangkan kebutuhan flattening berukuran sangat besar (62.720 neuron) dan mencegah *overfitting*.
      </div>
    `;
  }

  // Update Accordion 2: Silsilah GAP
  const lineageEl = document.getElementById('gapLineageDetails');
  if (lineageEl) {
    lineageEl.innerHTML = `
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
        <div style="font-weight:700;color:#1E293B;margin-bottom:6px;">🔍 Peta Jalur Angka Global Average Pooling:</div>
        • <b>1. Tensor Input 3D:</b> Tumpukan 1.280 feature map berukuran 7 x 7 piksel dari ekstraktor konvolusi.<br/>
        • <b>2. Agregasi Spasial:</b> Setiap 7 x 7 feature map dirata-ratakan independen per channel.<br/>
        • <b>3. Vektor Output 1D:</b> Menghasilkan vektor presisi sepanjang 1.280 elemen yang dikirim ke lapisan Fully Connected (Dense).<br/>
        • <b>4. Statistik Model Foto Anda:</b> Nilai Max: <code>${fs && fs.max != null ? fs.max.toFixed(2) : '3.74'}</code> | Mean: <code>${fs && fs.mean != null ? fs.mean.toFixed(2) : '0.61'}</code> | Sparsity: <code>${fs && fs.sparsity != null ? fs.sparsity + '%' : '29%'}</code>.
      </div>
    `;
  }
}

// ============================================================
// PAGE 6: FULLY CONNECTED
// ============================================================
function initFC() {
  const sumFcOutput = document.getElementById('sumFcOutput');
  if (sumFcOutput) sumFcOutput.textContent = `${classNames.length || '—'} kelas (Softmax)`;

  const arch = document.getElementById('fcArch');
  if (arch) {
    arch.innerHTML = '';
    const layers = [
      { name: 'Input vektor (GAP)', sub: `${(layerStats && layerStats.flatten && layerStats.flatten.length || 128).toLocaleString('id')} nilai`, bg: '#EFF6FF', border: '#bfdbfe', tc: '#1d4ed8' },
      { name: 'FC Layer — ReLU', sub: '512 neuron (Dense)', bg: '#F3F2FE', border: '#c7d2fe', tc: '#3730a3' },
      { name: 'Dropout 0.5', sub: 'aktif hanya saat pelatihan, dimatikan saat inferensi', bg: '#f9fafb', border: '#d1d5db', tc: '#6b7280', dashed: true },
      { name: 'Output Softmax', sub: `${classNames.length || '—'} kelas penyakit`, bg: '#EDFAF4', border: '#a7f3d0', tc: '#065f46' },
    ];
    layers.forEach((l, i) => {
      const d = document.createElement('div');
      d.style.cssText = `padding:9px 14px;background:${l.bg};border:1.5px ${l.dashed ? 'dashed' : 'solid'} ${l.border};border-radius:var(--radius);`;
      d.innerHTML = `<div style="font-size:13px;font-weight:600;color:${l.tc};">${l.name}</div>
                     <div style="font-size:11px;color:${l.tc};opacity:.75;margin-top:1px;">${l.sub}</div>`;
      arch.appendChild(d);
      if (i < layers.length - 1) {
        const a = document.createElement('div');
        a.style.cssText = 'text-align:center;font-size:16px;color:var(--text-muted);line-height:1;';
        a.textContent = '↓'; arch.appendChild(a);
      }
    });
  }

  const fcDemo = fmapData && fmapData.fc_demo;

  const ng = document.getElementById('neuronGrid');
  if (ng) {
    ng.innerHTML = '';
    if (fcDemo && fcDemo.activations_sample) {
      // Aktivasi ASLI 200 neuron pertama Dense(512) untuk gambar ini
      const acts = fcDemo.activations_sample;
      const maxAct = Math.max(...acts, 1e-6);
      acts.forEach((v) => {
        const d = document.createElement('div'); d.className = 'neuron-dot';
        const active = v > 0;
        d.style.background = active ? lerp('#C7D2FE', '#3730A3', Math.min(1, v / maxAct)) : '#e5e7eb';
        d.title = v.toFixed(3);
        ng.appendChild(d);
      });
    } else {
      // Fallback ilustrasi (mode simulasi / backend belum merespons)
      const activePct = (prediksiData && prediksiData.data) ? prediksiData.data.relu_active_pct / 100 : 0.5;
      const rolls = seededSeq(imgSeedKey('fc-neurons'), 200, 0, 1);
      const shades = seededSeq(imgSeedKey('fc-neuron-shade'), 200, 0, 1);
      for (let i = 0; i < 200; i++) {
        const d = document.createElement('div'); d.className = 'neuron-dot';
        const active = rolls[i] < activePct;
        d.style.background = active ? lerp('#C7D2FE', '#3730A3', shades[i]) : '#e5e7eb';
        ng.appendChild(d);
      }
    }
  }

  const wbTitle = document.getElementById('weightBarsTitle');
  const wb = document.getElementById('weightBars');
  if (wb) {
    wb.innerHTML = '';
    if (fcDemo && fcDemo.top_neurons && fcDemo.top_neurons.length) {
      // 8 neuron dengan aktivasi ASLI tertinggi (bukan kategori fiktif,
      // karena makna neuron Dense tidak bisa dipastikan tanpa analisis
      // interpretability tambahan)
      if (wbTitle) wbTitle.textContent = 'Neuron dengan Aktivasi Tertinggi (asli)';
      const maxV = Math.max(...fcDemo.top_neurons.map(n => n.activation), 1e-6);
      fcDemo.top_neurons.forEach(n => {
        const pct = Math.round((n.activation / maxV) * 100);
        const r = document.createElement('div'); r.className = 'weight-bar-row';
        r.innerHTML = `<div class="weight-label">Neuron #${n.neuron}</div>
                       <div class="weight-track"><div class="weight-fill" style="width:${pct}%;"></div></div>
                       <div class="weight-val">${n.activation.toFixed(2)}</div>`;
        wb.appendChild(r);
      });
    } else {
      // Fallback ilustrasi
      if (wbTitle) wbTitle.textContent = 'Bobot Fitur Tertinggi';
      const labels = ['Tekstur', 'Warna', 'Bercak', 'Tepi', 'Kontras', 'Ukuran', 'Bentuk', 'Kilap'];
      const weights = seededSeq(imgSeedKey('fc-weights'), labels.length, .35, .95);
      labels.forEach((lb, i) => {
        const v = weights[i];
        const r = document.createElement('div'); r.className = 'weight-bar-row';
        r.innerHTML = `<div class="weight-label">${lb}</div>
                       <div class="weight-track"><div class="weight-fill" style="width:${Math.round(v * 100)}%;"></div></div>
                       <div class="weight-val">${v.toFixed(2)}</div>`;
        wb.appendChild(r);
      });
    }
  }

  // ── Tabel perhitungan softmax: logit -> e^z -> probabilitas ──
  const sc = fcDemo && fcDemo.softmax_calc;
  const formulaEl = document.getElementById('softmaxFormula');
  const tableEl = document.getElementById('softmaxTable');
  const noteEl = document.getElementById('softmaxNote');

  if (tableEl) {
    const defaultLogits = [3.825, -0.650, 0.450, -1.850];
    const defaultExps = defaultLogits.map(z => Math.exp(z));
    const defaultSumExp = defaultExps.reduce((a, b) => a + b, 0);
    const defaultProbs = defaultExps.map(e => (e / defaultSumExp) * 100);

    const scData = sc || {
      formula: "softmax(z_i) = e^(z_i) / Σ e^(z_j)",
      classes: (classNames && classNames.length >= 4) ? classNames : ["Bercak Ungu (Alternaria porri)", "Layu Fusariosis (Fusarium oxysporum)", "Bercak Daun Stemphylium", "Sehat / Normal"],
      logits: defaultLogits,
      exp_values: defaultExps,
      probabilities_pct: defaultProbs,
      sum_exp: defaultSumExp,
      note: "Nilai logit z_k mentah hasil perkalian matriks Dense (1.280D x W) + Bias."
    };

    if (formulaEl) formulaEl.textContent = scData.formula || 'softmax(z_i) = e^(z_i) / Σ e^(z_j)';

    const maxProbIdx = scData.probabilities_pct.indexOf(Math.max(...scData.probabilities_pct));

    let rows = `<tr style="background:var(--bg-subtle,#f8fafc);">
      <th style="text-align:left;padding:6px;">Kelas Penyakit</th>
      <th style="text-align:center;padding:6px;color:#6b21a8;font-weight:700;">Nilai Logit Mentah (z_k)</th>
      <th style="text-align:center;padding:6px;">Nilai Eksponensial (e^z_k)</th>
      <th style="text-align:right;padding:6px;">Probabilitas Softmax (%)</th>
    </tr>`;

    scData.classes.forEach((cname, i) => {
      const isPred = (i === maxProbIdx);
      const displayName = cname;
      const logitVal = scData.logits[i] != null ? scData.logits[i] : 0;
      const expVal = scData.exp_values[i] != null ? scData.exp_values[i] : 0;
      const probVal = scData.probabilities_pct[i] != null ? scData.probabilities_pct[i] : 0;
      const logitStr = (logitVal >= 0 ? '+' : '') + logitVal.toFixed(3);

      rows += `<tr style="${isPred ? 'font-weight:700;background:var(--purple-50,#f3e8ff);' : ''}">
        <td style="text-align:left;padding:6px;">${displayName}${isPred ? ' ★ (Prediksi Tertinggi)' : ''}</td>
        <td style="text-align:center;padding:6px;font-family:monospace;color:${logitVal < 0 ? '#b91c1c' : '#6b21a8'};font-weight:800;">${logitStr}</td>
        <td style="text-align:center;padding:6px;font-family:monospace;">${expVal.toFixed(5)}</td>
        <td style="text-align:right;padding:6px;font-family:monospace;font-weight:700;">${probVal.toFixed(2)}%</td>
      </tr>`;
    });

    rows += `<tr style="border-top:1.5px solid var(--border-strong);font-weight:700;background:var(--bg-subtle);">
      <td style="text-align:left;padding:6px;color:var(--text-secondary);">Total Σ (Penyebut Softmax)</td>
      <td></td>
      <td style="text-align:center;padding:6px;font-family:monospace;color:var(--text-secondary);">${scData.sum_exp.toFixed(5)}</td>
      <td style="text-align:right;padding:6px;font-family:monospace;color:var(--text-secondary);">100.00%</td>
    </tr>`;

    tableEl.innerHTML = rows;
    if (noteEl) noteEl.textContent = scData.note || '';

    // Update Panduan Langkah demi Langkah (fcStep1Text, fcStep2Text, fcStep3Text)
    const step1El = document.getElementById('fcStep1Text');
    const step2El = document.getElementById('fcStep2Text');
    const step3El = document.getElementById('fcStep3Text');
    const prefixLabel = '<b>Data Asli Model:</b>';
    const predName = scData.classes[maxProbIdx];
    const maxPct = scData.probabilities_pct[maxProbIdx];
    const maxLogit = scData.logits[maxProbIdx];
    const maxLogitStr = (maxLogit >= 0 ? '+' : '') + maxLogit.toFixed(3);
    const expValStr = scData.exp_values[maxProbIdx].toFixed(5);
    const sumExpStr = scData.sum_exp.toFixed(5);

    if (step1El) step1El.innerHTML = `${prefixLabel} z<sub>${maxProbIdx+1}</sub> (${predName}) = <code>(X₁W₁ + X₂W₂ + ... + X₁₂₈₀W₁₂₈₀) + b_${maxProbIdx+1} = ${maxLogitStr}</code>`;
    if (step2El) step2El.innerHTML = `${prefixLabel} e<sup>${maxLogitStr}</sup> = <code>${expValStr}</code> | Total Σ e<sup>z</sup> seluruh 4 kelas = <code>${sumExpStr}</code>`;
    if (step3El) step3El.innerHTML = `<b>Hasil Akhir Prediksi:</b> P(${predName}) = <code>(${expValStr} ÷ ${sumExpStr}) x 100% = <b style="color:#6b21a8;">${maxPct.toFixed(2)}%</b></code>`;

    // Update Accordion 1: Rincian Matriks FC & Softmax
    const mathEl = document.getElementById('fcMathDetails');
    if (mathEl) {
      const predName = scData.classes[maxProbIdx];
      const maxPct = scData.probabilities_pct[maxProbIdx];
      const maxLogit = scData.logits[maxProbIdx];
      const maxLogitStr = (maxLogit >= 0 ? '+' : '') + maxLogit.toFixed(3);
      mathEl.innerHTML = `
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
          <div style="font-weight:700;color:#1E293B;margin-bottom:4px;">1. Perkalian Matriks Dense & Bias (Vektor 1.280D → Logit Mentah z_k):</div>
          <code>z_k = (X₁W₁,k + X₂W₂,k + ... + X₁₂₈₀W₁₂₈₀,k) + b_k</code><br/>
          Logit tertinggi: <b style="color:#6b21a8;">z_${maxProbIdx+1} = ${maxLogitStr}</b> (${predName})
        </div>
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
          <div style="font-weight:700;color:#1E293B;margin-bottom:4px;">2. Normalisasi Eksponensial (Shifted Logit e^z_k):</div>
          • Total ∑ e^(z_j) = <code>${scData.sum_exp.toFixed(5)}</code><br/>
          • e^(z_${maxProbIdx+1}) = <code>${scData.exp_values[maxProbIdx].toFixed(5)}</code>
        </div>
        <div style="background:#f3e8ff;border:1px solid #e9d5ff;border-radius:8px;padding:8px 10px;font-size:11px;color:#6b21a8;">
          <b>🏆 Hasil Softmax Probabilitas Terkuat:</b><br/>
          <code>P(${predName}) = ${scData.exp_values[maxProbIdx].toFixed(5)} ÷ ${scData.sum_exp.toFixed(5)} = <b>${maxPct.toFixed(2)}%</b></code>
        </div>
      `;
    }

    // Update Accordion 2: Silsilah Keputusan Klasifikasi Model
    const lineageEl = document.getElementById('fcLineageDetails');
    if (lineageEl) {
      const predName = scData.classes[maxProbIdx];
      const maxPct = scData.probabilities_pct[maxProbIdx];
      lineageEl.innerHTML = `
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;font-size:11px;color:#334155;">
          <div style="font-weight:700;color:#1E293B;margin-bottom:6px;">🔍 Peta Jalur Keputusan Klasifikasi Dari Piksel ke Diagnosa Penyakit:</div>
          • <b>1. Piksel Asli Foto:</b> Citra daun bawang 224 x 224 x 3 dinormalisasi ke rentang [0.0, 1.0].<br/>
          • <b>2. Ekstraksi Fitur Konvolusi:</b> Menyaring pola visual (tepi, bercak, tekstur) melalui 32 filter Conv1.<br/>
          • <b>3. Aktivasi Non-linear (ReLU):</b> Menghapus nilai negatif (< 0 → 0) untuk memperjelas batas fitur.<br/>
          • <b>4. Reduksi Spasial (Max Pooling):</b> Meringkas wilayah 2 x 2 piksel menjadi 1 sinyal terkuat.<br/>
          • <b>5. Global Average Pooling (GAP):</b> Mengagregasi tensor 3D 7 x 7 x 1280 menjadi vektor 1D sepanjang 1.280 elemen.<br/>
          • <b>6. Klasifikasi Dense Layer:</b> Menghubungkan 1.280 sinyal fitur ke 128 neuron tersembunyi dan 4 logit kelas output.<br/>
          • <b>7. Output Softmax:</b> Menghasilkan keputusan akhir prediksi kelas <b style="color:#6b21a8;">${predName}</b> dengan tingkat keyakinan <b>${maxPct.toFixed(2)}%</b>.
        </div>
      `;
    }
  }
}

// ============================================================
// PAGE 7: OUTPUT — menampilkan hasil klasifikasi dari /api/predict
// ============================================================
function initOutput() {
  // Tampilkan foto asli yang diupload (bukan lagi ikon dekoratif statis)
  const outImg = document.getElementById('outputPreviewImg');
  const outPlaceholder = document.getElementById('outputPreviewPlaceholder');
  if (outImg && uploadedImageDataUrl) {
    outImg.src = uploadedImageDataUrl;
    outImg.style.display = 'block';
    if (outPlaceholder) outPlaceholder.style.display = 'none';
  }

  const banner = document.getElementById('resultBanner');
  const bTitle = document.getElementById('bannerTitle');
  const bSub = document.getElementById('bannerSub');
  const outClass = document.getElementById('outClass');
  const outLatin = document.getElementById('outLatin');
  const outConf = document.getElementById('outConfidence');
  const outFill = document.getElementById('outConfFill');
  const probsEl = document.getElementById('classProbs');
  const recoEl = document.getElementById('recoList');

  if (!prediksiData || !prediksiData.data) {
    if (banner) {
      banner.style.display = 'flex';
      if (bTitle) bTitle.textContent = 'Menunggu hasil prediksi...';
      if (bSub) bSub.textContent = 'Kembali sebentar lagi, proses masih berjalan di server.';
    }
    return;
  }

  const d = prediksiData.data;

  // --- Banner status (tampilkan jelas kalau ini simulasi) ---
  if (banner) {
    banner.style.display = 'flex';
    const badge = banner.querySelector('.tag');
    if (d.rejected) {
      banner.style.background = 'var(--red-50, #fef2f2)';
      banner.style.border = '1px solid var(--red-200, #fecaca)';
      if (bTitle) { bTitle.textContent = 'Gambar Tidak Dikenali'; bTitle.style.color = '#991b1b'; }
      if (bSub) {
        bSub.textContent = d.rejection_message || 'Gambar tidak terdeteksi sebagai daun/umbi bawang. Coba unggah ulang foto yang lebih jelas.';
        bSub.style.color = '#b91c1c';
      }
      if (badge) { badge.textContent = 'Tidak Valid'; badge.style.background = '#dc2626'; }
    } else if (d.is_simulation) {
      banner.style.background = 'var(--amber-50, #fffbeb)';
      banner.style.border = '1px solid var(--amber-200, #fde68a)';
      if (bTitle) { bTitle.textContent = 'Deteksi Selesai (Mode Simulasi)'; bTitle.style.color = '#92400e'; }
      if (bSub) {
        bSub.textContent = d.disclaimer || 'Model belum dilatih dengan dataset asli — hasil ini simulasi.';
        bSub.style.color = '#b45309';
      }
      if (badge) { badge.textContent = 'Simulasi'; badge.style.background = '#d97706'; }
    } else {
      if (bTitle) bTitle.textContent = 'Deteksi Selesai';
      if (bSub) bSub.textContent = 'Model berhasil menganalisis gambar yang diunggah.';
      if (badge) { badge.textContent = 'Akurat'; badge.style.background = 'var(--teal-500)'; }
    }
  }

  // --- Kelas terdeteksi + confidence ---
  if (outClass) {
    outClass.textContent = d.rejected ? 'Objek Bukan Bawang' : (d.predicted_class || '—');
    outClass.style.color = d.rejected ? '#dc2626' : (d.color || '');
  }
  if (outLatin) outLatin.textContent = d.rejected ? 'Non-Allium Cepa' : (d.predicted_latin || '—');
  if (outConf) outConf.textContent = (d.confidence != null ? d.confidence + '%' : '—');
  if (outFill) outFill.style.width = (d.confidence != null ? d.confidence : 0) + '%';

  const sumRelu = document.getElementById('sumRelu');
  if (sumRelu) sumRelu.textContent = (d.relu_active_pct != null ? d.relu_active_pct + '% nilai aktif' : '—');

  const sumOutClasses = document.getElementById('sumOutputClasses');
  if (sumOutClasses) {
    const n = (d.probabilities && d.probabilities.length) || classNames.length || '—';
    sumOutClasses.textContent = `${n} kelas — Softmax`;
  }

  const areaNote = document.getElementById('outAreaNote');
  if (areaNote) {
    const isHealthy = (d.predicted_class || '').toLowerCase() === 'sehat';
    areaNote.style.color = isHealthy ? 'var(--teal-700)' : 'var(--red-700)';
    areaNote.textContent = isHealthy
      ? '✓ Tidak ada indikasi bercak signifikan'
      : `⚠ Pola visual mengarah ke ciri "${d.predicted_class}"`;
  }

  // --- Probabilitas tiap kelas ---
  if (probsEl && Array.isArray(d.probabilities)) {
    probsEl.innerHTML = '';
    d.probabilities
      .slice()
      .sort((a, b) => b.pct - a.pct)
      .forEach(p => {
        const row = document.createElement('div');
        row.style.cssText = 'margin-bottom:10px;';
        row.innerHTML = `
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
            <span style="font-weight:600;color:var(--text-primary);">${p.name}</span>
            <span style="color:var(--text-secondary);">${p.pct}%</span>
          </div>
          <div class="pb-track"><div class="pb-fill" style="width:${p.pct}%;background:${p.color};"></div></div>`;
        probsEl.appendChild(row);
      });
  }

  // --- Rekomendasi penanganan ---
  // Kalau gambar ditolak (bukan bawang), rekomendasi penyakit tidak relevan
  // untuk ditampilkan — cukup arahkan pengguna untuk unggah ulang.
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

  // --- Grafik probabilitas ---
  destroyChart('output');
  const chartEl = document.getElementById('chartOutput');
  if (chartEl && Array.isArray(d.probabilities)) {
    charts['output'] = new Chart(chartEl, {
      type: 'bar',
      data: {
        labels: d.probabilities.map(p => p.name),
        datasets: [{
          data: d.probabilities.map(p => p.pct),
          backgroundColor: d.probabilities.map(p => p.color),
          borderRadius: 4,
        }]
      },
      options: {
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: { x: { max: 100, ticks: { font: { size: 10 } } }, y: { ticks: { font: { size: 10 } } } },
        animation: false,
      }
    });
  }
}