// ============================================================
// HELPER: MATEMATIS & RENDER WARNA
// ============================================================
function lerp(lo, hi, t) {
  const p = h => [parseInt(h.slice(1,3),16), parseInt(h.slice(3,5),16), parseInt(h.slice(5,7),16)];
  const a = p(lo), b = p(hi);
  return `rgb(${Math.round(a[0]+(b[0]-a[0])*t)},${Math.round(a[1]+(b[1]-a[1])*t)},${Math.round(a[2]+(b[2]-a[2])*t)})`;
}

function rnd(lo, hi) { 
  return lo + Math.random() * (hi - lo); 
}

// ============================================================
// HELPER: ANGKA "ACAK" YANG DI-SEED PER GAMBAR
// ------------------------------------------------------------
// Beberapa kartu di halaman pipeline (contoh patch konvolusi, grid
// pooling, batang vektor, bobot fitur) cuma ilustrasi — datanya tidak
// dikirim backend per piksel. Sebelumnya dipakai Math.random() polos,
// akibatnya nilai yang tampil BERUBAH-UBAH tiap kali pindah halaman
// walau gambar yang dianalisis SAMA. Dengan seed dari session_id gambar
// yang sedang aktif, hasilnya konsisten selama gambar yang sama, tapi
// otomatis berbeda begitu ganti/unggah gambar lain.
// ============================================================
function _hashSeed(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function () {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  };
}

// Urutan N angka acak yang deterministik untuk `key` tertentu.
function seededSeq(key, n, lo, hi) {
  const gen = _hashSeed(String(key));
  gen(); // buang nilai pertama (kualitas seed lebih baik)
  const out = [];
  for (let i = 0; i < n; i++) out.push(lo + gen() * (hi - lo));
  return out;
}

// Satu angka acak deterministik untuk `key` tertentu.
function seededVal(key, lo, hi) {
  return seededSeq(key, 1, lo, hi)[0];
}

// Kunci dasar = gambar yang sedang aktif (session_id). Ditambah label
// per elemen (mis. 'conv-patch') supaya tiap elemen tetap independen.
function imgSeedKey(label) {
  const base = (typeof sessionId !== 'undefined' && sessionId) ? sessionId : 'no-session';
  return base + '::' + label;
}

// ============================================================
// HELPER: RENDER FEATURE MAPS
// ============================================================
function buildFmap(el, lo, hi, zeroChance = 0, seedLabel = 'fmap') {
  el.innerHTML = '';
  el.className = 'fmap-grid g8';
  const rolls = seededSeq(imgSeedKey(seedLabel + '-val'), 64, 0, 1);
  const zeros = seededSeq(imgSeedKey(seedLabel + '-zero'), 64, 0, 1);
  for (let i = 0; i < 64; i++) {
    const c = document.createElement('div');
    c.className = 'fmap-cell';
    const zero = zeroChance > 0 && zeros[i] < zeroChance;
    c.style.background = zero ? '#e5e7eb' : lerp(lo, hi, rolls[i]);
    el.appendChild(c);
  }
}

// ============================================================
// HELPER: KARTU FILTER BISA DIKLIK -> TAMPILKAN BOBOT KERNEL 3x3
// ------------------------------------------------------------
// kernel3x3: array 3x3 (list of 3 list) berisi bobot kernel ASLI dari
// backend (atau nilai ilustrasi ter-seed kalau backend belum merespons).
// null artinya filter ini BUKAN kernel konvolusi 3x3 (mis. fitur warna
// turunan seperti "warna hijau"/"kecerahan"), jadi tidak ada bobot untuk
// ditampilkan -- panel akan menjelaskan itu apa adanya, bukan mengarang angka.
function attachKernelClick(wrap, kernel3x3, stageNote, filterIndex) {
  wrap.classList.add('fmap-clickable');
  wrap.setAttribute('role', 'button');
  wrap.setAttribute('tabindex', '0');
  wrap.title = 'Klik untuk pilih filter ini & lihat simulasi rincinya di sisi kiri';

  const openPanel = () => {
    if (typeof filterIndex === 'number' && typeof window.selectFilterForEducationalState === 'function') {
      window.selectFilterForEducationalState(filterIndex);
    }

    let panel = wrap.querySelector('.fmap-kernel-panel');
    if (!panel) {
      panel = document.createElement('div');
      panel.className = 'fmap-kernel-panel';
      if (kernel3x3) {
        const title = document.createElement('div');
        title.className = 'fmap-kernel-title';
        title.textContent = 'Bobot Kernel 3x3';
        panel.appendChild(title);

        const grid = document.createElement('div');
        grid.className = 'fmap-kernel-grid';
        kernel3x3.flat().forEach(v => {
          const c = document.createElement('div'); c.className = 'fmap-kernel-cell';
          const t = Math.max(0, Math.min(1, (v + 1) / 2));
          c.style.background = lerp('#FCEBEB', '#085041', t);
          c.style.color = v < 0 ? '#b91c1c' : '#085041';
          c.textContent = v.toFixed(2);
          grid.appendChild(c);
        });
        panel.appendChild(grid);

        if (stageNote) {
          const note = document.createElement('div');
          note.className = 'fmap-kernel-note';
          note.style.marginTop = '6px';
          note.textContent = stageNote;
          panel.appendChild(note);
        }
      } else {
        const note = document.createElement('div');
        note.className = 'fmap-kernel-note';
        note.textContent = 'Filter ini bukan kernel konvolusi 3x3, melainkan fitur turunan (warna/kecerahan) — tidak ada bobot untuk ditampilkan.';
        panel.appendChild(note);
      }
      wrap.appendChild(panel);
    }
    wrap.classList.toggle('expanded');
  };

  wrap.addEventListener('click', openPanel);
  wrap.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPanel(); }
  });
}

function renderFmapFromAPI(container, maps, badge, badgeStyle) {
  container.innerHTML = '';
  maps.forEach((m, i) => {
    const wrap = document.createElement('div'); wrap.className = 'fmap-card';
    const hdr  = document.createElement('div'); hdr.className = 'fmap-header';
    hdr.innerHTML = `<span class="fmap-name">${m.label}</span>
                     <span class="fmap-badge" style="${badgeStyle}">${badge}</span>`;
    wrap.appendChild(hdr);

    if (m.image_b64) {
      const img = document.createElement('img');
      img.src   = 'data:image/png;base64,' + m.image_b64;
      img.style.cssText = 'width:100%;display:block;image-rendering:pixelated;';
      wrap.appendChild(img);
    } else {
      const grid = document.createElement('div'); grid.className = 'fmap-grid g8';
      buildFmap(grid, '#F3F2FE', '#3C3489', badge === 'ReLU' ? 0.3 : 0, badge + '-' + i);
      wrap.appendChild(grid);
    }
    const hint = document.createElement('div'); hint.className = 'fmap-click-hint';
    hint.textContent = 'Klik untuk lihat bobot kernel ›';
    wrap.appendChild(hint);

    container.appendChild(wrap);

    let kernelMatrix;
    if (i === 0 && window.CNN_EDUCATIONAL_STATE && Array.isArray(window.CNN_EDUCATIONAL_STATE.filterData)) {
      const f = window.CNN_EDUCATIONAL_STATE.filterData;
      kernelMatrix = [
        [f[0], f[1], f[2]],
        [f[3], f[4], f[5]],
        [f[6], f[7], f[8]]
      ];
    } else {
      kernelMatrix = m.kernel;
    }
    attachKernelClick(wrap, kernelMatrix || null, m.kernel_stage_note || null, i);
  });
}

function buildFmapCanvas(parentEl, lo, hi, seedLabel = 'fmap') {
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  canvas.style.cssText = 'width:100%;aspect-ratio:1;display:block;';
  const ctx = canvas.getContext('2d');
  
  const seq = seededSeq(imgSeedKey(seedLabel + '-canvas'), 16 * 16, 0, 1);
  const imgData = ctx.createImageData(64, 64);
  
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const gx = Math.min(15, Math.floor(x / 4));
      const gy = Math.min(15, Math.floor(y / 4));
      const v = seq[gy * 16 + gx];
      const col = lerp(lo, hi, v);
      const matches = col.match(/\d+/g);
      const idx = (y * 64 + x) * 4;
      if (matches) {
        imgData.data[idx]     = parseInt(matches[0]);
        imgData.data[idx + 1] = parseInt(matches[1]);
        imgData.data[idx + 2] = parseInt(matches[2]);
        imgData.data[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
  parentEl.appendChild(canvas);
}

function makeFmapsLocal(container, palettes, badge, badgeStyle, zeroChance = 0, count = 32) {
  container.innerHTML = '';
  const stageNote = (badge === 'ReLU' || badge === 'Pool')
    ? 'Bobot ini sama dengan Filter ini di halaman Konvolusi — ReLU dan Pooling tidak punya bobot sendiri, keduanya hanya memproses hasil konvolusi (aktivasi & downsampling).'
    : null;
  for (let i = 0; i < count; i++) {
    const pal = badge === 'ReLU' ? ['#dcfce7', '#15803d'] : (badge === 'Pool' ? ['#faeeda', '#854f0b'] : palettes[i % palettes.length]);
    const wrap = document.createElement('div'); wrap.className = 'fmap-card';
    const hdr  = document.createElement('div'); hdr.className = 'fmap-header';
    const nameStr = ((badge === 'Conv' || badge === 'ReLU' || badge === 'Pool') && i < convLabels.length)
      ? convLabels[i]
      : `Filter #${String(i + 1).padStart(2, '0')} — ~fitur visual`;
    hdr.innerHTML = `<span class="fmap-name">${nameStr}</span>
                     <span class="fmap-badge" style="${badgeStyle}">${badge}</span>`;
    
    wrap.appendChild(hdr);

    if (badge === 'Conv' || badge === 'ReLU' || badge === 'Pool') {
      buildFmapCanvas(wrap, pal[0], pal[1], badge + '-' + i);
    } else {
      const grid = document.createElement('div'); grid.className = 'fmap-grid g8';
      wrap.appendChild(grid);
      buildFmap(grid, pal[0], pal[1], zeroChance, badge + '-' + i);
    }

    const hint = document.createElement('div'); hint.className = 'fmap-click-hint';
    hint.textContent = 'Klik untuk lihat bobot kernel ›';
    wrap.appendChild(hint);

    container.appendChild(wrap);

    let flatK;
    if (i === 0 && window.CNN_EDUCATIONAL_STATE && Array.isArray(window.CNN_EDUCATIONAL_STATE.filterData)) {
      flatK = window.CNN_EDUCATIONAL_STATE.filterData.map(v => Math.round(v * 100) / 100);
    } else {
      flatK = seededSeq(imgSeedKey(badge + '-kernel-' + i), 9, -0.5, 0.5).map(v => Math.round(v * 100) / 100);
    }
    attachKernelClick(wrap, [flatK.slice(0, 3), flatK.slice(3, 6), flatK.slice(6, 9)], stageNote, i);
  }
}

// ============================================================
// DATA PALET & LABEL
// ============================================================
const convPalettes = [['#818cf8','#312e81'],['#94a3b8','#1e293b'],['#d97706','#451a03'],['#4ade80','#14532d']];
const convLabels   = [
  'Filter #01 — ~tepi diagonal',
  'Filter #02 — ~perataan (blur)',
  'Filter #03 — ~tepi vertikal',
  'Filter #04 — ~bentuk/tepi silang',
  'Filter #05 — ~deteksi warna hijau',
  'Filter #06 — ~kontras kecerahan',
  'Filter #07 — ~tekstur halus',
  'Filter #08 — ~tepi horizontal'
];