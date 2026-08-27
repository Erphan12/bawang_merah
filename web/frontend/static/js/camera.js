// ============================================================
// camera.js — Akses kamera langsung lewat browser (bukan lewat File Explorer).
// Dipakai oleh tombol "Ambil Foto Langsung" di halaman 01_input.
// Hasil jepretan dikirim ke uploadFile() yang sama persis dipakai
// drag&drop dan pilih file (lihat main.js), supaya alurnya konsisten.
// ============================================================

let cameraStream = null;
let currentFacingMode = 'user'; // 'user' = kamera depan, 'environment' = kamera belakang

function triggerNativeCamera() {
  closeCamera();
  let input = document.getElementById('nativeCameraPickerInput');
  if (!input) {
    input = document.createElement('input');
    input.id = 'nativeCameraPickerInput';
    input.type = 'file';
    input.accept = 'image/*';
    input.setAttribute('capture', 'environment');
    input.style.display = 'none';
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file && typeof uploadFile === 'function') {
        uploadFile(file);
      }
    };
    document.body.appendChild(input);
  }
  input.click();
}

async function openCamera() {
  const modal = document.getElementById('cameraModal');
  const video = document.getElementById('cameraVideo');
  const errBox = document.getElementById('cameraError');
  const videoWrap = video ? video.parentElement : null;
  if (!modal || !video) return;

  // Jika WebRTC getUserMedia tidak didukung (misal diakses via HTTP IP VPS non-HTTPS),
  // langsung panggil kamera bawaan HTML5 (capture="environment") agar tidak terhambat modal error!
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    triggerNativeCamera();
    return;
  }

  modal.style.display = 'flex';
  errBox.style.display = 'none';
  video.style.display = 'block';

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: currentFacingMode },
      audio: false,
    });
    video.srcObject = cameraStream;

    if (videoWrap) videoWrap.classList.toggle('rear', currentFacingMode === 'environment');

    // Cek apakah perangkat ini punya lebih dari satu kamera (mis. HP: depan & belakang).
    // Laptop biasanya cuma satu (webcam depan) -> tombol "Ganti Kamera" disembunyikan.
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoInputs = devices.filter(d => d.kind === 'videoinput');
    const btnSwitch = document.getElementById('btnSwitchCamera');
    if (btnSwitch) btnSwitch.style.display = videoInputs.length > 1 ? 'flex' : 'none';

  } catch (err) {
    console.error('Gagal mengakses kamera:', err);
    let msg = 'Tidak dapat mengakses kamera WebRTC.';
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      msg = 'Akses kamera ditolak oleh browser. Gunakan pemilih foto perangkat.';
    } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      msg = 'Tidak ditemukan perangkat kamera WebRTC.';
    }
    showCameraError(msg);
  }
}

function showCameraError(msg) {
  const video  = document.getElementById('cameraVideo');
  const errBox = document.getElementById('cameraError');
  if (video) video.style.display = 'none';
  if (errBox) {
    errBox.innerHTML = `${msg}<br><br><button class="btn btn-primary" style="margin-top:6px;" onclick="closeCamera();triggerNativeCamera();">Pilih / Bidik dari perangkat</button>`;
    errBox.style.display = 'flex';
  }
  const btnCapture = document.getElementById('btnCapture');
  if (btnCapture) btnCapture.style.display = 'none';
}

function switchCamera() {
  currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
  stopCameraStream();
  openCamera();
}

function stopCameraStream() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
}

function closeCamera() {
  stopCameraStream();
  const modal = document.getElementById('cameraModal');
  const btnCapture = document.getElementById('btnCapture');
  if (modal) modal.style.display = 'none';
  if (btnCapture) btnCapture.style.display = 'flex';
}

function capturePhoto() {
  const video  = document.getElementById('cameraVideo');
  const canvas = document.getElementById('cameraCanvas');
  if (!video || !canvas || !cameraStream) return;

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(blob => {
    if (!blob) {
      showToast('Gagal mengambil foto, coba lagi.', 'error');
      return;
    }
    const filename = `kamera_${Date.now()}.jpg`;
    const file = new File([blob], filename, { type: 'image/jpeg' });
    closeCamera();
    uploadFile(file); // fungsi yang sama dipakai drag&drop / pilih file (lihat main.js)
  }, 'image/jpeg', 0.92);
}
