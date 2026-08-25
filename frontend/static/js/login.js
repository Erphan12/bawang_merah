// ============================================================
// login.js — logika halaman login.html (autentikasi admin)
// Token disimpan di localStorage sebagai 'adminToken' dan dipakai oleh
// admin.js / report.js lewat header "Authorization: Bearer <token>".
// ============================================================
const metaApi = document.querySelector('meta[name="api-base"]');
const API_BASE = (metaApi && metaApi.content && !metaApi.content.includes('localhost'))
  ? metaApi.content
  : window.location.origin;

// Kalau sudah punya token di localStorage, coba langsung lanjut ke admin.html
// (token akan divalidasi ulang di sana; kalau ternyata sudah kedaluwarsa,
// admin.js akan melempar balik ke halaman ini).
window.addEventListener('DOMContentLoaded', () => {
  const existing = localStorage.getItem('adminToken');
  if (existing) {
    window.location.replace(redirectTarget());
  }
});

function redirectTarget() {
  const params = new URLSearchParams(window.location.search);
  const next = params.get('next');
  // Hanya izinkan redirect ke halaman internal (bukan URL luar) demi keamanan.
  if (next && /^[a-zA-Z0-9_\-]+\.html$/.test(next)) return next;
  return 'admin.html';
}

async function handleLogin(event) {
  event.preventDefault();
  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;
  const btn = document.getElementById('loginBtn');
  const errBox = document.getElementById('loginError');
  errBox.style.display = 'none';
  btn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/admin/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();

    if (!data.success) {
      document.getElementById('loginErrorMsg').textContent = data.error || 'Login gagal.';
      errBox.style.display = 'flex';
      btn.disabled = false;
      return false;
    }

    localStorage.setItem('adminToken', data.token);
    window.location.href = redirectTarget();
  } catch (err) {
    document.getElementById('loginErrorMsg').textContent = 'Tidak dapat terhubung ke server. Pastikan backend Flask menyala.';
    errBox.style.display = 'flex';
    btn.disabled = false;
  }
  return false;
}
