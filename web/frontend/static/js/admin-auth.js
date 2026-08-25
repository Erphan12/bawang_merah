// ============================================================
// admin-auth.js — helper autentikasi dipakai bersama oleh admin.js &
// report.js (halaman-halaman yang HANYA boleh diakses admin yang sudah
// login). Harus di-include SEBELUM admin.js / report.js di HTML.
//
// Cara kerja:
//   1. Begitu halaman dimuat, kalau tidak ada token di localStorage ->
//      langsung dilempar ke login.html (tidak perlu tunggu request API).
//   2. adminFetch() dipakai menggantikan fetch() biasa untuk SEMUA
//      endpoint /api/admin/* dan /api/history* -- otomatis menambahkan
//      header Authorization, dan kalau server balas 401 (token tidak
//      valid/kedaluwarsa) token lama dihapus & user dilempar ke login.html.
// ============================================================
const metaApiAuth = document.querySelector('meta[name="api-base"]');
const API_BASE_AUTH = metaApiAuth ? metaApiAuth.content : 'http://localhost:5000';

function currentPageFile() {
  const path = window.location.pathname.split('/');
  return path[path.length - 1] || 'admin.html';
}

function goToLogin() {
  localStorage.removeItem('adminToken');
  window.location.replace(`login.html?next=${encodeURIComponent(currentPageFile())}`);
}

// Panggil di awal setiap halaman admin-only (admin.html, report.html).
function requireAdminAuth() {
  if (!localStorage.getItem('adminToken')) {
    goToLogin();
    return false;
  }
  return true;
}

// Pengganti fetch() biasa untuk endpoint admin -- menambahkan header
// Authorization dan menangani token yang tidak valid/kedaluwarsa secara
// otomatis (redirect ke login.html).
async function adminFetch(url, options = {}) {
  const token = localStorage.getItem('adminToken');
  if (!token) {
    goToLogin();
    throw new Error('Belum login');
  }
  const headers = Object.assign({}, options.headers, { Authorization: `Bearer ${token}` });
  const res = await fetch(url, Object.assign({}, options, { headers }));
  if (res.status === 401) {
    goToLogin();
    throw new Error('Sesi admin berakhir, silakan login kembali.');
  }
  return res;
}

async function adminLogout() {
  try {
    await adminFetch(`${API_BASE_AUTH}/api/admin/logout`, { method: 'POST' });
  } catch (e) {
    // token sudah tidak valid / server tidak terjangkau -- tidak masalah,
    // tetap lanjut hapus token lokal & kembali ke halaman login di bawah.
  }
  localStorage.removeItem('adminToken');
  window.location.href = 'login.html';
}
