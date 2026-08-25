"""
Script untuk menjalankan Ngrok tunnel secara otomatis.
Jalankan: python start_ngrok.py
"""
from pyngrok import ngrok, conf

# Set authtoken
conf.get_default().auth_token = "39GLFy8izbeeaeTOIx2btUqRvn8_5DAfStPQ1MdUJy4yEyFn"

# Buka tunnel ke port 8000
tunnel = ngrok.connect(8000, "http")

print("=" * 60)
print("  NGROK TUNNEL AKTIF!")
print("=" * 60)
print(f"  Link Publik: {tunnel.public_url}")
print(f"  Forwarding:  {tunnel.public_url} -> http://localhost:8000")
print("=" * 60)
print()
print("  Bagikan link di atas ke siapapun untuk testing.")
print("  Tekan Ctrl+C untuk menghentikan tunnel.")
print()

# Tetap berjalan sampai user tekan Ctrl+C
try:
    ngrok_process = ngrok.get_ngrok_process()
    ngrok_process.proc.wait()
except KeyboardInterrupt:
    print("\nMenghentikan tunnel...")
    ngrok.kill()
    print("Tunnel dimatikan.")
