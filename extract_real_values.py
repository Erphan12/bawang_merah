"""
Skrip ekstraksi nilai asli dari models/best_bawang_model.h5 - versi diperbaiki
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import json
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_bawang_model.h5')

print("=" * 60)
print("EKSTRAKSI NILAI ASLI DARI MODEL best_bawang_model.h5")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

# Temukan base model
base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and len(layer.layers) > 10:
        base_model = layer
        break

# ---- CONV1 ----
conv1 = None
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        conv1 = layer
        break

print(f"\n[CONV1] Layer: {conv1.name}, use_bias={conv1.use_bias}")
kernel_np = conv1.kernel.numpy()  # (3,3,3,32)
w_f1 = np.mean(kernel_np[:, :, :, 0], axis=2)
print(f"  Kernel Filter #1 (avg 3 channel RGB):")
print(f"  Baris 1: [{w_f1[0,0]:+.5f}, {w_f1[0,1]:+.5f}, {w_f1[0,2]:+.5f}]")
print(f"  Baris 2: [{w_f1[1,0]:+.5f}, {w_f1[1,1]:+.5f}, {w_f1[1,2]:+.5f}]")
print(f"  Baris 3: [{w_f1[2,0]:+.5f}, {w_f1[2,1]:+.5f}, {w_f1[2,2]:+.5f}]")

# ---- BATCH NORM ----
bn1 = None
found = False
for layer in base_model.layers:
    if layer.name == conv1.name:
        found = True
        continue
    if found and isinstance(layer, tf.keras.layers.BatchNormalization):
        bn1 = layer
        break

if bn1:
    g = bn1.gamma.numpy()[0]
    b = bn1.beta.numpy()[0]
    mu = bn1.moving_mean.numpy()[0]
    var = bn1.moving_variance.numpy()[0]
    eps = 1e-3
    std = np.sqrt(var + eps)
    print(f"\n[BATCH NORM] Layer: {bn1.name}")
    print(f"  gamma (scale)       = {g:+.6f}")
    print(f"  beta (shift)        = {b:+.6f}")
    print(f"  moving_mean         = {mu:+.6f}")
    print(f"  moving_variance     = {var:+.6f}")
    print(f"  sqrt(var + eps)     = {std:+.6f}  (eps={eps})")

# ---- DENSE LAYERS ----
dense128, dense_out = None, None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        if layer.units == 128:
            dense128 = layer
        else:
            dense_out = layer

print(f"\n[DENSE 128] Layer: {dense128.name}")
W128 = dense128.kernel.numpy()
b128 = dense128.bias.numpy()
print(f"  W shape: {W128.shape}")
print(f"  Contoh W[0, 0..4]: {[round(float(x),5) for x in W128[0,:5]]}")
print(f"  Contoh b[0..4]:    {[round(float(x),5) for x in b128[:5]]}")

print(f"\n[DENSE OUTPUT] Layer: {dense_out.name}")
Wout = dense_out.kernel.numpy()
bout = dense_out.bias.numpy()
print(f"  W shape: {Wout.shape}")
print(f"  bias: {[round(float(x),6) for x in bout]}")

# ---- PROSES GAMBAR SAMPEL ----
# Cari gambar sampel dari tiap kelas
class_file = os.path.join(BASE_DIR, 'models', 'class_indices.json')
with open(class_file) as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
print(f"\n[KELAS] {idx_to_class}")

# Pilih 1 gambar dari test/trotol (atau kelas apapun yang ada)
sample_path = None
test_dir = os.path.join(BASE_DIR, 'dataset_split', 'test')
for cname in ['trotol', 'moler', 'sehat', 'non_bawang']:
    cdir = os.path.join(test_dir, cname)
    if os.path.exists(cdir):
        for f in os.listdir(cdir):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                sample_path = os.path.join(cdir, f)
                print(f"\n[GAMBAR SAMPEL] Kelas asli: {cname}")
                print(f"  Path: {sample_path}")
                break
    if sample_path:
        break

img = Image.open(sample_path).convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Ambil patch 3x3 di tengah
cx, cy = 112, 112
patch_rgb = img_array[cy-1:cy+2, cx-1:cx+2, :]  # (3,3,3)
patch_mean = np.mean(patch_rgb, axis=2)
print(f"\n[NORMALISASI] Patch 3x3 piksel (rata-rata RGB) di tengah gambar ({cx},{cy}):")
for i in range(3):
    row = [f"{patch_mean[i,j]:.4f}" for j in range(3)]
    print(f"  Baris {i+1}: [{', '.join(row)}]")
print(f"  Patch piksel RGB asli (sebelum normalisasi):")
for i in range(3):
    row_r = [f"{int(img_array[cy-1+i, cx-1+j, 0]*255)}" for j in range(3)]
    row_g = [f"{int(img_array[cy-1+i, cx-1+j, 1]*255)}" for j in range(3)]
    row_b = [f"{int(img_array[cy-1+i, cx-1+j, 2]*255)}" for j in range(3)]
    print(f"  Baris {i+1}: R=[{','.join(row_r)}] G=[{','.join(row_g)}] B=[{','.join(row_b)}]")

# Dot product Conv1 Filter #1
conv_result = np.sum(patch_mean * w_f1)
print(f"\n[KONVOLUSI] Dot product Filter #1 pada patch:")
for i in range(3):
    for j in range(3):
        print(f"  ({i+1},{j+1}): {patch_mean[i,j]:.4f} x {w_f1[i,j]:+.5f} = {patch_mean[i,j]*w_f1[i,j]:+.6f}")
print(f"  Jumlah semua 9 sel = {conv_result:+.6f}")

# BatchNorm manual
x_hat = (conv_result - mu) / std
y_bn = g * x_hat + b
relu_out = max(0.0, y_bn)
print(f"\n[BATCH NORM] Kalkulasi:")
print(f"  y_conv = {conv_result:+.6f}")
print(f"  hat_x  = ({conv_result:+.6f} - {mu:+.6f}) / {std:.6f} = {x_hat:+.6f}")
print(f"  y_BN   = ({g:+.6f} x {x_hat:+.6f}) + {b:+.6f} = {y_bn:+.6f}")
print(f"\n[RELU] f({y_bn:+.6f}) = max(0, {y_bn:+.6f}) = {relu_out:+.6f}")

# Buat submodel untuk mendapatkan LOGIT 4 KELAS (output sebelum softmax)
logit_model = tf.keras.Model(inputs=model.input, outputs=dense_out.output)
logits_4 = logit_model.predict(img_batch, verbose=0)[0]

print(f"\n[LOGIT 4 KELAS] Nilai z_k sebelum Softmax:")
for i, z in enumerate(logits_4):
    cname = idx_to_class.get(i, f'Kelas{i}')
    print(f"  z_{i+1} ({cname}) = {z:+.6f}")

exps = np.exp(logits_4)
sum_exp = np.sum(exps)
probs = exps / sum_exp
print(f"\n[SOFTMAX] Nilai eksponensial e^z_k:")
for i, (z, e) in enumerate(zip(logits_4, exps)):
    cname = idx_to_class.get(i, f'Kelas{i}')
    print(f"  e^({z:+.6f}) = {e:.6f}  [{cname}]")
print(f"  Sigma e^z = {sum_exp:.6f}")

print(f"\n[PROBABILITAS] P(y_k) = e^z_k / Sigma_e x 100%:")
for i, (e, p) in enumerate(zip(exps, probs)):
    cname = idx_to_class.get(i, f'Kelas{i}')
    print(f"  P({cname}) = {e:.6f} / {sum_exp:.6f} x 100% = {p*100:.5f}%")

pred_idx = np.argmax(probs)
print(f"\n[PREDIKSI] Kelas: {idx_to_class.get(pred_idx)} ({probs[pred_idx]*100:.2f}%)")
print("\n" + "=" * 60)
