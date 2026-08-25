import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set style Matplotlib agar bersih dan cocok untuk dokumen akademis
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def generate_metrics_per_class_chart():
    """
    1. Membuat Bar Chart Metrik Evaluasi per Kelas (Precision, Recall, F1-Score)
    Berdasarkan data evaluasi aktual pada data test (300 sampel).
    """
    classes = ['Moler', 'Bukan Bawang', 'Sehat', 'Trotol']
    precision = [95.95, 100.00, 92.96, 86.42]
    recall = [94.67, 98.67, 88.00, 93.33]
    f1_score = [95.30, 99.33, 90.41, 89.74]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    rects1 = ax.bar(x - width, precision, width, label='Precision (%)', color='#2b5c8f')
    rects2 = ax.bar(x, recall, width, label='Recall (%)', color='#4682b4')
    rects3 = ax.bar(x + width, f1_score, width, label='F1-Score (%)', color='#6baed6')

    ax.set_ylabel('Persentase (%)', fontsize=11, fontweight='bold')
    ax.set_title('Perbandingan Metrik Evaluasi per Kelas pada Data Test (300 Sampel)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, fontweight='bold')
    ax.set_ylim(70, 105)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Menambahkan label nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=0)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "metrics_per_class.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafik Metrik Evaluasi per Kelas disimpan ke: {save_path}")

def generate_flowchart_diagram():
    """
    2. Membuat Diagram Flowchart Alur Sistem (Gaya Standar Flowchart Hitam Putih)
    """
    fig, ax = plt.subplots(figsize=(7, 9), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)

    # 1. MULAI (Oval)
    ellipse_start = patches.Ellipse((5, 12), 3.2, 1.0, fc='white', ec='black', lw=1.5)
    ax.add_patch(ellipse_start)
    ax.text(5, 12, "MULAI", ha="center", va="center", fontsize=9, fontweight='bold')

    # Panah MULAI -> UNGGAH
    ax.annotate('', xy=(5, 10.7), xytext=(5, 11.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # 2. UNGGAH GAMBAR (Persegi Panjang)
    rect_upload = patches.Rectangle((3.1, 9.5), 3.8, 1.2, fc='white', ec='black', lw=1.5)
    ax.add_patch(rect_upload)
    ax.text(5, 10.1, "UNGGAH GAMBAR\nDAUN BAWANG\nMERAH", ha="center", va="center", fontsize=8, fontweight='bold')

    # Panah UNGGAH -> VALIDASI
    ax.annotate('', xy=(5, 8.3), xytext=(5, 9.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # 3. VALIDASI FORMAT BERKAS (Belah Ketupat / Diamond)
    diamond = patches.Polygon([[5, 8.3], [6.8, 7.3], [5, 6.3], [3.2, 7.3]], fc='white', ec='black', lw=1.5)
    ax.add_patch(diamond)
    ax.text(5, 7.3, "VALIDASI FORMAT\nBERKAS", ha="center", va="center", fontsize=8, fontweight='bold')

    # 4. PESAN ERROR (Persegi Panjang Kanan)
    rect_error = patches.Rectangle((7.0, 6.7), 2.8, 1.2, fc='white', ec='black', lw=1.5)
    ax.add_patch(rect_error)
    ax.text(8.4, 7.3, "PESAN ERROR DAN COBA\nUPLOAD KEMBALI", ha="center", va="center", fontsize=7.5, fontweight='bold')

    # Panah TIDAK (Validasi -> Error)
    ax.annotate('', xy=(7.0, 7.3), xytext=(6.8, 7.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(6.9, 7.5, "TIDAK", ha="left", va="center", fontsize=8, fontweight='bold')

    # Panah Kembali (Error -> Unggah Gambar)
    ax.annotate('', xy=(5.0, 10.7), xytext=(8.4, 7.9),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="black", connectionstyle="bar,fraction=-0.3"))

    # Panah YA (Validasi -> Preprocessing)
    ax.annotate('', xy=(5, 5.1), xytext=(5, 6.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))
    ax.text(4.8, 5.7, "YA", ha="right", va="center", fontsize=8, fontweight='bold')

    # 5. PRA PEMROSESAN CITRA (Persegi Panjang)
    rect_prep = patches.Rectangle((2.6, 3.9), 4.8, 1.2, fc='white', ec='black', lw=1.5)
    ax.add_patch(rect_prep)
    ax.text(5, 4.5, "PRA PEMROSESAN CITRA\n(RESIZE DAN RESCALE)", ha="center", va="center", fontsize=8, fontweight='bold')

    # Panah Preprocessing -> Ekstraksi
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # 6. EKSTRAKSI FITUR (Persegi Panjang)
    rect_cnn = patches.Rectangle((2.2, 1.5), 5.6, 1.2, fc='white', ec='black', lw=1.5)
    ax.add_patch(rect_cnn)
    ax.text(5, 2.1, "EKSTRAKSI FITUR DAN INFERENSI CNN\nMOBILENETV2 + GAP + DENSE LAYER", ha="center", va="center", fontsize=7.5, fontweight='bold')

    # Panah Ekstraksi -> Tampilkan
    ax.annotate('', xy=(5, 0.3), xytext=(5, 1.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # 7. TAMPILKAN DIAGNOSIS (Persegi Panjang Bottom)
    rect_diag = patches.Rectangle((2.0, -0.9), 6.0, 1.2, fc='white', ec='black', lw=1.5)
    ax.add_patch(rect_diag)
    ax.text(5, -0.3, "TAMPILKAN DIAGNOSIS DAN CONFIDENCE\nSCORE SERTA REKOMENDASI\nPENANGANAN", ha="center", va="center", fontsize=7.5, fontweight='bold')

    # Panah Tampilkan -> SELESAI
    ax.annotate('', xy=(5, -2.1), xytext=(5, -0.9), arrowprops=dict(arrowstyle="->", lw=1.5, color="black"))

    # 8. SELESAI (Oval Bottom)
    ellipse_end = patches.Ellipse((5, -2.6), 3.2, 1.0, fc='white', ec='black', lw=1.5)
    ax.add_patch(ellipse_end)
    ax.text(5, -2.6, "SELESAI", ha="center", va="center", fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "flowchart_sistem.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Diagram Flowchart Sistem disimpan ke: {save_path}")


def generate_cnn_architecture_diagram():
    """
    3. Membuat Diagram Arsitektur CNN MobileNetV2 Transfer Learning
    """
    fig, ax = plt.subplots(figsize=(11, 4), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)

    layers_info = [
        ("Input Image\n(224x224x3)", 1.2, "#e9ecef"),
        ("Base Model\nMobileNetV2\n(Frozen Weight)", 3.2, "#ffe3e3"),
        ("Global Average\nPooling (GAP)", 5.4, "#d0ebff"),
        ("Batch Normalization\n& Dropout (0.3)", 7.6, "#fff3bf"),
        ("Dense Layer\n(128 Neuron, ReLU)", 9.8, "#d3f9d8"),
        ("Dropout (0.2)", 11.4, "#fff3bf"),
        ("Output Layer\nSoftmax (4 Kelas)", 13.0, "#ffdeeb")
    ]

    for label, x, color in layers_info:
        bbox_props = dict(boxstyle="round,pad=0.5", fc=color, ec="#495057", lw=1.5)
        ax.text(x, 2.5, label, ha="center", va="center", fontsize=8, fontweight='bold', bbox=bbox_props)

        if x < 13.0:
            ax.annotate('', xy=(x + 0.8, 2.5), xytext=(x + 0.5, 2.5),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="#212529"))

    ax.set_title("Arsitektur Model Transfer Learning (MobileNetV2 Bawang Merah)", fontsize=11, fontweight='bold', y=0.9)

    plt.tight_layout()
    save_path = os.path.join(REPORTS_DIR, "arsitektur_cnn.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[OK] Diagram Arsitektur CNN disimpan ke: {save_path}")


if __name__ == "__main__":
    generate_metrics_per_class_chart()
    generate_flowchart_diagram()
    generate_cnn_architecture_diagram()
