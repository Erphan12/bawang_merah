import os
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hitung jumlah riil dari folder dataset_split
classes = ['moler', 'non_bawang', 'sehat', 'trotol']
display_names = ['Moler', 'Non Bawang', 'Sehat', 'Trotol']

train_counts = []
val_counts = []
test_counts = []

for cls in classes:
    t_count = len(os.listdir(os.path.join(DATASET_SPLIT_DIR, 'train', cls)))
    v_count = len(os.listdir(os.path.join(DATASET_SPLIT_DIR, 'val', cls)))
    ts_count = len(os.listdir(os.path.join(DATASET_SPLIT_DIR, 'test', cls)))
    train_counts.append(t_count)
    val_counts.append(v_count)
    test_counts.append(ts_count)

total_per_class = [t + v + ts for t, v, ts in zip(train_counts, val_counts, test_counts)]
total_dataset = sum(total_per_class)

# Setting style akademis
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Create figure with 2 subplots (Bar Chart & Donut Chart)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, gridspec_kw={'width_ratios': [1.3, 1]})

# --- 1. GROUPED BAR CHART ---
x = np.arange(len(classes))
width = 0.26

rects1 = ax1.bar(x - width, train_counts, width, label=f'Train (70%) - Total: {sum(train_counts)}', color='#2b5c8f')
rects2 = ax1.bar(x, val_counts, width, label=f'Val (15%) - Total: {sum(val_counts)}', color='#4682b4')
rects3 = ax1.bar(x + width, test_counts, width, label=f'Test (15%) - Total: {sum(test_counts)}', color='#74c476')

ax1.set_title('Distribusi Sampel per Split (Train, Val, Test)', fontsize=12, fontweight='bold', pad=12)
ax1.set_xlabel('Kelas Target', fontsize=11, fontweight='bold')
ax1.set_ylabel('Jumlah Citra', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(display_names, fontsize=10, fontweight='bold')
ax1.set_ylim(0, max(train_counts) * 1.18)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=9)
ax1.grid(axis='y', linestyle='--', alpha=0.6)

# Anotasi angka di atas bar
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

autolabel(rects1, ax1)
autolabel(rects2, ax1)
autolabel(rects3, ax1)

# --- 2. DONUT / PIE CHART ---
colors = ['#e6550d', '#3182bd', '#31a354', '#756bb1']
explode = (0.03, 0.05, 0.03, 0.03)

wedges, texts, autotexts = ax2.pie(
    total_per_class,
    labels=display_names,
    autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct*total_dataset/100))} citra)',
    startangle=140,
    colors=colors,
    explode=explode,
    pctdistance=0.72,
    wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
)

for autotext in autotexts:
    autotext.set_fontsize(8.5)
    autotext.set_fontweight('bold')
for text in texts:
    text.set_fontsize(10)
    text.set_fontweight('bold')

ax2.set_title(f'Proporsi Total Dataset (Total: {total_dataset} Citra)', fontsize=12, fontweight='bold', pad=12)

plt.suptitle('Analisis & Distribusi Dataset Citra Daun Bawang Merah', fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout()

save_path = os.path.join(REPORTS_DIR, "distribusi_dataset.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"[SUCCESS] Grafik distribusi dataset berhasil diperbarui di: {save_path}")
