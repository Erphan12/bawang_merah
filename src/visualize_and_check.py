import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_split")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

SPLITS = ['train', 'val', 'test']

def visualize_dataset(dataset_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = dataset_dir
        
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print("=" * 50)
    print("1. MENAMPILKAN SAMPEL GAMBAR PER KELAS")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]
        
    for i, cls in enumerate(classes):
        cls_path = os.path.join(train_dir, cls)
        img_paths = glob.glob(os.path.join(cls_path, "*.[jJ][pP][gG]")) + \
                    glob.glob(os.path.join(cls_path, "*.[pP][nN][gG]")) + \
                    glob.glob(os.path.join(cls_path, "*.[jJ][pP][eE][gG]"))
        
        if img_paths:
            img = Image.open(img_paths[0])
            axes[i].imshow(img)
            axes[i].set_title(f"Kelas: {cls}\nUkuran: {img.size}")
            axes[i].axis('off')
            
    plt.tight_layout()
    sample_img_path = os.path.join(REPORTS_DIR, "sample_images.png")
    plt.savefig(sample_img_path)
    print(f"Sampel gambar disimpan ke: {sample_img_path}")
    plt.close()


    print("\n" + "=" * 50)
    print("2. MENGECEK DISTRIBUSI JUMLAH DATA PER KELAS")
    print("=" * 50)
    
    data_counts = []
    
    for split in SPLITS:
        split_path = os.path.join(dataset_dir, split)
        if not os.path.exists(split_path):
            continue
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            count = len(os.listdir(cls_path)) if os.path.exists(cls_path) else 0
            data_counts.append({'Split': split, 'Class': cls, 'Count': count})
            
    class_totals = {}
    for entry in data_counts:
        cls = entry['Class']
        class_totals[cls] = class_totals.get(cls, 0) + entry['Count']

    print("\nRingkasan Jumlah Data per Kelas:")
    for cls, total in class_totals.items():
        print(f"  - {cls:12s}: {total} gambar")
        
    max_count = max(class_totals.values())
    min_count = min(class_totals.values())
    ratio = min_count / max_count if max_count > 0 else 0
    print(f"\nRasio Keseimbangan Dataset (min/max): {ratio:.2f}")
    if ratio < 0.8:
        print("Status: Dataset kurang seimbang (imbalanced). Sangat disarankan Augmentasi Data!")
    else:
        print("Status: Dataset relatif seimbang (balanced).")

    # Plot Distribution Chart with Matplotlib
    plt.figure(figsize=(9, 5))
    x_indices = range(len(classes))
    width = 0.25
    
    train_counts = [next(item['Count'] for item in data_counts if item['Class'] == c and item['Split'] == 'train') for c in classes]
    val_counts = [next(item['Count'] for item in data_counts if item['Class'] == c and item['Split'] == 'val') for c in classes]
    test_counts = [next(item['Count'] for item in data_counts if item['Class'] == c and item['Split'] == 'test') for c in classes]

    plt.bar([x - width for x in x_indices], train_counts, width=width, label='Train', color='#4C72B0')
    plt.bar(x_indices, val_counts, width=width, label='Val', color='#55A868')
    plt.bar([x + width for x in x_indices], test_counts, width=width, label='Test', color='#C44E52')

    plt.xlabel('Kelas')
    plt.ylabel('Jumlah Gambar')
    plt.title('Distribusi Dataset Bawang Merah (Train / Val / Test)')
    plt.xticks(x_indices, classes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    dist_img_path = os.path.join(REPORTS_DIR, "distribusi_dataset.png")
    plt.savefig(dist_img_path)
    print(f"Grafik distribusi disimpan ke: {dist_img_path}")
    plt.close()

if __name__ == "__main__":
    visualize_dataset(DATASET_DIR)


