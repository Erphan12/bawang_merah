import os
from collections import Counter
from PIL import Image

# Path relatif dari folder src/ ke folder dataset/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def inspect_dataset(dataset_path):
    print("=" * 50)
    print(f"EKSPLORASI DATASET: {dataset_path}")
    print("=" * 50)
    
    if not os.path.exists(dataset_path):
        print(f"Directory {dataset_path} tidak ditemukan!")
        return

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    total_images = 0
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        files = os.listdir(cls_path)
        
        image_files = [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]
        count = len(image_files)
        total_images += count
        
        formats = Counter()
        sizes = Counter()
        corrupt_files = 0
        
        for img_name in image_files:
            img_path = os.path.join(cls_path, img_name)
            try:
                with Image.open(img_path) as img:
                    formats[img.format] += 1
                    sizes[img.size] += 1  # (width, height)
            except Exception:
                corrupt_files += 1
        
        print(f"\nKelas: '{cls}'")
        print(f"  - Jumlah Gambar  : {count}")
        print(f"  - Format File    : {dict(formats)}")
        print(f"  - Variasi Ukuran : {len(sizes)} ukuran berbeda (Contoh terbanyak: {sizes.most_common(3)})")
        if corrupt_files > 0:
            print(f"  - File Corrupt   : {corrupt_files} gambar (PERHATIAN!)")

    print("\n" + "=" * 50)
    print(f"TOTAL SELURUH GAMBAR: {total_images}")
    print("=" * 50)

if __name__ == "__main__":
    inspect_dataset(DATASET_DIR)
