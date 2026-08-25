import os
import shutil
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_split")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # Total 1.0 (70% train, 15% val, 15% test)


VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def split_dataset(source_dir, output_dir, seed=42):
    random.seed(seed)
    
    # Otomatis hapus/bersihkan folder dataset_split lama jika ada
    if os.path.exists(output_dir):
        print(f"Membersihkan folder split lama: {output_dir}...")
        shutil.rmtree(output_dir)
        print("Folder lama berhasil dibersihkan!")
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(VALID_EXTENSIONS)]
        
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        for split_name, split_images in splits.items():
            split_cls_dir = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            
            for img_name in split_images:
                src_file = os.path.join(cls_path, img_name)
                dst_file = os.path.join(split_cls_dir, img_name)
                shutil.copy2(src_file, dst_file)
                
        print(f"Kelas '{cls}': {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    print("\nPembagian dataset selesai! Folder tersimpan di:", output_dir)

if __name__ == "__main__":
    split_dataset(SOURCE_DIR, OUTPUT_DIR)
