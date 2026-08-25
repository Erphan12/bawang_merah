import os
import tensorflow as tf

# Impor ImageDataGenerator langsung dari tf.keras
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator



# Parameter Preprocessing & Augmentasi
IMG_SIZE = (224, 224)  # Ukuran standar untuk CNN (MobileNetV2, ResNet, EfficientNet, dll)
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
TRAIN_DIR = os.path.join(DATASET_SPLIT_DIR, 'train')
VAL_DIR = os.path.join(DATASET_SPLIT_DIR, 'val')
TEST_DIR = os.path.join(DATASET_SPLIT_DIR, 'test')


def get_data_generators(target_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Membuat ImageDataGenerator dengan Preprocessing & Augmentasi Data (TensorFlow / Keras).
    
    1. Preprocessing:
       - Resizing gambar secara konstan ke target_size (default 224x224)
       - Normalisasi piksel dari range [0, 255] menjadi [0.0, 1.0] (rescale=1./255)
    
    2. Augmentasi Data (hanya diterapkan pada training set):
       - Rotation Range: Rotasi acak hingga 20 derajat
       - Width/Height Shift: Pergeseran posisi horizontal/vertikal hingga 15%
       - Shear Range: Pergeseran sudut (shear transformation)
       - Zoom Range: Perbesaran/perkecilan acak hingga 20%
       - Brightness Range: Variasi pencahayaan acak antara 80% - 120%
       - Horizontal & Vertical Flip: Pembalikan posisi secara acak
       - Fill Mode: Penanganan area piksel kosong akibat rotasi/geser
    """
    print("Membuat Data Generator TensorFlow/Keras...")
    
    # 1. Augmentasi + Normalisasi untuk Dataset Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalisasi pixel ke [0, 1]
        rotation_range=20,         # Rotasi acak max 20 derajat
        width_shift_range=0.15,    # Geser horizontal
        height_shift_range=0.15,   # Geser vertikal
        shear_range=0.15,          # Transformaasi shear
        zoom_range=0.2,            # Zoom in / zoom out
        brightness_range=[0.8, 1.2],# Variasi kecerahan/brightness
        horizontal_flip=True,      # Flip horizontal
        vertical_flip=True,        # Flip vertikal
        fill_mode='nearest'
    )
    
    # 2. Hanya Normalisasi (tanpa augmentasi) untuk Validation & Test set
    val_test_datagen = ImageDataGenerator(
        rescale=1./255             # Normalisasi pixel saja
    )
    
    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

if __name__ == "__main__":
    train_gen, val_gen, test_gen = get_data_generators()
    print("\nData Generator Berhasil Dibuat!")
    print(f"Kelas terdeteksi: {train_gen.class_indices}")
    
    # Cek 1 Batch Data Preprocessing
    images, labels = next(train_gen)
    print(f"Bentuk Batch Gambar Train : {images.shape} (Batch, Height, Width, Channels)")
    print(f"Bentuk Batch Label Train  : {labels.shape}")
    print(f"Rentang Piksel setelah Normalisasi: Min = {images.min():.2f}, Max = {images.max():.2f}")
