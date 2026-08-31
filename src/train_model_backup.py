import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from preprocessing_augmentation import get_data_generators

# Parameter Pelatihan
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 3  # sehat, trotol, moler
LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_bawang_model.h5")
PLOT_SAVE_PATH = os.path.join(REPORTS_DIR, "training_history.png")


def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Membangun model Transfer Learning berbasis MobileNetV2.
    Pre-trained weights: ImageNet.
    """
    # 1. Base Model dengan Pre-trained Weights ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model agar weight dasar tidak rusak di tahap awal
    base_model.trainable = False

    # 2. Arsitektur Head Classification
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name="BawangMerah_MobileNetV2")
    return model

def plot_training_history(history, save_path=PLOT_SAVE_PATH):
    """
    Plot grafik akurasi dan loss pelatihan vs validasi.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='s')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='s')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nGrafik performa pelatihan berhasil disimpan ke: {save_path}")
    plt.close()

def main():
    print("=" * 60)
    print("1. MEMUAT DATASET DENGAN PREPROCESSING & AUGMENTASI")
    print("=" * 60)
    
    train_gen, val_gen, test_gen = get_data_generators(target_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # Simpan mapping class_indices ke JSON
    import json
    class_indices_path = os.path.join(MODELS_DIR, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=4)
    print(f"Mapping class_indices disimpan ke: {class_indices_path}")


    print(f"Sub-folder terdeteksi: {list(train_gen.class_indices.keys())}")
    actual_num_classes = len(train_gen.class_indices)

    
    print("\n" + "=" * 60)
    print("2. MEMBANGUN DAN COMPILE MODEL TRANSFER LEARNING (MobileNetV2)")
    print("=" * 60)
    
    model = build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=actual_num_classes)
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n" + "=" * 60)
    print("3. MENYIAPKAN CALLBACKS (EarlyStopping & ModelCheckpoint)")
    print("=" * 60)
    
    cb_early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    cb_model_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    cb_reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    callbacks_list = [cb_early_stopping, cb_model_checkpoint, cb_reduce_lr]

    print("\n" + "=" * 60)
    print("4. MEMULAI TRAINING MODEL")
    print("=" * 60)
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_list
    )

    print("\n" + "=" * 60)
    print("5. MEMBUAT PLOT AKURASI & LOSS DENGAN MATPLOTLIB")
    print("=" * 60)
    
    plot_training_history(history, save_path=PLOT_SAVE_PATH)
    
    print("\n" + "=" * 60)
    print("6. EVALUASI MODEL PADA TEST SET")
    print("=" * 60)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"Hasil Test Accuracy : {test_acc * 100:.2f}%")
    print(f"Hasil Test Loss     : {test_loss:.4f}")

if __name__ == "__main__":
    main()
