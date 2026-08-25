import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing_augmentation import get_data_generators

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODELS_DIR, "best_bawang_model.h5")
CONFUSION_MATRIX_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.png")
MISCLASSIFIED_PLOT_PATH = os.path.join(REPORTS_DIR, "misclassified_samples.png")


def evaluate_model():
    print("=" * 60)
    print("1. MEMUAT MODEL DAN DATASET TEST")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"File model tidak ditemukan di {MODEL_PATH}. Pastikan sudah menjalankan train_model.py!")
        return

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model berhasil dimuat dari: {MODEL_PATH}")

    # Load Data Generators (Test Generator)
    _, _, test_gen = get_data_generators(target_size=(224, 224), batch_size=32)
    
    class_labels = list(test_gen.class_indices.keys())
    print(f"Kelas terdeteksi: {class_labels}")

    print("\n" + "=" * 60)
    print("2. MELAKUKAN PREDIKSI PADA DATA TEST")
    print("=" * 60)
    
    # Reset generator agar urutan data sesuai
    test_gen.reset()
    
    # Prediksi seluruh data test
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    print("\n" + "=" * 60)
    print("3. METRIK EVALUASI (Accuracy, Precision, Recall, F1-Score)")
    print("=" * 60)
    
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)

    print("\n" + "=" * 60)
    print("4. VISUALISASI CONFUSION MATRIX")
    print("=" * 60)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Bawang Merah CNN')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    # Menampilkan angka di dalam grid confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual Label (Kenyataan)')
    plt.xlabel('Predicted Label (Prediksi Model)')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix disimpan ke: {CONFUSION_MATRIX_PATH}")
    plt.close()

    print("\n" + "=" * 60)
    print("5. ANALISIS ERROR (Gambar yang Salah Diklasifikasikan)")
    print("=" * 60)
    
    # Mencari indeks gambar yang salah diklasifikasikan
    misclassified_idx = np.where(y_pred != y_true)[0]
    print(f"Total gambar yang salah diklasifikasikan: {len(misclassified_idx)} dari {len(y_true)} data test")

    if len(misclassified_idx) > 0:
        # Tampilkan maksimal 9 contoh gambar misklasifikasi
        num_display = min(9, len(misclassified_idx))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()

        filepaths = test_gen.filepaths

        for idx, mis_i in enumerate(misclassified_idx[:num_display]):
            img_path = filepaths[mis_i]
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            
            true_label = class_labels[y_true[mis_i]]
            pred_label = class_labels[y_pred[mis_i]]
            confidence = y_pred_probs[mis_i][y_pred[mis_i]] * 100

            axes[idx].imshow(img)
            axes[idx].set_title(f"Asli: {true_label}\nPrediksi: {pred_label} ({confidence:.1f}%)", color='red')
            axes[idx].axis('off')

        # Sembunyikan subplot sisa jika kurang dari 9
        for i in range(num_display, 9):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(MISCLASSIFIED_PLOT_PATH)
        print(f"Sampel kesalahan prediksi disimpan ke: {MISCLASSIFIED_PLOT_PATH}")
        plt.close()
    else:
        print("Sempurna! Tidak ada kesalahan klasifikasi pada data test.")

if __name__ == "__main__":
    evaluate_model()
