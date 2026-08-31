# LAMPIRAN KODE PROGRAM (LISTING CODE SKRIPSI)

---

## LAMPIRAN A: PRA-PEMROSESAN DATA & AUGMENTASI CITRA

### 1. Pembagian Dataset Train, Validation, dan Test (`src/split_dataset.py`)
```python
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
```

### 2. Preprocessing & Data Augmentation Generator (`src/preprocessing_augmentation.py`)
```python
import os
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_SPLIT_DIR = os.path.join(BASE_DIR, "dataset_split")
TRAIN_DIR = os.path.join(DATASET_SPLIT_DIR, 'train')
VAL_DIR = os.path.join(DATASET_SPLIT_DIR, 'val')
TEST_DIR = os.path.join(DATASET_SPLIT_DIR, 'test')

def get_data_generators(target_size=IMG_SIZE, batch_size=BATCH_SIZE):
    print("Membuat Data Generator TensorFlow/Keras...")
    
    # Augmentasi + Normalisasi untuk Dataset Training
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # Normalisasi pixel ke [0.0, 1.0]
        rotation_range=20,         # Rotasi acak max 20 derajat
        width_shift_range=0.15,    # Geser horizontal
        height_shift_range=0.15,   # Geser vertikal
        shear_range=0.15,          # Transformasi shear
        zoom_range=0.2,            # Zoom in / zoom out
        brightness_range=[0.8, 1.2],# Variasi kecerahan
        horizontal_flip=True,      # Flip horizontal
        vertical_flip=True,        # Flip vertikal
        fill_mode='nearest'
    )
    
    # Hanya Normalisasi untuk Validation & Test set
    val_test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
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
```

---

## LAMPIRAN B: ARSITEKTUR MODEL CNN & PROSES PELATIHAN (TRAINING)

### 1. Pembangunan Arsitektur CNN & Training Loop (`src/train_model.py`)
```python
import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from preprocessing_augmentation import get_data_generators

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_bawang_model.h5")
PLOT_SAVE_PATH = os.path.join(REPORTS_DIR, "training_history.png")

def build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=4):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

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

def main():
    train_gen, val_gen, test_gen = get_data_generators(target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    actual_num_classes = len(train_gen.class_indices)
    
    model = build_transfer_learning_model(input_shape=(224, 224, 3), num_classes=actual_num_classes)
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

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

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[cb_early_stopping, cb_model_checkpoint, cb_reduce_lr]
    )

if __name__ == "__main__":
    main()
```

---

## LAMPIRAN C: EVALUASI PERFORMA & MATRIKS PENGUJIAN

### 1. Evaluasi Classification Report & Confusion Matrix (`src/evaluate_model.py`)
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing_augmentation import get_data_generators

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_bawang_model.h5")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "reports", "confusion_matrix.png")

def evaluate_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    _, _, test_gen = get_data_generators(target_size=(224, 224), batch_size=32)
    class_labels = list(test_gen.class_indices.keys())

    test_gen.reset()
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    print("\n--- CLASSIFICATION REPORT ---")
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Bawang Merah CNN')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

if __name__ == "__main__":
    evaluate_model()
```

---

## LAMPIRAN D: SERVER REST API & INFERENSI INTI

### 1. Server FastAPI Inferensi CNN (`src/main_api.py`)
```python
import os
import io
import json
import uuid
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API Klasifikasi Penyakit Bawang Merah (CNN)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_bawang_model.h5")
TARGET_IMAGE_SIZE = (224, 224)

model = None
class_labels = ["moler", "non_bawang", "sehat", "trotol"]

@app.on_event("startup")
def load_model_and_classes():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(TARGET_IMAGE_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict_direct(file: UploadFile = File(...)):
    contents = await file.read()
    input_tensor = preprocess_image(contents)

    predictions = model.predict(input_tensor)[0]
    predicted_class_idx = int(np.argmax(predictions))
    predicted_label = class_labels[predicted_class_idx]
    confidence_score = float(predictions[predicted_class_idx] * 100)

    return {
        "success": True,
        "filename": file.filename,
        "prediction": predicted_label,
        "confidence": round(confidence_score, 2),
        "probabilities": [
            {"class": label, "pct": round(float(prob * 100), 2)}
            for label, prob in zip(class_labels, predictions)
        ]
    }
```

---

## LAMPIRAN E: KODE ANTARMUKA WEB & VISUALISASI INTERAKTIF

### 1. Pengendali Navigasi & Komunikasi API Frontend (`frontend/static/js/main.js`)
```javascript
const metaApi = document.querySelector('meta[name="api-base"]');
const API_BASE = (metaApi && metaApi.content && !metaApi.content.includes('localhost')) 
  ? metaApi.content 
  : window.location.origin;

let currentPage = 0;
let sessionId = null;
let prediksiData = null;
let layerStats = null;
let fmapData = null;

const pageFiles = [
  'pages/01_input.html',
  'pages/02_conv.html',
  'pages/03_relu.html',
  'pages/04_pooling.html',
  'pages/05_flatten.html',
  'pages/06_fc.html',
  'pages/07_output.html'
];

async function goTo(n) {
  if (n > 0 && !sessionId) return;
  currentPage = n;
  window.scrollTo({ top: 0, behavior: 'smooth' });

  const contentContainer = document.getElementById('app-content');
  const response = await fetch(`${API_BASE}/${pageFiles[n]}`, { cache: 'no-store' });
  contentContainer.innerHTML = await response.text();

  if (n === 0) initInputEvents();
  if (n === 1 && typeof initConv === 'function') initConv();
  if (n === 2 && typeof initRelu === 'function') initRelu();
  if (n === 3 && typeof initPool === 'function') initPool();
  if (n === 4 && typeof initFlat === 'function') initFlat();
  if (n === 5 && typeof initFC === 'function') initFC();
  if (n === 6 && typeof initOutput === 'function') initOutput();
}

async function uploadFile(file) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: form });
  const data = await res.json();
  if (data.success) {
    sessionId = data.session_id;
    await jalankanPrediksi();
  }
}

async function jalankanPrediksi() {
  if (!sessionId) return;
  const res = await fetch(`${API_BASE}/api/predict?session_id=${sessionId}`);
  const data = await res.json();
  if (data.success) prediksiData = data;
}
```

### 2. Logika Visualisasi Matematika Layer CNN (`frontend/static/js/chart.js`)
```javascript
// Render Operasi Matriks Dot Product Conv1 (Patch 3x3 × Filter 3x3)
function initConv() {
  const state = window.CNN_EDUCATIONAL_STATE;
  const demo = fmapData && fmapData.conv_demo;
  const patchData = (demo && demo.patch) ? demo.patch : [0.20, 0.22, 0.26, 0.21, 0.23, 0.27, 0.22, 0.25, 0.28];
  const filterData = (demo && demo.filter) ? demo.filter : [-0.172, -0.287, -0.071, 0.285, 0.387, 0.175, -0.100, 0.005, -0.090];
  
  let sumProd = 0;
  patchData.forEach((x, i) => {
    sumProd += x * (filterData[i] || 0);
  });
  
  const calcRowsEl = document.getElementById('convCalcRows');
  if (calcRowsEl) {
    calcRowsEl.innerHTML = `Patch 3x3 x Filter 3x3 = Output: ${sumProd.toFixed(4)}`;
  }
}

// Render Aktivasi ReLU max(0, x)
function initRelu() {
  const state = window.CNN_EDUCATIONAL_STATE;
  const reluRowsEl = document.getElementById('reluCalcRows');
  if (reluRowsEl) {
    let rowsHtml = '';
    state.conv_outputs_3x3.forEach((convVal, i) => {
      const reluVal = Math.max(0, convVal);
      rowsHtml += `<tr><td>Conv: ${convVal.toFixed(2)}</td><td>ReLU: ${reluVal.toFixed(2)}</td></tr>`;
    });
    reluRowsEl.innerHTML = rowsHtml;
  }
}
```
