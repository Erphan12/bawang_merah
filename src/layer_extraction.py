import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image

FILTER_LABELS = [
    "Filter #01 — ~tepi diagonal",
    "Filter #02 — ~perataan (blur)",
    "Filter #03 — ~tepi vertikal",
    "Filter #04 — ~bentuk/tepi silang",
    "Filter #05 — ~deteksi warna hijau",
    "Filter #06 — ~kontras kecerahan",
    "Filter #07 — ~tekstur halus",
    "Filter #08 — ~tepi horizontal",
    "Filter #09 — ~gradien diagonal 2",
    "Filter #10 — ~penajaman detail",
]

def generate_conv_maps(conv_arr, conv_weights=None, conv_biases=None, max_filters=32):
    maps = []
    if conv_arr is None or len(conv_arr.shape) < 3:
        return maps
    num_filters = min(conv_arr.shape[-1], max_filters)
    
    tints = [
        [100, 110, 190],  # purple/blue (#01)
        [130, 140, 180],  # slate/blue (#02)
        [180, 150, 120],  # sepia/brown (#03)
        [110, 150, 110],  # green (#04)
        [90, 160, 130],   # teal
        [160, 140, 190],  # violet
        [170, 170, 140],  # warm olive
        [120, 140, 160]   # steel blue
    ]
    
    for i in range(num_filters):
        fmap = conv_arr[:, :, i]
        min_v, max_v = float(np.min(fmap)), float(np.max(fmap))
        if max_v > min_v:
            norm = (fmap - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(fmap)
            
        tint = tints[i % len(tints)]
        rgb = np.zeros((fmap.shape[0], fmap.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            rgb[:, :, c] = np.clip(norm * tint[c], 0, 255).astype(np.uint8)
            
        # Resize feature map ke ukuran kecil (max 112x112) untuk optimasi kecepatan encoding PNG base64
        fmap_img = Image.fromarray(rgb)
        if fmap_img.width > 112 or fmap_img.height > 112:
            fmap_img = fmap_img.resize((112, 112), Image.NEAREST)
        buf = io.BytesIO()
        fmap_img.save(buf, format="PNG", optimize=False)
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        label = FILTER_LABELS[i] if i < len(FILTER_LABELS) else f"Filter #{i+1:02d} — ~fitur visual"
        
        kernel = None
        if conv_weights is not None and len(conv_weights.shape) == 4:
            # Gunakan Kanal 0 (Ch0) secara konsisten seragam untuk seluruh 32 filter Conv1
            k3x3 = conv_weights[:, :, 0, i]
            kernel = [[round(float(v), 3) for v in row] for row in k3x3]

        bias_val = None
        if conv_biases is not None and i < len(conv_biases):
            bias_val = round(float(conv_biases[i]), 4)

        maps.append({
            "label": label,
            "image_b64": b64_str,
            "kernel": kernel,
            "bias": bias_val
        })
        
    return maps


def generate_relu_maps(relu_arr, conv_weights=None, max_filters=32):
    maps = []
    if relu_arr is None or len(relu_arr.shape) < 3:
        return maps
    num_filters = min(relu_arr.shape[-1], max_filters)
    
    for i in range(num_filters):
        fmap = relu_arr[:, :, i]
        max_v = float(np.max(fmap))
        
        # Color mapping for ReLU (mint green activated area, light gray background for 0)
        rgb = np.full((fmap.shape[0], fmap.shape[1], 3), 220, dtype=np.uint8)
        
        mask = fmap > 0
        if np.any(mask) and max_v > 0:
            norm = fmap[mask] / max_v
            rgb[mask, 0] = np.clip(180 - norm * 160, 0, 255).astype(np.uint8)
            rgb[mask, 1] = np.clip(230 - norm * 100, 0, 255).astype(np.uint8)
            rgb[mask, 2] = np.clip(200 - norm * 120, 0, 255).astype(np.uint8)
            
        fmap_img = Image.fromarray(rgb)
        if fmap_img.width > 112 or fmap_img.height > 112:
            fmap_img = fmap_img.resize((112, 112), Image.NEAREST)
        buf = io.BytesIO()
        fmap_img.save(buf, format="PNG", optimize=False)
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        label = FILTER_LABELS[i] if i < len(FILTER_LABELS) else f"Filter #{i+1:02d} — ~fitur visual"
        
        kernel = None
        if conv_weights is not None and len(conv_weights.shape) == 4:
            k3x3 = np.mean(conv_weights[:, :, :, i], axis=2)
            kernel = [[round(float(v), 3) for v in row] for row in k3x3]
            
        maps.append({
            "label": label,
            "image_b64": b64_str,
            "kernel": kernel
        })
        
    return maps


def generate_pool_maps(pool_arr, conv_weights=None, max_filters=32):
    maps = []
    if pool_arr is None or len(pool_arr.shape) < 3:
        return maps
    num_filters = min(pool_arr.shape[-1], max_filters)
    
    for i in range(num_filters):
        fmap = pool_arr[:, :, i]
        max_v = float(np.max(fmap))
        
        # Color mapping for Pool (warm gold/amber activated area)
        rgb = np.full((fmap.shape[0], fmap.shape[1], 3), 245, dtype=np.uint8)
        
        mask = fmap > 0
        if np.any(mask) and max_v > 0:
            norm = fmap[mask] / max_v
            rgb[mask, 0] = np.clip(250 - norm * 120, 0, 255).astype(np.uint8)
            rgb[mask, 1] = np.clip(220 - norm * 150, 0, 255).astype(np.uint8)
            rgb[mask, 2] = np.clip(160 - norm * 140, 0, 255).astype(np.uint8)
            
        fmap_img = Image.fromarray(rgb)
        if fmap_img.width > 112 or fmap_img.height > 112:
            fmap_img = fmap_img.resize((112, 112), Image.NEAREST)
        buf = io.BytesIO()
        fmap_img.save(buf, format="PNG", optimize=False)
        b64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        label = FILTER_LABELS[i] if i < len(FILTER_LABELS) else f"Filter #{i+1:02d} — ~fitur visual"
        
        kernel = None
        if conv_weights is not None and len(conv_weights.shape) == 4:
            k3x3 = np.mean(conv_weights[:, :, :, i], axis=2)
            kernel = [[round(float(v), 3) for v in row] for row in k3x3]
            
        maps.append({
            "label": label,
            "image_b64": b64_str,
            "kernel": kernel
        })
        
    return maps


def build_extraction_models(model):
    result = {}

    if model is None:
        return result

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model is None:
        return result

    conv_layer = None
    relu_layer = None
    pool_layer = None

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) and conv_layer is None:
            conv_layer = layer
        if isinstance(layer, tf.keras.layers.ReLU) and relu_layer is None:
            relu_layer = layer
        name_lower = layer.name.lower()
        if 'block_1_project_bn' in name_lower:
            pool_layer = layer

    base_outputs = {'final': base_model.output}
    if conv_layer:
        base_outputs['conv'] = conv_layer.output
    if relu_layer:
        base_outputs['relu'] = relu_layer.output
    if pool_layer:
        base_outputs['pool'] = pool_layer.output

    try:
        result['base'] = tf.keras.Model(inputs=base_model.input, outputs=base_outputs)
    except Exception:
        pass

    if conv_layer:
        try:
            weights = conv_layer.get_weights()
            result['conv_weights'] = weights[0]
            if len(weights) > 1:
                result['conv_biases'] = weights[1]
        except Exception:
            pass

    head_outputs = {'output': model.output}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            head_outputs['gap'] = layer.output
        elif isinstance(layer, tf.keras.layers.Dense):
            if hasattr(layer, 'units') and layer.units == 128:
                head_outputs['dense128'] = layer.output

    try:
        result['head'] = tf.keras.Model(inputs=model.input, outputs=head_outputs)
    except Exception:
        pass

    try:
        last_dense = model.layers[-1]
        if isinstance(last_dense, tf.keras.layers.Dense):
            W, b = last_dense.get_weights()
            result['logit_W'] = W
            result['logit_b'] = b
    except Exception:
        pass

    return result


def extract_layer_data(image_bytes, extraction_models, preprocess_fn, class_labels):
    if not extraction_models:
        return None

    input_tensor = preprocess_fn(image_bytes)
    result = {}

    # ====================================================================================================================================
    # STEP 1: INPUT (PREPROCESSED IMAGE TENSOR)
    # ====================================================================================================================================
    img_arr = input_tensor[0]
    result['input'] = {
        'shape': list(img_arr.shape),
        'min': round(float(np.min(img_arr)), 4),
        'max': round(float(np.max(img_arr)), 4),
        'mean': round(float(np.mean(img_arr)), 4),
        'std': round(float(np.std(img_arr)), 4),
    }

    if 'base' in extraction_models:
        try:
            base_out = extraction_models['base'].predict(input_tensor, verbose=0)
            for key in ['conv', 'relu', 'pool', 'final']:
                if key in base_out:
                    arr = base_out[key][0]
                    stats = {
                        'shape': list(arr.shape),
                        'min': round(float(np.min(arr)), 4),
                        'max': round(float(np.max(arr)), 4),
                        'mean': round(float(np.mean(arr)), 4),
                        'std': round(float(np.std(arr)), 4),
                    }
                    if key == 'conv':
                        # ================================================================================================================
                        # STEP 2: KONVOLUSI (FEATURE MAP EXTRACTION)
                        # ================================================================================================================
                        counts, edges = np.histogram(arr.flatten(), bins=20)
                        stats['histogram'] = {
                            'counts': counts.tolist(),
                            'edges': [round(float(e), 4) for e in edges.tolist()]
                        }
                        conv_w = extraction_models.get('conv_weights')
                        conv_b = extraction_models.get('conv_biases')
                        result['conv_maps'] = generate_conv_maps(arr, conv_w, conv_b)
                    if key == 'relu':
                        # ================================================================================================================
                        # STEP 3: RELU (ACTIVATION THRESHOLDING)
                        # ================================================================================================================
                        total = int(arr.size)
                        active = int(np.sum(arr > 0))
                        stats['active_neurons'] = active
                        stats['total_neurons'] = total
                        stats['active_pct'] = round(active / total * 100, 1)
                        stats['zero_pct'] = round((total - active) / total * 100, 1)
                        conv_w = extraction_models.get('conv_weights')
                        result['relu_maps'] = generate_relu_maps(arr, conv_w)
                    if key == 'pool':
                        # ================================================================================================================
                        # STEP 4: POOLING (SPATIAL DOWNSAMPLING)
                        # ================================================================================================================
                        conv_w = extraction_models.get('conv_weights')
                        result['pool_maps'] = generate_pool_maps(arr, conv_w)
                    result[key] = stats
        except Exception:
            pass

    if 'head' in extraction_models:
        try:
            head_out = extraction_models['head'].predict(input_tensor, verbose=0)
            if 'gap' in head_out:
                # ========================================================================================================================
                # STEP 5: GLOBAL AVERAGE POOLING (FEATURE VECTOR FLATTENING)
                # ========================================================================================================================
                gap_arr = head_out['gap'][0]
                result['gap'] = {
                    'shape': list(gap_arr.shape),
                    'min': round(float(np.min(gap_arr)), 4),
                    'max': round(float(np.max(gap_arr)), 4),
                    'mean': round(float(np.mean(gap_arr)), 4),
                    'std': round(float(np.std(gap_arr)), 4),
                    'sparsity': round(float(np.mean(gap_arr == 0) * 100), 1),
                    'length': int(gap_arr.shape[0]),
                    'values_sample': [round(float(v), 4) for v in gap_arr[:200].tolist()]
                }
            if 'dense128' in head_out:
                # ========================================================================================================================
                # STEP 6: FULLY CONNECTED (DENSE LAYER CLASSIFICATION HEAD)
                # ========================================================================================================================
                dense_arr = head_out['dense128'][0]
                result['dense128'] = {
                    'shape': list(dense_arr.shape),
                    'min': round(float(np.min(dense_arr)), 4),
                    'max': round(float(np.max(dense_arr)), 4),
                    'mean': round(float(np.mean(dense_arr)), 4),
                    'std': round(float(np.std(dense_arr)), 4),
                    'active_neurons': int(np.sum(dense_arr > 0)),
                    'total_neurons': int(dense_arr.shape[0]),
                    'active_pct': round(float(np.mean(dense_arr > 0) * 100), 1),
                    'activations': [round(float(v), 4) for v in dense_arr.tolist()]
                }
        except Exception:
            pass

    if 'logit_W' in extraction_models and 'dense128' in result:
        try:
            # ============================================================================================================================
            # STEP 7: HASIL PREDIKSI & SOFTMAX (OUTPUT LAYER PROBABILITIES)
            # ============================================================================================================================
            dense_out = np.array(result['dense128']['activations']).reshape(1, -1)
            logits = (dense_out @ extraction_models['logit_W'] + extraction_models['logit_b'])[0]
            shifted = logits - np.max(logits)
            exp_vals = np.exp(shifted)
            sum_exp = float(np.sum(exp_vals))
            softmax_probs = exp_vals / sum_exp
            result['logits'] = {
                'values': [round(float(v), 4) for v in logits.tolist()],
                'exp_values': [round(float(v), 6) for v in exp_vals.tolist()],
                'sum_exp': round(sum_exp, 6),
                'softmax_probs': [round(float(v), 6) for v in softmax_probs.tolist()],
                'classes': list(class_labels)
            }
        except Exception:
            pass

    if 'conv_weights' in extraction_models:
        W = extraction_models['conv_weights']
        b = extraction_models.get('conv_biases')
        first_filter_ch0 = W[:, :, 0, 0]
        biases_list = [round(float(v), 4) for v in b.tolist()] if b is not None else []
        result['conv_filter'] = {
            'kernel_shape': list(W.shape),
            'total_filters': int(W.shape[3]),
            'first_filter_ch0': [round(float(v), 4) for v in first_filter_ch0.flatten().tolist()],
            'biases': biases_list
        }

    return result
