import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'gradcam')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def find_last_conv_layer(model):
    """MobileNetV2 is nested inside our model — dig into it to find the last conv layer."""
    backbone = model.get_layer('mobilenetv2_1.00_224')
    for layer in reversed(backbone.layers):
        if len(layer.output_shape) == 4:  # conv-like layer (has spatial dims)
            return backbone, layer.name
    raise ValueError("No conv layer found")


def make_gradcam_heatmap(img_array, model, backbone_name, pred_index=None):
    backbone = model.get_layer(backbone_name)

    with tf.GradientTape() as tape:
        conv_outputs = backbone(img_array, training=False)
        tape.watch(conv_outputs)

        # Manually replay the head layers (must match model.py architecture order)
        x = model.get_layer('gap')(conv_outputs)
        x = model.get_layer('dense')(x)
        x = model.get_layer('batch_normalization')(x)
        x = model.get_layer('dropout')(x)
        predictions = model.get_layer('dense_1')(x)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index), predictions[0][pred_index].numpy()

def overlay_heatmap(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = (heatmap_color * alpha + original_img * (1 - alpha)).astype('uint8')
    return superimposed


def preprocess_image(img_path, img_size=224):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    original = img_array.copy().astype('uint8')
    preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    preprocessed = np.expand_dims(preprocessed, axis=0)
    return preprocessed, original


def run_gradcam_on_image(img_path, model, backbone_name, save_name):
    preprocessed, original = preprocess_image(img_path)
    heatmap, pred_idx, confidence = make_gradcam_heatmap(preprocessed, model, backbone_name)
    overlay = overlay_heatmap(original, heatmap)
    ...

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Pred: {CLASS_NAMES[pred_idx]} ({confidence:.2f})')
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'emotion_model_final.h5'))
    backbone_name = 'mobilenetv2_1.00_224'

    test_dir = os.path.expanduser('~/emotion-ai-data/data/raw/test')
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(test_dir, cls)
        sample_files = os.listdir(cls_dir)
        if not sample_files:
            continue
        sample_path = os.path.join(cls_dir, sample_files[0])
        run_gradcam_on_image(
            sample_path, model, backbone_name,
            save_name=f'gradcam_{cls}.png'
        )