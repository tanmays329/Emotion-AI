import tensorflow as tf
from tensorflow.keras import layers, models

def build_transfer_model(input_shape=(224, 224, 3), num_classes=7, backbone='efficientnet'):
    if backbone == 'efficientnet':
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=input_shape
        )
    elif backbone == 'mobilenet':
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights='imagenet', input_shape=input_shape
        )
    else:
        raise ValueError("backbone must be 'efficientnet' or 'mobilenet'")

    base.trainable = False  # freeze for phase 1

    inputs = layers.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model, base

def unfreeze_top_layers(base_model, num_layers=30):
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False