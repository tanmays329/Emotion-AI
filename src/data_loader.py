import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_generators(train_dir, val_dir, img_size=224, batch_size=32, backbone='mobilenet'):
    if backbone == 'mobilenet':
        preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    else:
        preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(img_size, img_size),
        color_mode='rgb', batch_size=batch_size, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(img_size, img_size),
        color_mode='rgb', batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen