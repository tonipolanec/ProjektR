import numpy as np
import tensorflow as tf

#path = "./resources/original_images"
#train, valid = generate_datasets(path, 1)


def generate_datasets(images_path, batch_size_images):
    img_height = 256
    img_width = 256

    train_dataset, validation_dataset = tf.keras.utils.image_dataset_from_directory(
        images_path,
        labels=None,
        label_mode='int',
        batch_size=batch_size_images,
        image_size=(img_height, img_width),
        seed=42,
        validation_split=0.2,
        subset="both"
    )
    
    return train_dataset, validation_dataset