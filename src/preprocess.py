# src/preprocess.py

from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir, img_size=(48, 48), batch_size=32):
    # ImageDataGenerator for data augmentation and rescaling
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Loading training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    # Loading testing data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    return train_generator, test_generator

if __name__ == "__main__":
    train_dir = '../data/train/train'
    test_dir = '../data/train/test'
    train_gen, test_gen = load_data(train_dir, test_dir)
    print(f'Classes: {train_gen.class_indices}')
