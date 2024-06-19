# src/model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_model(input_shape, num_classes):
    model = Sequential()

    # Convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def train_model(train_gen, test_gen, epochs=10):
    input_shape = (48, 48, 1)  # Grayscale images of size 48x48
    num_classes = len(train_gen.class_indices)

    model = create_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, epochs=epochs, validation_data=test_gen)
    
    return model

if __name__ == "__main__":
    import preprocess
    train_dir = '../data/train/train'
    test_dir = '../data/train/test'
    train_gen, test_gen = preprocess.load_data(train_dir, test_dir)
    model = train_model(train_gen, test_gen)
    model.save('../models/emotion_detection_model.h5')
