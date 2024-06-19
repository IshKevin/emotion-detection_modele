# src/main.py

import preprocess
import model

def main():
    train_dir = '../data/train/train'
    test_dir = '../data/train/test'

    # Load and preprocess data
    train_gen, test_gen = preprocess.load_data(train_dir, test_dir)

    # Train and save the model
    emotion_model = model.train_model(train_gen, test_gen)
    emotion_model.save('../models/emotion_detection_model.h5')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
