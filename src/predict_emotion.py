from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model_path = '../models/emotion_detection_model.h5'  # Adjust this path as necessary
emotion_model = load_model(model_path)

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Function to predict emotion
def predict_emotion(img_path):
    preprocessed_img = preprocess_image(img_path)
    predictions = emotion_model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions)
    
    class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    return class_labels[predicted_class]

# Example usage
if __name__ == "__main__":
    img_path = '../data/train/test/test/why.png'  # Replace with the path to your image
    predicted_emotion = predict_emotion(img_path)
    print(f'Predicted emotion: {predicted_emotion}')
