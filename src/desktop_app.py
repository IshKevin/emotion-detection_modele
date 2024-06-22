# src/desktop_app.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

model_path = '../models/emotion_detection_model.h5'
try:
    emotion_model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit("Could not load the model. Exiting...")

def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to the model's input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
    return img_array

# Function to predict emotion
def predict_emotion(img):
    preprocessed_img = preprocess_image(img)
    predictions = emotion_model.predict(preprocessed_img)
    predicted_class = np.argmax(predictions)
    
    class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    confidence = predictions[0][predicted_class] * 100  # Confidence percentage
    return class_labels[predicted_class], confidence

# Function to handle file selection
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.convert('RGB')  # Ensure image is in RGB mode
            
            # Display the image in the GUI
            img_display = img.copy()
            img_display.thumbnail((400, 400))  # Resize image for display
            img_display = ImageTk.PhotoImage(img_display)
            panel.configure(image=img_display)
            panel.image = img_display
            
            # Preprocess and predict emotion
            emotion, confidence = predict_emotion(img)
            messagebox.showinfo("Emotion Prediction", f"Predicted Emotion: {emotion}\nConfidence: {confidence:.2f}%")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

# Create GUI window
window = tk.Tk()
window.title("Emotion Detection")

# Create GUI components
btn_open = tk.Button(window, text="Open Image", command=open_image)
btn_open.pack(pady=20)

panel = tk.Label(window)
panel.pack(padx=20, pady=20)

# Run the GUI main loop
window.mainloop()
