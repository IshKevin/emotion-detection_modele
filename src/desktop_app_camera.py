# src/desktop_app_camera.py

import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the Keras model
model_path = '../models/emotion_detection_model.h5'
try:
    emotion_model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load Haar cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print(f"Error: Could not load face cascade from {cascade_path}.")
    exit()

# Function to capture frames from the camera
def capture_frame():
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        print("Error: Failed to capture frame from camera.")
        window.after(10, capture_frame)
        return
    
    frame = cv2.flip(frame, 1)  # Flip frame horizontally for natural viewing
    
    # Convert the OpenCV frame to Pillow Image format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    # Display the frame in the GUI window
    img_tk = ImageTk.PhotoImage(image=img_pil)
    panel.img_tk = img_tk  # Keep a reference to prevent garbage collection
    panel.configure(image=img_tk)
    panel.image = img_tk
    
    # Process the frame for face detection and emotion prediction
    process_frame(frame)

    # Schedule the function to run again after 10 ms (for smooth video)
    window.after(10, capture_frame)

# Function to process the frame for face detection and emotion prediction
def process_frame(frame):
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Adjusted parameters for better face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjust for better detection accuracy
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        print("No faces detected.")
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Adding padding to ensure the whole face is captured
        padding = 20
        x_start = max(x - padding, 0)
        y_start = max(y - padding, 0)
        x_end = min(x + w + padding, frame.shape[1])
        y_end = min(y + h + padding, frame.shape[0])
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        
        # Extract the face region and preprocess for emotion prediction
        face_roi = gray[y_start:y_end, x_start:x_end]
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize to model input size
        face_roi = face_roi.astype('float32') / 255.0  # Normalize
        
        # Expand dimensions and predict emotion
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        
        prediction = emotion_model.predict(face_roi)
        print(f"Prediction: {prediction}")  # Debug statement to see prediction output
        predicted_class = np.argmax(prediction)
        confidence = round(prediction[0][predicted_class] * 100, 2)
        
        # Map predicted class index to emotion label
        class_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        predicted_emotion = class_labels.get(predicted_class, 'Unknown')
        
        # Display emotion prediction and confidence on the frame
        cv2.putText(frame, f'{predicted_emotion} ({confidence}%)', (x_start, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Convert processed frame back to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the OpenCV frame to Pillow Image format
    img_pil = Image.fromarray(frame_rgb)
    
    # Display the frame in the GUI window
    img_tk = ImageTk.PhotoImage(image=img_pil)
    panel.img_tk = img_tk  # Keep a reference to prevent garbage collection
    panel.configure(image=img_tk)
    panel.image = img_tk

# Create GUI window
window = tk.Tk()
window.title("Emotion Detection with Camera")

# Create GUI components
panel = tk.Label(window)
panel.pack(padx=20, pady=20)

# Run the camera capture function
capture_frame()

# Function to handle closing the window
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        cap.release()  # Release the camera
        window.destroy()

# Bind closing event to the window
window.protocol("WM_DELETE_WINDOW", on_closing)

# Run the GUI main loop
window.mainloop()
