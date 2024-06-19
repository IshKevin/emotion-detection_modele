import tensorflow as tf

def main():
    
    model_path = '../models/emotion_detection_model.h5'
    emotion_model = tf.keras.models.load_model(model_path)

    
    converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
    tflite_model = converter.convert()

    
    tflite_model_path = '../models/emotion_detection_model.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f'TensorFlow Lite model saved to {tflite_model_path}')

if __name__ == "__main__":
    main()
