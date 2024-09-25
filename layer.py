import tensorflow as tf
from keras import layers, models
import keras

# L1 Distance layer class
@keras.saving.register_keras_serializable()
class L1Dist(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        input_embedding = tf.convert_to_tensor(input_embedding)
        validation_embedding = tf.convert_to_tensor(validation_embedding)
        return tf.math.abs(input_embedding - validation_embedding)

# Importing Kivy dependencies
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# UX dependencies
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# Other dependencies
import cv2 as cv
import os
import numpy as np

# Building layout
class CamApp(App):
    def build(self):
        self.web_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text='Verify', on_press=self.verify, size_hint=(1, 0.1))
        self.verification = Label(text='Verification uninitiated', size_hint=(1, 0.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification)

        # Load the model with the custom layer
        self.model = models.load_model('siamesemodel.keras', custom_objects={'L1Dist': L1Dist})

        # Start the webcam feed
        self.capture = cv.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    # Updating the webcam feed
    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        buf = cv.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Preprocessing the image
    def preprocessing(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, [100, 100])
        img = img / 255.0  # Normalize to [0, 1]
        return img

    # Verification process
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.5

        # Save the captured image
        SAVE_PATH = os.path.join('app_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        cv.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        for image in os.listdir(os.path.join('app_data', 'verification_image')):
            input_img = self.preprocessing(os.path.join('app_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocessing(os.path.join('app_data', 'verification_image', image))

            # Expand dimensions to match batch size expected by the model
            input_img = np.expand_dims(input_img, axis=0)  # Shape: (1, 100, 100, 3)
            validation_img = np.expand_dims(validation_img, axis=0)  # Shape: (1, 100, 100, 3)

            # Stack input and validation images
            inputs = [input_img, validation_img]

            # Ensure that inputs are converted to the same tensor types (float32)
            inputs = [tf.convert_to_tensor(i, dtype=tf.float32) for i in inputs]

            # Make Predictions
            result = self.model([inputs[0], inputs[1]], training=False)  # Predicting with the model
            results.append(result.numpy()[0][0])  # Extract result and append to results

        # Detection Threshold: Number of positive predictions
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions
        verification = detection / len(os.listdir(os.path.join('app_data', 'verification_image')))
        verified = verification > verification_threshold

        # Update the label text based on verification result
        self.verification.text = 'Verified' if verified else 'Unverified'
        print(f"Results: {results}, Verified: {verified}")

        return results, verified

# Run the app
if __name__ == '__main__':
    CamApp().run()
