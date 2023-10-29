import os
import tensorflow as tf
from flask import jsonify

class BuildingDamageClassifier:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ["Not Damaged", "Damaged"]

    def classify_image(self, image_path):
        img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        result = {
            "prediction": self.class_names[1 if predictions[0][0] > 0.5 else 0],
            "confidence": float(predictions[0][0])
        }
        return result
