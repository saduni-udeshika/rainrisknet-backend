import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load your trained model
model = tf.keras.models.load_model('flood_damage_model.h5')

# Define your image preprocessing function
def preprocess_image(image_path, target_size):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    return input_arr

# Get user inputs
#user_image_path = input("Enter the path or filename of the flood image: ")
#date = input("Enter the date of the flood: ")
#location = input("Enter the location of the flood: ")
#disaster_type = input("Enter the type of disaster: ")

# Preprocess the user-provided image
#input_arr = preprocess_image(user_image_path, target_size=(224, 224))

# Predict the mask for the user-provided image
#user_prediction = model.predict(np.array([input_arr]))

# Define a threshold value
#threshold = 0.35

# Convert the predicted mask to binary
#binary_mask = np.where(user_prediction > threshold, 1, 0)

# Calculate the percentage of flood damage
#percentage_damage = (np.sum(binary_mask) / np.prod(binary_mask.shape)) * 100

#formatted_percentage = "{:.2f}".format(percentage_damage)
#print(f"Percentage of Flood Damage: {formatted_percentage}%")

# Display the user-provided image and its predicted damage mask
#plt.figure(figsize=(10, 8))
#plt.subplot(1, 2, 1)
#plt.imshow(input_arr)
#plt.title("User Provided Image")

#plt.subplot(1, 2, 2)
#plt.imshow(binary_mask[0] * 255, cmap='gray')
#plt.title("Predicted Damage Mask")

# Display the additional information
#plt.suptitle(f"Date: {date}\nLocation: {location}\nDisaster Type: {disaster_type}", fontsize=12, y=1.02)

#plt.show()