import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re

# Define input image size
IMG_HEIGHT, IMG_WIDTH = 224, 224
OUTPUT_CHANNELS = 1  # Since it's a binary prediction

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor
    for _ in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
    return x

def encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3):
    conv_output = conv2d_block(inputs, n_filters=n_filters)
    pool_output = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(conv_output)
    dropout_output = tf.keras.layers.Dropout(dropout)(pool_output)
    return conv_output, dropout_output

def encoder(inputs):
    f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2, 2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2, 2), dropout=0.3)
    f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2, 2), dropout=0.3)
    return p4, (f1, f2, f3, f4)

def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)
    return bottle_neck

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    upsampled_output = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    concatenated_output = tf.keras.layers.concatenate([upsampled_output, conv_output])
    dropout_output = tf.keras.layers.Dropout(dropout)(concatenated_output)
    decoded_output = conv2d_block(dropout_output, n_filters, kernel_size=3)
    return decoded_output

def decoder(inputs, convs, output_channels):
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='sigmoid')(c9)
    return outputs

def unet():
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load the saved landslide model with custom objects
def dice_coefficient(y_true, y_pred):
    y_true_flatten = tf.keras.layers.Flatten()(y_true)
    y_pred_flatten = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_flatten * y_pred_flatten)
    return (2.0 * intersection + 1) / (tf.reduce_sum(y_true_flatten) + tf.reduce_sum(y_pred_flatten) + 1)

custom_objects = {'dice_coefficient': dice_coefficient}
saved_model_path = 'landslide_damage_model.h5'
model = tf.keras.models.load_model(saved_model_path, custom_objects=custom_objects)

# Prompt the user to input the path or filename of the landslide image
#user_image_path = input("Enter the path or filename of the landslide image: ")

# Prompt the user to input additional information
#date = input("Enter the date of the landslide: ")
#location = input("Enter the location of the landslide: ")
#disaster_type = input("Enter the type of disaster: ")

# Load the user-provided image and preprocess it
#user_image = tf.keras.preprocessing.image.load_img(user_image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
#input_arr = tf.keras.preprocessing.image.img_to_array(user_image)
#input_arr = input_arr / 255.0

# Predict the mask for the user-provided image
#user_prediction = model.predict(np.array([input_arr]))

# Define a threshold value
#threshold = 0.35

# Convert the predicted mask to binary
#binary_mask = np.where(user_prediction > threshold, 1, 0)

# Calculate the percentage of landslide damage
#percentage_damage = (np.sum(binary_mask) / np.prod(binary_mask.shape)) * 100

#formatted_percentage = "{:.2f}".format(percentage_damage)
#print(f"Percentage of Landslide Damage: {formatted_percentage}%")

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
