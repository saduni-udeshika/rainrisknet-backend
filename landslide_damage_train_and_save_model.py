import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define input image size
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Define your image and mask directories for landslide
image_dir = 'Landslide Data/images'
mask_dir = 'Landslide Data/masks'

# List image and mask files
all_images = os.listdir(image_dir)
all_masks = os.listdir(mask_dir)

# Load and preprocess your data
images = []
masks = []

for img_name, mask_name in zip(all_images, all_masks):
    img_path = os.path.join(image_dir, img_name)
    mask_path = os.path.join(mask_dir, mask_name)

    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    images.append(np.array(input_arr))

    mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=(224, 224))
    input_mask = tf.keras.preprocessing.image.img_to_array(mask)
    masks.append(np.array(input_mask))

images = np.array(images)
masks = np.array(masks)

X_train, X_valid, y_train, y_valid = train_test_split(images, masks, test_size=0.15, random_state=42)

# Define the number of output channels for your model
OUTPUT_CHANNELS = 3  # Adjust this value according to your needs

def unet():
    inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    encoder_output, convs = encoder(inputs)
    bottle_neck = bottleneck(encoder_output)
    outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)
    return x

def encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3):
    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p

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
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)
    return c

def decoder(inputs, convs, output_channels):
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')(c9)
    return outputs

model = unet()

def dice_coefficient(y_true, y_pred):
    y_true_flatten = tf.keras.layers.Flatten()(y_true)
    y_pred_flatten = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_flatten * y_pred_flatten)
    return (2.0 * intersection + 1) / (tf.reduce_sum(y_true_flatten) + tf.reduce_sum(y_pred_flatten) + 1)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=dice_coefficient,
              metrics=[dice_coefficient])

# Use data augmentation for training
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=8),
          validation_data=(X_valid, y_valid),
          epochs=5)

# Save the trained model
model.save('landslide_damage_model.h5')
