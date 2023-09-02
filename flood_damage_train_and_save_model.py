import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define your data paths
images_path = 'Flood Data/images'
masks_path = 'Flood Data/labels'

# List image and mask files
all_images = os.listdir(images_path)
all_masks = os.listdir(masks_path)

# Load and preprocess your data
images = []
masks = []

for img_name, mask_name in zip(all_images, all_masks):
    img_path = os.path.join(images_path, img_name)
    mask_path = os.path.join(masks_path, mask_name)

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

# Define the model architecture
def conv2d_block(input_tensor, n_filters, kernel_size = 3):
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                                   kernel_initializer = 'he_normal', padding = 'same')(x)
        x = tf.keras.layers.Activation('relu')(x)
    return x

def encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3):
    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(f)
    p = tf.keras.layers.Dropout(0.3)(p)
    return f, p


def encoder(inputs):
    f1, p1 = encoder_block(inputs, n_filters=64, pool_size=(2,2), dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=128, pool_size=(2,2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=256, pool_size=(2,2), dropout=0.3)
    f4, p4 = encoder_block(p3, n_filters=512, pool_size=(2,2), dropout=0.3)
    return p4, (f1, f2, f3, f4)

def bottleneck(inputs):
    bottle_neck = conv2d_block(inputs, n_filters=1024)
    return bottle_neck

def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):
    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides = strides, padding = 'same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)
    return c

def decoder(inputs, convs, output_channels):
    f1, f2, f3, f4 = convs
    c6 = decoder_block(inputs, f4, n_filters=512, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c7 = decoder_block(c6, f3, n_filters=256, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=128, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=64, kernel_size=(3,3), strides=(2,2), dropout=0.3)
    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')(c9)
    return outputs

OUTPUT_CHANNELS = 3

def unet():

  # specify the input shape
    inputs = tf.keras.layers.Input(shape=(224, 224,3))

  # feed the inputs to the encoder
    encoder_output, convs = encoder(inputs)

  # feed the encoder output to the bottleneck
    bottle_neck = bottleneck(encoder_output)

  # feed the bottleneck and encoder block outputs to the decoder
  # specify the number of classes via the `output_channels` argument
    outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)

  # create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate the model
model = unet()

# See the resulting model architecture
model.summary()

# Configure the optimizer, loss, and metrics for training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy',
              metrics=['accuracy'])

# Configure the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

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

# Train the model with memory cleanup
for epoch in range(10):  # 10 epochs as an example, adjust as needed
    print(f"Epoch {epoch + 1}/10")
    
    # Clear GPU memory
    tf.keras.backend.clear_session()
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=8),  # Reduced batch size
        validation_data=(X_valid, y_valid),
        epochs=1,  # Train for 1 epoch at a time
        callbacks=[early_stopping]
    )

# Save the trained model
model.save('flood_damage_model.h5')
