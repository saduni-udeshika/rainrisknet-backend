import tensorflow as tf

# Define data directories
train_dir = "images/training"
validation_dir = "images/validation"

# Define parameters
batch_size = 32
img_height = 224
img_width = 224
epochs = 10

# Prepare the training and validation datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  train_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  validation_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save the trained model
model.save('building_damage_model.h5')
