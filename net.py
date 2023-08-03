import os
import tensorflow as tf

CURRENT_DIR = os.getcwd()

TRAINING_DATA_DIR = os.path.join(CURRENT_DIR, 'images/training')
VALID_DATA_DIR = os.path.join(CURRENT_DIR, 'images/validation')
# weights represent the numerical values assigned to each connection between neurons in the network
WEIGHTS_SAVE_DIR = os.path.join(CURRENT_DIR, 'weights')
WEIGHTS_SAVE_FILE = WEIGHTS_SAVE_DIR + '/data'

# number of training images
batch_size = 32
img_height = 180
img_width = 180
epochs = 25


class Net:

    def __init__(self):
        train_ds = self.generate_dataset(TRAINING_DATA_DIR, "training")
        val_ds = self.generate_dataset(VALID_DATA_DIR, "validation")
        class_names = train_ds.class_names
        self.class_names = class_names
        # automatically determine the optimal number of parallel calls during data preprocessing
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        num_classes = len(class_names)
        model = self.build_model(num_classes)
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=['accuracy'])
        if self.is_weights_exists():
            model.load_weights(WEIGHTS_SAVE_FILE)
        else:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs
            )
            model.save_weights(WEIGHTS_SAVE_FILE)
        self.model = model

    def is_weights_exists(self):
        return len(os.listdir(WEIGHTS_SAVE_DIR)) != 0

    def generate_dataset(self, path, subset):
        return tf.keras.utils.image_dataset_from_directory(
            path,
            validation_split=0.2,
            subset=subset,
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size)

    # defining and constructing the model architecture
    def build_model(self, num_classes):
        return tf.keras.Sequential([
            tf.keras.layers.Rescaling(
                1. / 255, input_shape=(img_height, img_width, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(num_classes)
        ])

    def predict(self, img_path):
        img = tf.keras.utils.load_img(
            img_path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.model.predict(img_array)
        if predictions[0][0] > predictions[0][1]:
            return self.class_names[0]
        else:
            return self.class_names[1]
