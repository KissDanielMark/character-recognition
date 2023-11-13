import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ConvolutionalNeuralNetwork:
    """Class for doing the character recognition"""

    def __init__(self, input_images, input_labels):
        self.images = input_images
        self.labels = input_labels

        self.img_train = np.empty([1, 1])
        self.img_test = np.empty([1, 1])
        self.label_train = np.empty([1, 1])
        self.label_test = np.empty([1, 1])

        self.img_val = np.empty([1, 1])
        self.label_val = np.empty([1, 1])

        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
            ]
        )
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation="relu"))
        self.model.add(
            layers.Dense(62, activation="softmax")
        )  # 10 classes for digits 0-9
        return

    def split(self):
        """Splitng the dataset to train and test set"""
        kepek = np.array(self.images)
        tagek = np.array(self.labels)

        # Add channel dimension
        kepek = np.expand_dims(kepek, axis=-1)

        (
            self.img_train,
            self.img_val,
            self.label_train,
            self.label_val,
        ) = train_test_split(kepek, tagek, test_size=0.2, random_state=42)

        print("Split completed.")
        return

    def compile(self):
        """Compiling the model"""
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return

    def train(self):
        """Training the model"""
        # Apply data augmentation to the training set

        print("Train started...")

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

        datagen.fit(self.img_train)

        self.model.fit(
            datagen.flow(self.img_train, self.label_train, batch_size=64),
            epochs=10,
            validation_data=(self.img_val, self.label_val),
        )
        print("Train finished...")
        return

    def evaluate(self):
        """Evaluate model on test set"""
        print("Evaluation...")
        test_loss, test_acc = self.model.evaluate(self.img_test, self.label_test)
        print(f"Test accuracy: {test_acc}")

    def predict(self, img):
        """Predicting the label of the image"""
        img = img.reshape((1,) + img.shape)
        prediction = self.model.predict(img)
        return np.argmax(prediction)
