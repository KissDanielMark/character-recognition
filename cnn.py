import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

class ConvolutionalNeuralNetwork:
    """Class for doing the character recognition"""

    def __init__(self, input_images, input_labels):
        self.images = input_images
        self.labels = input_labels

        self.img_train = np.empty([1,1])
        self.img_test = np.empty([1,1])
        self.label_train = np.empty([1,1])
        self.label_test = np.empty([1,1])

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu')
        ])
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(62, activation='softmax'))  # 10 classes for digits 0-9

        return
    
    def split(self):
        """Splitng the dataset to train and test set"""
        kepek = np.array(self.images)
        tagek = np.array(self.labels)
        self.img_train, self.img_test, self.label_train, self.label_test = train_test_split(kepek, tagek, test_size=0.33, random_state=42)
        #print(len(self.img_train), len(self.label_train))
        #print(len(self.img_test), len(self.label_test))
        print("Split completed.")
        return
    
    def compile(self):
        """Compiling the model"""
        self.model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        return
        
    def train(self):
        """Training the model"""
        print("Train started...")
        self.model.fit(self.img_train, self.label_train, epochs=5, batch_size=64, validation_split=0.2)
        print("Train finished...")
        return
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("Evaluation...")
        test_loss, test_acc = self.model.evaluate(self.img_test, self.label_test)
        print(f"Test accuracy: {test_acc}")