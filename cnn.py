import numpy as np
from sklearn.model_selection import train_test_split

class ConvolutionalNeuralNetwork:
    """Class for doing the character recognition"""

    def __init__(self, input_images, input_labels):
        self.images = input_images
        self.labels = input_labels
        return
    
    def split(self):
        """Splitng the dataset to train and test set"""
        img_train, img_test, label_train, label_test = train_test_split(self.images, self.labels, test_size=0.33, random_state=42)
        print(len(img_train), len(label_train))
        print(len(img_test), len(label_test))
        return
    