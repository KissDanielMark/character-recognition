import numpy as np
from sklearn.model_selection import train_test_split

class ConvolutionalNeuralNetwork:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def split(self):
        X, y = np.arange(10).reshape((5, 2)), range(5)
        print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        print(X_train)
        return
    