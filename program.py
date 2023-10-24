from file_reader import FileReader
from cnn import ConvolutionalNeuralNetwork

def program():
    """Core program that shoul be run"""
    reader = FileReader()
    reader.read_all_files()
    reader.create_train_set()
    reader.show()
    cnn = ConvolutionalNeuralNetwork(reader.train_set)
    cnn.split()

    #reader._image_to_array('/Users/kissdanielmark/Documents/01.Iskola/MSc/2/MediaAndTextmining/character-recognition/Train1/Sample001/img001-00001.png')    
program()
