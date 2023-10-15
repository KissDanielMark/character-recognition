import os
from PIL import Image
import numpy as np

class FileReader:
    def __init__(self):
        self.subdirectories = [f.path for f in os.scandir(os.getcwd()) if f.is_dir()]

    def show(self):
        # Print the list of subdirectories
        for subdir in self.subdirectories:
            print(subdir)
    def image_to_array(self, image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        print(f"Shape of the image array: {img_array.shape}")
        print(img_array)
        normlaized_img_array  = img_array / 255.0
        print(normlaized_img_array)
        return img_array