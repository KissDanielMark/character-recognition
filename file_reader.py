import os
from PIL import Image
import numpy as np

class FileReader:
    def __init__(self):
        self.subdirectories = self._read_subdirectories(os.getcwd())
        self.files = []

    def _read_subdirectories(self, utvonal):
        return [f.path for f in os.scandir(utvonal) if f.is_dir()]

    def show(self):
        # Print the list of subdirectories
        '''for subdir in self.subdirectories:
            print(subdir)
        
        for file in self.files:
            #print(file)'''
        print(len(self.files))
    
    def _list_files_recursively(self, directory):
        file_list = []

        for root, directories, files in os.walk(directory):
            for filename in files:
                file_list.append(os.path.join(root, filename))
        self.files.extend(file_list)

    def read_all_files(self):
        for subdir in self.subdirectories:
            #print(subdir)
            if ('Train2' in subdir) or ('Train1' in subdir):
                self._list_files_recursively(subdir)

    def convert_all_to_array(self):
        for image in self.files:
            self._image_to_array(image)

    def _image_to_array(self, image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        #print(f"Shape of the image array: {img_array.shape}")
        #print(img_array)
        normlaized_img_array  = img_array / 255.0
        #print(normlaized_img_array)
        return img_array