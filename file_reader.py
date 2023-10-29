import os
from PIL import Image
import numpy as np
from labeler import Labeler

class FileReader:
    """Class for reading the train set"""
    def __init__(self):
        self.subdirectories = self._read_subdirectories(os.getcwd())
        self.files = []
        self.labels= Labeler()
        self.train_set_imgs = []
        self.train_set_labels = []

    def _read_subdirectories(self, utvonal):
        return [f.path for f in os.scandir(utvonal) if f.is_dir()]

    def show(self):
        # Print the list of subdirectories
        '''for subdir in self.subdirectories:
            print(subdir)
        
        for file in self.files:
            #print(file)'''
        #print(len(self.files))
        print(len(self.train_set_labels))
        #print(self.train_set[0])
        return
    
    def _list_files_recursively(self, directory):
        """Recursively reading and then adding the read files to an arary"""
        current_file_list = []
        for root, directories, files in os.walk(directory):
            for filename in files:
                current_file_list.append(os.path.join(root, filename))
        self.files.extend(current_file_list)


    def read_all_files(self):
        print("Reading files...")
        """Reads all available files from the directory only from Train1 and Train2 called directories"""
        for subdir in self.subdirectories:
            if ('Train2' in subdir) or ('Train1' in subdir):
                self._list_files_recursively(subdir)
        print("File reading finished."+str(len(self.files)))
        return

    def create_train_set(self):
        """At first the images are read and then converted to Array format"""
        print("Preparing train set with transformations...")
        for image in self.files:
            img, lbl = self._process_image(image)
            self.train_set_imgs.append(img)
            self.train_set_labels.append(lbl)
        print("Preparation finished.")

    def _process_image(self, image_path):
        """Convertig img to numpy array and normalazing it with divison"""
        img = Image.open(image_path)
        filename = img.filename[-16:]
        #print(filename[:6])
        label = self.labels.values[filename[:6]]
        #print("\tlabel for IMG:", label)
        img_array = np.array(img)
        normlaized_img_array  = img_array / 255.0
        #ready_img = (normlaized_img_array, label)
        return normlaized_img_array, label