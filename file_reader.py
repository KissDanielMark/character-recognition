import os

class FileReader:
    def __init__(self):
        self.subdirectories = [f.path for f in os.scandir(os.getcwd()) if f.is_dir()]
         
    def show(self):
        # Print the list of subdirectories
        for subdir in self.subdirectories:
            print(subdir)