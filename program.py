from file_reader import FileReader

def program():
    """Core program that shoul be run"""
    reader = FileReader()
    reader.read_all_files()
    reader.convert_all_to_array()
    reader.show()

    #reader._image_to_array('/Users/kissdanielmark/Documents/01.Iskola/MSc/2/MediaAndTextmining/character-recognition/Train1/Sample001/img001-00001.png')    
program()
