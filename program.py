from file_reader import FileReader

def program():
    print("Semmiti")
    reader = FileReader()
    reader.show()
    reader.image_to_array('/Users/kissdanielmark/Documents/01.Iskola/MSc/2/MediaAndTextmining/character-recognition/Train1/Sample001/img001-00001.png')

program()