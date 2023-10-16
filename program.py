from file_reader import FileReader

def program():
    reader = FileReader()
    reader.read_all_files()
    reader.convert_all_to_array()
    reader.show()
    
program()