from file_reader import FileReader
from cnn import ConvolutionalNeuralNetwork


def export_to_txt(reader, cnn):
    """Export the output to a text file"""
    output_lines = ["class;TestImage"]
    tuple_array = []
    for test_image, test_name in zip(reader.test_set_imgs, reader.test_set_imgname):
        class_label = cnn.predict(test_image)
        tuple_array.append((class_label, test_name))

    sorted_proba = sorted(
        tuple_array, key=lambda x: int(x[1].split("Test")[1].split(".png")[0])
    )

    for class_label, test_name in sorted_proba:
        output_lines.append(f"{class_label};{test_name}")

    output_file_path = "output.txt"
    # Save the output to a text file
    with open(output_file_path, "w") as file:
        for line in output_lines:
            file.write(line + "\n")

    print(f"Output saved to: {output_file_path}")


def program():
    """Core program that shoul be run"""
    reader = FileReader()
    reader.read_training_files()
    reader.create_train_set()

    # reader.show()
    cnn = ConvolutionalNeuralNetwork(reader.train_set_imgs, reader.train_set_labels)
    cnn.split()
    cnn.compile()
    cnn.train()
    cnn.evaluate()

    # reader._image_to_array('/Users/kissdanielmark/Documents/01.Iskola/MSc/2/MediaAndTextmining/character-recognition/Train1/Sample001/img001-00001.png')
    reader.read_test_files()
    reader.create_test_set()
    export_to_txt(reader, cnn)


program()


def ordering():
    proba = [("11", "Test4751.png"), ("1", "Test8751.png"), ("169", "Test751.png")]
    sorted_proba = sorted(
        proba, key=lambda x: int(x[1].split("Test")[1].split(".png")[0])
    )
    print(sorted_proba)


# ordering()
