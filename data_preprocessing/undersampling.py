import os
import random
import shutil

#For testing purposes
# source_dir = "D:\misclassified"
# target_dir = "D:\\undersampling"
# number_to_move = 100

#Targetted folders
source_dir = "D:\images_resized"
target_dir = "D:\\undersampling"
number_to_move = 7317

for class_folder in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_folder)

    # Check if it's a directory (ignores any files in the source directory)
    if os.path.isdir(class_path):
        # Create a test folder for the current class in the test directory
        target_class_dir = os.path.join(target_dir, class_folder)
        os.makedirs(target_class_dir, exist_ok=True)

        # List all files in the class folder
        files = os.listdir(class_path)

        # Calculate the number of files to move for this class
        num_to_move = number_to_move

        # Randomly select 'num_to_move' files
        random_files = random.sample(files, num_to_move)

        # Move the selected files to the test folder for this class
        for file_name in random_files:
            src_file = os.path.join(class_path, file_name)
            dest_file = os.path.join(target_class_dir, file_name)
            shutil.copy(src_file, dest_file)