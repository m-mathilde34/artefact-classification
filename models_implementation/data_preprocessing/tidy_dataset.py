import cv2
import numpy as np
import os
import shutil

img_1 = "D:\\1_LOUVRE_0001138450_OG.JPG"

img1 = cv2.imread(img_1)

#For test purposes
#img_2 = ""
# img2 = cv2.imread(img_2)
#
# if img1.shape == img2.shape:
#     print("The images have same size and channels")
#
# difference = cv2.subtract(img1, img2)
# b, g, r = cv2.split(difference)
# if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
#     print("The images are completely Equal")


# # # # # # # # # # # # # # # # # #


input_folder = "D:\images_resized"

all_files = []

# Walk through the directory and its subdirectories
for root, _, files in os.walk(input_folder):
    for file in files:
        file_path = os.path.join(root, file)
        all_files.append(file_path)

#print(all_files)

#Go through each image individually
for file_path in all_files:
    img = cv2.imread(file_path)

    #Check whether an image is the same as the empty test one
    if img.shape == img1.shape:
        difference = cv2.subtract(img1, img)
        b, g, r = cv2.split(difference)

        #If they are, move it elsewhere
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            try:
                shutil.move(file_path, "D:\empty_pictures")
            except:
                print("Error Occured while copying")
