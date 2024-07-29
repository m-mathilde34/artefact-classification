import os.path
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import cv2

#source_data = "D:\images_resized\\5_byzantium"


# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
    rotation_range=90,
    shear_range=40,
    zoom_range=[0.5,1.3],
    horizontal_flip=True,
    brightness_range=(0.2, 1.7))

#Get your images paths
images_folder = "D:\images_resized\\5_byzantium"
all_files = []

# Walk through the directory and its subdirectories
for root, _, files in os.walk(images_folder):
    for file in files:
        file_path = os.path.join(root, file)
        all_files.append(file_path)

data = []
for f in all_files:
    img = cv2.imread(f)
    img_colour = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = img_to_array(img_colour)
    x = x.reshape((1,) + x.shape)

    i = 0
    path, dirs, files = next(os.walk("D:\images_resized\\5_byzantium"))
    stop = 5 # Set how many augmented images you would like

    base = os.path.splitext(os.path.split(f)[-1])[0] #Get original name of file to add to new name of file.

    for batch in datagen.flow (x, batch_size=1, save_to_dir =r'D:\images_resized\5_byzantium_augmented',save_prefix=base,save_format='jpg'):
        i+=1
        if i==stop:
          break
