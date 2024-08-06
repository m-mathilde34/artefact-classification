import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

#model = load_model('../models/efficientnet/adamW_opt/ENB0_model_256_adamW.h5')
model = load_model('/app/models/efficientnet/adam_opt/ENB0_model_Adam_DATA_UNDERSAMPLING.h5')

test_dir = '/app/data/'
batch_size = 32

# To analyse misclassified data
misclassified_folder = "D:\misclassified"

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=(256,256),
    batch_size=batch_size,
    shuffle=False,
    )

class_names = test_generator.class_names
print(class_names)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - DOWNLOADING MISCLASSIFIED IMAGES - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def pre_process_filepaths(filepath_list):
    new_preprocessed_paths = []

    for element in filepath_list:
        string_split = element.split("\\")
        new_name = misclassified_folder + "\\" + string_split[-2] + "\\" + string_split[-1]
        new_preprocessed_paths.append(new_name)

    return new_preprocessed_paths


filepaths = []
y_true = []
misclassified_names = []
print(test_generator.file_paths)

predictions = model.predict(test_generator)
y_pred = tf.argmax(predictions, axis=1).numpy()

for image, labels in test_generator:
    true_labels = tf.argmax(labels, axis=1).numpy()
    y_true.extend(true_labels)

y_true = np.array(y_true)

misclassified_index = predictions.argmax(axis=1) != y_true
print(misclassified_index)

# print(y_true)
# print(y_pred)

#### PRINT DICTIONARY SHOWING FOR CLASS X, WHAT THEY ARE MISCLASSIFIED AS ####
dictionary = {
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0,
    "5": 0,
    "6": 0
}

counter1 = 0
while(counter1<len(y_true)):
    #Check to find misclassified
    if y_true[counter1] != y_pred[counter1]:
        #If misclassified is from a class we want to examine (in our case class 5)
        if y_true[counter1] == 4:
            #Add +1 in above dictionary to indicate which class it mistakes it for
            if y_pred[counter1] == 0:
                dictionary["1"] += 1
            elif y_pred[counter1] == 1:
                dictionary["2"] += 1
            elif y_pred[counter1] == 2:
                dictionary["3"] += 1
            elif y_pred[counter1] == 3:
                dictionary["4"] += 1
            elif y_pred[counter1] == 5:
                dictionary["6"] += 1
    counter1 = counter1 +1

print(dictionary)



##### GET FILEPATH OF MISCLASSIFIED #####

counter=0
while(counter<len(misclassified_index)):
    if misclassified_index[counter] == True:
        misclassified_names.append(test_generator.file_paths[counter])
    counter = counter+1

print(misclassified_names)
print(len(misclassified_names))

#### GET FILEPATH FOR NEW DESTINATION TO COPY TO #####

new_filepaths = pre_process_filepaths(misclassified_names)
print(new_filepaths)

##### MAKE A COPY OF MISCLASSIFIED IMAGES IN NEW FOLDER #####
new_counter = 0
for path in misclassified_names:
    try:
        shutil.copy(path, new_filepaths[new_counter])
        new_counter = new_counter + 1
    except:
        print("Error Occured while copying")
