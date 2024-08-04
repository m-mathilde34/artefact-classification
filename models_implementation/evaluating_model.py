import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

#For test purposes only
#test_dir = "D:/images_resized_test"
#test_dir = "D:/tiny_test_set"

#model = load_model('../models/efficientnet/adamW_opt/ENB0_model_256_adamW.h5')
model = load_model('/app/models/efficientnet/adam_opt/ENB0_model_Adam_DATA_UNDERSAMPLING.h5')

test_dir = '/app/data/'
batch_size = 32

test_generator = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=(256,256),
    batch_size=batch_size,
    shuffle=False,
    )

# test_loss, test_acc = model.evaluate(test_generator)
# print('Test loss: ', test_loss)
# print('Test accuracy: ', test_acc)

class_names = test_generator.class_names
print(class_names)

# - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - EVALUATING MODEL - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - -
target_names=['egyptian', 'grecoroman', 'orient', 'islam', 'byzantium', 'asian']
y_true = []

predictions = model.predict(test_generator)
y_pred = tf.argmax(predictions, axis=1).numpy()

for image, labels in test_generator:
    true_labels = tf.argmax(labels, axis=1).numpy()
    y_true.extend(true_labels)

y_true = np.array(y_true)

print(classification_report(y_true, y_pred, target_names=target_names, digits=6))
