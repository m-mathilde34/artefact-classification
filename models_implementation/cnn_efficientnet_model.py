import tensorflow as tf
#import tensorflow_addons as tfa
import os
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras import layers
from keras.layers import Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import Model, Input
from keras.optimizers import SGD
from keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

#get_image_labels method for class_weight when dealing with ohe labels from :
#Code from from get_image_labels() from :
#@trav-d13, 2023. Spatiotemporal Wildlife Image Classification [computer program]. Available from: https://github.com/Spatiotemporal-Wildlife-Classification/Wildlife-Classification/ [Accessed 20 March 2024]
# def get_image_labels(ds: tf.data.Dataset):
#     """Method generates class names from the dataset. This helps generate class weightings
#
#     Args:
#         ds (tf.data.Dataset): Either the train or test dataset which labels must be generated for.
#         classes (list): A list of the class labels (alphabetically ordered).
#
#     Returns:
#         (list): A list of labels in the provided dataset, in the same order as specified in the dataset.
#     """
#     ohe_labels = []
#     labels = [] # Container to hold the categorical labels
#     classes = ds.class_names
#
#     for x, y in ds:
#         ohe_labels.extend(np.argmax(y.numpy().tolist(), axis=1))  # Find index of biggest value (index of ohe label)
#     for label in ohe_labels:
#         labels.append(classes[label])  # Turn the index into the name of the class (categorical label)
#     return labels
#
#
# def get_classes_weights(train_data):
#     all_labels = get_image_labels(train_data)
#
#     training_weights = compute_class_weight(class_weight='balanced', classes=train_data.class_names,
#                                             y=all_labels)
#
#     #Turn our training_weights list into a dictionary for .fit method
#     class_labels = list(range(0, len(train_data.class_names)))
#     weights = dict(zip(class_labels, training_weights))
#     print(weights)
#
#     return weights

# Define the path to your dataset directory
dataset_directory = '/app/data/'

# Save epoch history
def save_history(history):
    history.history['epoch'] = [i for i in range(len(history.history['loss']))]
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('ENB0_epoch_history_Adam_UNDERSAMPLING.csv')


# Define parameters for the dataset
batch_size = 32
image_size = (256, 256)
seed = 42
input = (256,256,3)
epoch = 100
class_number = 6

print("About to create the training dataset...")

# Create a training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_directory,
    label_mode='categorical',
    validation_split=0.2,  # Split 80% of the data for training
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)

print("Training Dataset Created")
print("About to create the validation dataset...")

# Create a validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_directory,
    label_mode='categorical',
    validation_split=0.2,  # Split 20% of the data for validation
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
)

print("Validation Dataset Created")

# print("Getting training weights...")

# training_weights = get_classes_weights(train_dataset)

print("Building the model...")

#Base Model
efficientNet_base_model = EfficientNetB0(weights='imagenet',
                  include_top=False,
                  input_shape=input)

for layer in efficientNet_base_model.layers:
    layer.trainable = False

inputs = Input(shape=input)

build = efficientNet_base_model(inputs)
build = Flatten()(build)
build = Dense(512)(build)
build = Activation('relu')(build)
build = BatchNormalization()(build)
build = Dropout(0.5)(build)
build = layers.Dense(class_number, activation='softmax')(build)

artefactModel = Model(inputs=inputs, outputs=build)

#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4)

artefactModel.compile(optimizer='adam',
              loss=['categorical_crossentropy'],
              metrics=['accuracy'])

print("Model Compiled")

# checkpoint_callback = ModelCheckpoint(
#     filepath='/app/model_weight_saves/',
#     #save_best_only = True,
#     save_weights_only=True,
#     save_freq='epoch',
# )
#
# if os.path.exists('/app/model_weight_saves/'):
#   artefactModel = tf.keras.models.load_model('/app/model_weight_saves/')
#   print('checkpoint restored')

print("Starting the training!")

history = artefactModel.fit(
        train_dataset,
        shuffle=True,
        steps_per_epoch=35122 // (batch_size*epoch),
        epochs=epoch,
        #callbacks=[checkpoint_callback],
        #class_weight=training_weights,
        validation_data=validation_dataset,
        validation_steps=8780 // (batch_size*epoch),
        verbose=2
)

save_history(history)

model_folder = '/app/models/efficientnet/adam_opt/'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
artefactModel.save(model_folder+"ENB0_model_Adam_DATA_UNDERSAMPLING.h5")
