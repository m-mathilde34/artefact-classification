import tensorflow as tf
import os
from tensorflow.keras.applications import ResNet50
from keras.applications.densenet import preprocess_input, decode_predictions
from tensorflow.keras import layers
from keras.layers import Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import Model, Input
from keras.optimizers import SGD
from keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd

# Define the path to your dataset directory
dataset_directory = '/app/data/'

# Save epoch history
def save_history(history):
    history.history['epoch'] = [i for i in range(len(history.history['loss']))]
    df = pd.DataFrame.from_dict(history.history)
    df.to_csv('ResNet50_epoch_history_256_CLEAN_SDG.csv')


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
print("Building the model...")

#Base Model
resnet_50_base = ResNet50(weights='imagenet',
                          include_top=False,
                          input_shape=input)

for layer in resnet_50_base.layers:
    layer.trainable = False

inputs = Input(shape=input)

build = resnet_50_base(inputs)
build = Flatten()(build)
build = Dense(512)(build)
build = Activation('relu')(build)
build = BatchNormalization()(build)
build = Dropout(0.5)(build)
build = layers.Dense(class_number, activation='softmax')(build)

artefactModel = Model(inputs=inputs, outputs=build)

#optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

artefactModel.compile(optimizer = SGD(learning_rate=1e-4, momentum=0.9),
                      loss=['categorical_crossentropy'],
                      metrics=['accuracy'])

print("Model Compiled")

# checkpoint_callback = ModelCheckpoint(
#     filepath='/app/model_weight_saves/',
#     #save_best_only = True,
#     save_weights_only=True,
#     save_freq='epoch',
#     )
#
# if os.path.exists('/app/model_weight_saves/'):
#     artefactModel = tf.keras.models.load_model('/app/model_weight_saves/')
#     print('checkpoint restored')

print("Starting the training!")

history = artefactModel.fit(
    train_dataset,
    shuffle=True,
    steps_per_epoch=160899 // (batch_size*epoch),
    epochs=epoch,
    #callbacks=[checkpoint_callback],
    validation_data=validation_dataset,
    validation_steps=40224 // (batch_size*epoch),
    verbose=2
    )

save_history(history)

model_folder = '/app/models/resnet/sdg_opt/'
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
artefactModel.save(model_folder+"resnet50_model_256_SDG_CLEAN.h5")
