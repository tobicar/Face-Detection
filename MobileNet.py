## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

## CNN MobileNet V3
model = tf.keras.applications.MobileNetV3Large(
    input_shape=(None, None, 3), # default (224,224,3)
    alpha=1.0,
    minimalistic=False,
    include_top=True,
    weights=None,
    input_tensor=None,
    classes=2,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,)

## CNN info
model.summary()

## show image
def printOriginalImage(path, text):
    """ show image
    :param path: path to image file
    :return: -
    """
    image = tf.keras.preprocessing.image.load_img(path)
    figure, ax = plt.subplots()
    ax.imshow(image)

    # hide y-axis
    ax.get_yaxis().set_visible(False)

    ax.set_xticklabels([])

    ax.set_xlabel(text, loc='left')

def getImageFromDataset(dataset):
    """ extract image from dataset
    :param dataset: defined dataset
    :return: image (as array)
    """
    for example in dataset.take(1): #TODO: ersetzen durch direkten Zugriff auf ein Element
        #image = example[0]
        #label = example[1]
        #print(image.shape, label)
        return example[0]

## cut decimals
def trunc(values, decs=2):
    """ cuts decimal numbers
    :param values: number (float)
    :param decs: decimals to cut
    :return: number
    """
    return np.trunc(values*10**decs)/(10**decs)

## config output
labels = ["Kein Mensch", "Mensch"]
def getPredictionText(prediction_array):
    """ get label to prediction
    :param prediction_array: array of prediction
    :return: string with max prediction and label
    """
    sort_array = np.argsort(prediction_array)[::-1]
    print(sort_array)
    print(prediction_array)
    text = ""
    for e in sort_array:
        text += labels[e] + ": " + str(trunc(prediction_array[e], 4)) + "\n"
    return text

## import images from directory
def importImages(directory):
    """ import images from directory
    :param directory: path to image folder
    :return: -
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["Kein_Mensch", "Mensch"],
        color_mode="rgb",
        batch_size=1,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=True)

ds = importImages("images")

## get prediction
def loadImagePredict(path, model):
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    text = getPredictionText(predictions[0])
    printOriginalImage(path, text)
    return getPredictionText(predictions[0])

## compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

## split dataset

ds_size = len(ds.file_paths)
train_size = int(0.7 * ds_size)
val_size = int(0.15 * ds_size)
test_size = int(0.15 * ds_size)

ds_train = ds.take(train_size)
ds_test = ds.skip(train_size)
ds_val = ds_test.skip(val_size)
ds_test = ds_test.take(test_size)

## train the model
tf.debugging.set_log_device_placement(True)

model.fit(ds_train, batch_size=32, epochs=50, callbacks=[tensorboard_callback],
          validation_data=ds_val, validation_batch_size=32)
model.save("saved_model/model_50epochs")

## load model
model = tf.keras.models.load_model('saved_model/model_50epochs')

## OpenFileDialog

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

filetypes =[('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if file_path:
    loadImagePredict(file_path, model)

##

ds_size = len(ds.file_paths)
train_size = int(0.7 * ds_size)
val_size = int(0.15 * ds_size)
test_size = int(0.15 * ds_size)

full_dataset = ds
#full_dataset = full_dataset.shuffle()
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

##
model.evaluate(test_dataset)
