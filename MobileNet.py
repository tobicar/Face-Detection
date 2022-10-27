## imports
#import kwargs as kwargs
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tensorflow.keras.applications import MobileNetV3Large
import datetime


## CNN

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

##
model.summary()

##
ds, info = tfds.load("imagenet_v2", as_supervised=True, with_info=True)
NUM_CLASSES = info.features["label"].num_classes
#ds1, info1 = tfds.load("beans", as_supervised=True, with_info=True)
## how to resize images
size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))


##
ds_train = ds["test"] # gibt nur testdaten im Set
assert isinstance(ds_train, tf.data.Dataset)



#df = tfds.as_dataframe(ds_train.take(10), info)

## show examples
fig = tfds.show_examples(ds_train.take(10),info)

##
plt.figure(figsize=(10, 10))
for images, labels in ds_train.take(10):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images.numpy().astype("uint8"))
        plt.axis("off")

## print single picture

for images
plt.imshow(ds_train.get_single_element(10))


## Dictionary

for example in ds1.take(1):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    print(list(example.keys()))
    image = example["image"]
    label = example["label"]
    print(image.shape, label)

## Tuple
def getPicture(dataset):
    for example in dataset.take(1): #TODO: ersetzen durch direkten Zugriff auf ein Element
        #image = example[0]
        #label = example[1]
        #print(image.shape, label)
        return example[0]

## Preiction

def predictImage(model, image):
    """
    :param model: vortrainiertes Model
    :param image: EagerTensor mit Dimension (x, y, 3)
    :return: Tupel mit Wahrscheinlichkeit und Klasse
    """
    image_reshaped = tf.expand_dims(image, axis=0)
    prediction = model.predict(image_reshaped)
    return np.max(prediction), np.argmax(prediction)

prediction, prediction_class = predictImage(model, getPicture(ds_train))

##
def changeTensorShape(image, label):
    return tf.expand_dims(image, axis=0), label

##
def getLabel(number):
    pass

##
def importImages(directory):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=["Kein_Mensch", "Mensch"],
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=True)
## load the imgaes

ds = importImages("images")

##
def loadImagePredict(path, model):
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    return tf.keras.applications.mobilenet_v3.decode_predictions(predictions)


## compile the model
#TODO: from_logits ja oder nein?
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
## train the model
model.fit(ds,epochs=2,callbacks=[tensorboard_callback])