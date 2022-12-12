##
import pandas as pd
import tensorflow as tf
from datetime import datetime
import helper
import os
import time
from sklearn.utils import shuffle
import numpy as np

import helper_multitask


##
def clusterAges(x):
    if x < 1:
        return -1
    if 0 < x <= 10:
        return 0
    if 10 < x <= 20:
        return 1
    if 20 < x <= 30:
        return 2
    if 30 < x <= 40:
        return 3
    if 40 < x <= 50:
        return 4
    if 50 < x <= 60:
        return 5
    if 60 < x <= 70:
        return 6
    if 70 < x <= 80:
        return 7
    if 80 < x <= 90:
        return 8
    if 90 < x <= 100:
        return 9

##
@tf.function
def get_weights(weights):
    return {'face_detection': tf.reshape(tf.keras.backend.cast(weights["face_detection"], tf.keras.backend.floatx()), (-1, 1)),
            'mask_detection': tf.reshape(tf.keras.backend.cast(weights["mask_detection"], tf.keras.backend.floatx()), (-1, 1)),
            'age_detection': tf.reshape(tf.keras.backend.cast(weights["age_detection"], tf.keras.backend.floatx()), (-1, 1))}


@tf.function
def get_label(label):
    return {'face_detection': tf.reshape(tf.keras.backend.cast(label[0], tf.keras.backend.floatx()), (-1,1)),
               'mask_detection':  tf.reshape(tf.keras.backend.cast(label[1], tf.keras.backend.floatx()), (-1,1)),
                'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1,1))}


@tf.function
def decode_img(img_path):
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def process_path(file_path,labels,sample_weights):
    label = get_label(labels)
    img = decode_img(file_path)
    weight = get_weights(sample_weights)
    return img, label, weight

def process_path_classification(file_path,labels):
    label = get_label(labels)
    img = decode_img(file_path)
    return img, label


##
def create_dataset_regression(csv_path,only_age=False, only_mask=False):
    table_data = pd.read_csv(csv_path)
    if only_age:
        table_data = table_data[table_data["age"] >= 1]
    if only_mask:
        table_data = table_data[table_data["face"] == 1]
    table_data = shuffle(table_data, random_state=123)
    table_data['face_weights'] = 1
    table_data['mask_weights'] = table_data['face']
    table_data['age_weights'] = table_data["age"].apply(lambda x: 1 if x >= 1 else 0)
    dict_weighted = {"face_detection": np.array(table_data['face_weights']),
                     "mask_detection": np.array(table_data['mask_weights']),
                     "age_detection": np.array(table_data['age_weights'])}
    data = tf.data.Dataset.from_tensor_slices(
        (table_data["image_path"], table_data[["face", "mask", "age"]], dict_weighted))
    ds = data.map(process_path)
    ds = ds.batch(32)
    return ds, table_data


def create_dataset_classification(csv_path,only_age=False, only_mask=False):
    table_data = pd.read_csv(csv_path)
    table_data['age_clustered'] = table_data["age"].apply(clusterAges)
    if only_age:
        table_data = table_data[table_data["age"] >= 1]
    if only_mask:
        table_data = table_data[table_data["face"] == 1]
    table_data = shuffle(table_data, random_state=123)
    data = tf.data.Dataset.from_tensor_slices((table_data["image_path"], table_data[["face", "mask", "age_clustered"]]))
    ds = data.map(process_path_classification)
    ds = ds.batch(32)
    return ds, table_data


##
test_ds_face, test_table_face = create_dataset_regression("images/featureTableTest.csv")
test_ds_age, test_table_age = create_dataset_regression("images/featureTableTest.csv", only_age=True)
test_ds_mask, test_table_mask = create_dataset_regression("images/featureTableTest.csv", only_mask=True)

test_ds_face_class, test_table_face_class = create_dataset_classification("images/featureTableTest.csv")
test_ds_age_class, test_table_age_class = create_dataset_classification("images/featureTableTest.csv", only_age=True)
test_ds_mask_class, test_table_mask_class = create_dataset_classification("images/featureTableTest.csv", only_mask=True)
##

helper_multitask.create_categorical_dataset()

## evaluate through all models

data = []
directory = "saved_model"

for model_path in os.listdir(directory):
    # because of macOS DS_Store folder
    if model_path == ".DS_Store":
        continue
    model = tf.keras.models.load_model(directory + "/" + model_path)
    evaluation = model.evaluate(test_ds_face)