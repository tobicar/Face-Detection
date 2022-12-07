##
import pandas as pd
import tensorflow as tf
import helper
import numpy as np
import datetime

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
##
def createModel(alpha=0.25, dropout=0.2, large_version=False):
    """
    create multitask model with classification for age prediction
    :return: created model
    """
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)

    # face detection
    face_detection = tf.keras.layers.Dense(1, activation="sigmoid", name='face_detection')(feature_extractor)

    # mask detecion
    mask_detection = tf.keras.layers.Dense(1, activation="sigmoid", name='mask_detection')(feature_extractor)

    # age detecion
    # one Class for no age = 0
    # faces with unknown age = -1 --> ignored
    #  18 classes
    if large_version:
        age_detection = tf.keras.layers.Dense(1024, activation="relu")(feature_extractor)
        age_detection = tf.keras.layers.Dropout(dropout)(age_detection)
        age_detection = tf.keras.layers.Dense(512, activation="relu")(age_detection)
        age_detection = tf.keras.layers.Dropout(dropout)(age_detection)
        age_detection = tf.keras.layers.Dense(256, activation="relu")(age_detection)
    else:
        age_detection = tf.keras.layers.Dense(256, activation="relu",)(feature_extractor)
    age_detection = tf.keras.layers.Dropout(dropout)(age_detection)
    age_detection = tf.keras.layers.Dense(10, activation="softmax", name="age_detection")(age_detection)

    model = tf.keras.Model(inputs=inputs, outputs=[face_detection, mask_detection, age_detection])
    return model

##
def compileModel(model):
    model.compile(optimizer='adam', loss={'face_detection': 'binary_crossentropy',
                                          'mask_detection': 'binary_crossentropy',
                                          'age_detection': tf.keras.losses.SparseCategoricalCrossentropy(ignore_class=-1)},
                  loss_weights={'face_detection': 0.33, 'mask_detection': 0.33, 'age_detection': 0.33},
                  metrics={'face_detection': 'accuracy',
                           'mask_detection': 'accuracy',
                           'age_detection': 'accuracy'})
    return model

##
@tf.function
def get_label(label):
    # if label[2] == 0:
    #    age = label[2]
    # else:
    #    age = label[2]-9
    #if only_age:
    #return {'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1,1))}
    #else:
    return {'face_detection': tf.reshape(tf.keras.backend.cast(label[0], tf.keras.backend.floatx()), (-1,1)),
               'mask_detection':  tf.reshape(tf.keras.backend.cast(label[1], tf.keras.backend.floatx()), (-1,1)),
                'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1,1))} # -9 because there are no classes between age 0 and 9


@tf.function
def decode_img(img_path):
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img
    # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # return tf.keras.preprocessing.image.img_to_array(img)


def process_path(file_path,labels):
    label = get_label(labels)
    # label = {'face_detection': 1,'mask_detection': 2,'age_detection':3}
    # Load the raw data from the file as a string
    # img = tf.io.read_file(file_path)
    img = decode_img(file_path)
    #img = file_path
    return img, label



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


def create_dataset(csv_path,only_age=False):
    table_data = pd.read_csv(csv_path)
    table_data['age_clustered'] = table_data["age"].apply(clusterAges)
    if only_age:
        table_data = table_data[table_data["age"] >= 1]
    table_data = shuffle(table_data, random_state=123)
    data = tf.data.Dataset.from_tensor_slices((table_data["image_path"], table_data[["face", "mask", "age_clustered"]]))
    ds = data.map(process_path)
    ds = ds.batch(32)
    return ds, table_data

##
train_ds,train_table = create_dataset("images/featureTableTrain.csv")
val_ds,val_table = create_dataset("images/featureTableVal.csv", only_age=True)

##
model = createModel()
model = compileModel(model)
## generate classification trainings loop
EPOCHS = [100]
ALPHAS = [0.25]
DROPOUTS = [0.2]
LARGE_VERSION = [False, True]

for large in LARGE_VERSION:
    for alpha in ALPHAS:
        for dropout in DROPOUTS:
            for epochs in EPOCHS:
                model = createModel(alpha=alpha, dropout=dropout, large_version=large)
                model = compileModel(model)
                name = r"classification" + str(epochs) + "epochs_" + \
                       str(alpha) + "alpha_" + str(dropout) + "dropout"
                log_dir = "logs/fit/" + name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                tf.debugging.set_log_device_placement(True)
                model_history = model.fit(train_ds,
                                          epochs=epochs,
                                          validation_data=val_ds,
                                          callbacks=[tensorboard_callback])

                # save model
                model.save("saved_model/Milestone3/" + name)


## load test data
ONLY_AGE = True
test_table = pd.read_csv("images/featureTableVal.csv") #oder featureTableTest.csv
test_table['age_clustered'] = test_table["age"].apply(clusterAges)
if ONLY_AGE:
    test_table = test_table[test_table["age"] >= 1]
test_table = shuffle(test_table,random_state=123)
test_data = tf.data.Dataset.from_tensor_slices((test_table["image_path"], test_table[["face", "mask", "age_clustered"]]))
ds_test = test_data.map(process_path)
ds_test = ds_test.batch(32)
## augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])


batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)
    # Use buffered prefetching on all datasets.
    return ds


