##
import pandas as pd
import tensorflow as tf
import helper
import numpy as np
import datetime

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
##
def createModel():
    """
    create multitask model with classification for age prediction
    :return: created model
    """
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=0.25)
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
    age_detection = tf.keras.layers.Dense(1024, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01))(feature_extractor)
    age_detection = tf.keras.layers.Dropout(0.2)(age_detection)
    #age_detection = tf.keras.layers.BatchNormalization()(age_detection)
    age_detection = tf.keras.layers.Dense(512, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(0.01))(age_detection)
    #age_detection = tf.keras.layers.BatchNormalization()(age_detection)
    age_detection = tf.keras.layers.Dropout(0.2)(age_detection)
    age_detection = tf.keras.layers.Dense(18, activation="softmax", name="age_detection")(age_detection)

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
    if x < 10:
        return -1
    if 10 <= x < 15:
        return 0
    if 15 <= x < 20:
        return 1
    if 20 <= x < 25:
        return 2
    if 25 <= x < 30:
        return 3
    if 30 <= x < 35:
        return 4
    if 35 <= x < 40:
        return 5
    if 40 <= x < 45:
        return 6
    if 45 <= x < 50:
        return 7
    if 50 <= x < 55:
        return 8
    if 55 <= x < 60:
        return 9
    if 60 <= x < 65:
        return 10
    if 65 <= x < 70:
        return 11
    if 70 <= x < 75:
        return 12
    if 75 <= x < 80:
        return 13
    if 80 <= x < 85:
        return 14
    if 85 <= x < 90:
        return 15
    if 90 <= x < 95:
        return 16
    if 95 <= x <= 100:
        return 17

def create_dataset(csv_path,only_age=False):
    table_data = pd.read_csv(csv_path)
    table_data['age_clustered'] = table_data["age"].apply(clusterAges)
    if only_age:
        table_data = table_data[table_data["age"] >= 10]
    table_data = shuffle(table_data, random_state=123)
    data = tf.data.Dataset.from_tensor_slices((table_data["image_path"], table_data[["face", "mask", "age_clustered"]]))
    ds = data.map(process_path)
    ds = ds.batch(32)
    return ds, table_data

##
train_ds,train_table = create_dataset("images/featureTableTrain.csv",only_age=True)
val_ds,val_table = create_dataset("images/featureTableVal.csv", only_age=True)

##
model = createModel()
model = compileModel(model)
##
alpha = 0.25
epochs = 10
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tf.debugging.set_log_device_placement(True)
model_history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[tensorboard_callback], )
model.save("saved_model/" + "milestone2_classification_" + str(epochs) + "_epochs_" + str(alpha).split(".")[0] + str(alpha).split(".")[1] + "alpha")

## load test data
ONLY_AGE = True
test_table = pd.read_csv("images/featureTableVal.csv") #oder featureTableTest.csv
test_table['age_clustered'] = test_table["age"].apply(clusterAges)
if ONLY_AGE:
    test_table = test_table[test_table["age"] >= 10]
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


##