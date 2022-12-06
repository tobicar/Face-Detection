##
import pandas as pd
import tensorflow as tf
import helper
import numpy as np
import datetime
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

##
def add_face(x):
    greater = tf.keras.backend.greater_equal(x, 0.5) #will return boolean values
    greater = tf.keras.backend.cast(greater, dtype=tf.keras.backend.floatx()) #will convert bool to 0 and 1
    return greater


def createModel(multiple_dense_layers=False, alpha=0.25, dropout=0.2):
    """
    create multitask model with regression for age prediction
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
    face_detection_ground_truth = tf.keras.layers.Lambda(add_face)(face_detection)

    if multiple_dense_layers:
        feature_extractor = tf.keras.layers.Dense(1000, activation='relu')(feature_extractor)
        feature_extractor = tf.keras.layers.Dropout(dropout)(feature_extractor)
        feature_extractor = tf.keras.layers.Dense(500, activation='relu')(feature_extractor)
        feature_extractor = tf.keras.layers.Dropout(dropout)(feature_extractor)
    age_detection = tf.keras.layers.Dense(250, activation="relu")(feature_extractor)
    age_detection = tf.keras.layers.Dropout(dropout)(age_detection)
    age_detection = tf.keras.layers.Dense(1)(age_detection)
    age_detection = tf.keras.layers.multiply([age_detection, face_detection_ground_truth], name="age_detection")

    model = tf.keras.Model(inputs=inputs, outputs=[face_detection, mask_detection, age_detection])
    return model


## create model for age prediction only
def create_model_age_classification():
    """
    create model which predicts the age of a face by classification
    :return: created model
    """
    # TODO: Klassifizierung für Altersklassen (z.B. 10-20, 20-30, etc.)
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=0.25)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)

    age_detection = tf.keras.layers.Dense(102, activation="softmax", name="age_detection")(feature_extractor)
    model = tf.keras.Model(inputs=inputs, outputs=age_detection)
    return model


def create_model_age_regression(alpha=0.25):
    """
    create model which predicts the age of a face by regression
    :param alpha: parameter alpha of pretrained model (possible inputs: 0.25, 0.5, 0.75, 1)
    :return: created model
    """
    # TODO: alpha variieren
    # TODO: validierung hinzufügen
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)
    # TODO: mehr Dense Layer
    feature_extractor = tf.keras.layers.Dense(1000, activation='relu')(feature_extractor)
    feature_extractor = tf.keras.layers.Dense(500, activation='relu')(feature_extractor)
    feature_extractor = tf.keras.layers.Dense(250, activation='relu')(feature_extractor)
    age_detection = tf.keras.layers.Dense(1, name="age_detection")(feature_extractor)
    model = tf.keras.Model(inputs=inputs, outputs=age_detection)
    return model


def compile_model_age_regression(model, learning_rate=0.001):
    """
    compile regression model for age prediction only
    :param model: model to compile
    :param learning_rate: learning rate of optimizer
    :return: compiled model
    """
    # TODO: learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse',
                  metrics=['mse'])
    return model


def custom_sparse_categorical_crossentropy(y_true, y_pred):
    """
    create custom sparse categorical crossentropy
    :param y_true:
    :param y_pred:
    :return: loss function
    """
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, ignore_class=-1)


def compile_model_age(model):
    """
    compile classification model for age prediction only
    :param model: model to compile
    :return: compiled model
    """
    model.compile(optimizer='adam', loss=custom_sparse_categorical_crossentropy,
                  metrics='accuracy')
    return model

##

def customMSE(y_true, y_pred):
    mask = tf.keras.backend.cast(tf.keras.backend.not_equal(y_true, -1), tf.keras.backend.floatx())
    mask2 = tf.keras.backend.cast(tf.keras.backend.not_equal(y_true, 0), tf.keras.backend.floatx())
    return tf.keras.losses.mse(y_true * mask * mask2, y_pred * mask * mask2)


def compileModel(model, loss='huber'):
    model.compile(optimizer='adam', loss={'face_detection': 'binary_crossentropy',
                                          'mask_detection': 'binary_crossentropy',
                                          'age_detection': loss},
                  loss_weights={'face_detection': 0.33, 'mask_detection': 0.33, 'age_detection': 0.33},
                  metrics={'face_detection': 'accuracy',
                           'mask_detection': 'accuracy',
                           'age_detection': ['mae', 'mse']})
    return model


##
path3 = "images/test/no_face/00000000_(5).jpg"
path = "images/test/face/10_0_0_20161220222308131.jpg"
path2 = "/Users/tobias/Downloads/29177_1_mundschutz-typll-gruen_1.jpg"
image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
image2 = tf.keras.preprocessing.image.load_img(path2, target_size=(224, 224))
image3 = tf.keras.preprocessing.image.load_img(path3, target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr2 = tf.keras.preprocessing.image.img_to_array(image2)
input_arr3 = tf.keras.preprocessing.image.img_to_array(image3)
x_train = np.array([input_arr, input_arr2, input_arr3])
labels_face = np.array([1, 1, 0], dtype=np.float32)
labels_age = np.array([10, 30, 0], dtype=np.float32)
labels_mask = np.array([0, 1, 0], dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, {'face_detection': labels_face, 'mask_detection': labels_mask, 'age_detection': labels_age})).batch(2)

##
@tf.function
def get_weights(weights):
    return {'face_detection': tf.reshape(tf.keras.backend.cast(weights["face_detection"], tf.keras.backend.floatx()), (-1, 1)),
            'mask_detection': tf.reshape(tf.keras.backend.cast(weights["mask_detection"], tf.keras.backend.floatx()), (-1, 1)),
            'age_detection': tf.reshape(tf.keras.backend.cast(weights["age_detection"], tf.keras.backend.floatx()), (-1, 1))}


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
        img, channels=num_channels, expand_animations=False)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img
    # img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    # return tf.keras.preprocessing.image.img_to_array(img)


def process_path(file_path,labels,sample_weights):
    label = get_label(labels)
    # label = {'face_detection': 1,'mask_detection': 2,'age_detection':3}
    # Load the raw data from the file as a string
    # img = tf.io.read_file(file_path)
    img = decode_img(file_path)
    weight = get_weights(sample_weights)
    #img = file_path
    return img, label,weight


##
def create_dataset(csv_path):
    table_data = pd.read_csv(csv_path)
    table_data = shuffle(table_data, random_state=123)
    table_data['face_weights'] = 1
    table_data['mask_weights'] = table_data['face']
    table_data['age_weights'] = table_data["age"].apply(lambda x: 1 if x >= 10 else 0)
    dict_weighted = {"face_detection": np.array(table_data['face_weights']),
                     "mask_detection": np.array(table_data['mask_weights']),
                     "age_detection": np.array(table_data['age_weights'])}
    data = tf.data.Dataset.from_tensor_slices(
        (table_data["image_path"], table_data[["face", "mask", "age"]], dict_weighted))
    ds = data.map(process_path)
    ds = ds.batch(32)
    #ds = ds.shuffle(ds.__len__().numpy(), seed=123, reshuffle_each_iteration=False).batch(32)
    return ds


train_ds = create_dataset("images/featureTableTrain.csv")
val_ds = create_dataset("images/featureTableVal.csv")

##
model = createModel(multiple_dense_layers=True)
model = compileModel(model)
model_history = model.fit(train_ds, epochs=10, validation_data=val_ds)

##

# Regression
EPOCHS = [100]
MULTIPLE_DENSE_LAYERS = [False, True]
ALPHAS = [0.25, 1.00]
LOSS = ['mse', 'huber']
DROPOUTS = [0.2, 0.8]

for multiple_dense_layers in MULTIPLE_DENSE_LAYERS:
    for alpha in ALPHAS:
        for dropout in DROPOUTS:
            model = createModel(multiple_dense_layers, alpha=alpha, dropout=dropout)
            for loss in LOSS:
                model = compileModel(model, loss=loss)
                for epochs in EPOCHS:
                    name = "milestone2_regression_" + str(epochs) + "epochs_" +\
                           str(alpha).split(".")[0] + str(alpha).split(".")[1] + "alpha_" + str(dropout) + "dropout_" +\
                           loss
                    if multiple_dense_layers:
                        name += "_multipleDenseLayers"

                    log_dir = "logs/fit/" + name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                    tf.debugging.set_log_device_placement(True)

                    model_history = model.fit(train_ds,
                                              epochs=epochs,
                                              validation_data=val_ds,
                                              callbacks=[tensorboard_callback])

                    # save model
                    model.save("saved_model/" + name)

## generate History and plot it

epochs_range = range(len(model_history.epoch))

complete_loss = model_history.history['loss']
face_detection_loss = model_history.history["face_detection_loss"]
mask_detection_loss = model_history.history["mask_detection_loss"]
age_detection_loss = model_history.history["age_detection_loss"]

face_detection_accuracy = model_history.history['face_detection_accuracy']
mask_detection_accuracy = model_history.history['mask_detection_accuracy']
age_detection_mse = model_history.history["age_detection_mse"]

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, face_detection_accuracy, label='Face Accuracy')
plt.plot(epochs_range, mask_detection_accuracy, label='Mask Accuracy')
#plt.plot(epochs_range, age_detection_mse, label='Age MSE')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, complete_loss, label='Training Loss')
plt.plot(epochs_range, face_detection_loss, label='Face Loss')
plt.plot(epochs_range, mask_detection_loss, label='Mask Loss')
plt.plot(epochs_range, age_detection_loss, label='Age Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()