import tensorflow as tf
import helper
from sklearn.utils import shuffle
import numpy as np
import datetime
import pandas as pd

def add_face(x):
    """
    Function for Lambda Layer used in regression Model
    :param x: probability if picture contains a face
    :return: if x greater than 0.5 -> returns 1 otherwise 0
    """
    greater = tf.keras.backend.greater_equal(x, 0.5)  # will return boolean values
    greater = tf.keras.backend.cast(greater, dtype=tf.keras.backend.floatx())  # will convert bool to 0 and 1
    return greater


def create_model(version, alpha=0.25, dropout=0.2, large_version=False, regularizer=False):
    """
    function generates a keras model for training with the defined parameters
    :param version: possible versions are regression or classification
    :param alpha: parameter for the feature extractor (untrainable mobilenetv1)
    :param dropout: percentage of dropout for the dropout layers in the model
    :param large_version: true or false: large version contains a bigger head with more dense and Dropout layers for the age classification or age regression problem
    :param regularizer: true or false: decides if the age dense layers have a L2-regularization kernel or not
    :return: the generated keras model
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

    # mask detection
    mask_detection = tf.keras.layers.Dense(1, activation="sigmoid", name='mask_detection')(feature_extractor)

    # age detection
    feature_extractor_age = feature_extractor

    if large_version:
        feature_extractor_age = tf.keras.layers.Dense(1024,
                                                      activation='relu',
                                                      kernel_regularizer=tf.keras.regularizers.l2(
                                                          0.01) if regularizer else None
                                                      )(feature_extractor_age)
        feature_extractor_age = tf.keras.layers.Dropout(dropout)(feature_extractor_age)
        feature_extractor_age = tf.keras.layers.Dense(512,
                                                      activation='relu',
                                                      kernel_regularizer=tf.keras.regularizers.l2(
                                                          0.01) if regularizer else None
                                                      )(feature_extractor_age)
        feature_extractor_age = tf.keras.layers.Dropout(dropout)(feature_extractor_age)

    feature_extractor_age = tf.keras.layers.Dense(256,
                                                  activation="relu",
                                                  kernel_regularizer=tf.keras.regularizers.l2(
                                                      0.01) if regularizer else None
                                                  )(feature_extractor_age)
    feature_extractor_age = tf.keras.layers.Dropout(dropout)(feature_extractor_age)

    if version == "classification":
        age_detection = tf.keras.layers.Dense(10, activation="softmax", name="age_detection")(feature_extractor_age)
    elif version == "regression":
        feature_extractor_age = tf.keras.layers.Dense(1)(feature_extractor_age)
        face_detection_ground_truth = tf.keras.layers.Lambda(add_face)(face_detection)
        age_detection = tf.keras.layers.multiply([feature_extractor_age, face_detection_ground_truth],
                                                 name="age_detection")
    else:
        return None

    model = tf.keras.Model(inputs=inputs, outputs=[face_detection, mask_detection, age_detection])
    return model


def compile_model(model, version, loss_weight_face=0.33, loss_weight_mask=0.33, loss_weight_age=0.33):
    """
    function compiles the specified model (classification or regression) with different loss_weights
    :param model: the generated model from the create_model function
    :param version: regression or classification
    :param loss_weight_face: weight of the face loss for the total loss
    :param loss_weight_mask: weight of the mask loss for the total loss
    :param loss_weight_age: weight of the age loss for the total loss
    :return: the compiled keras model
    """
    model.compile(optimizer='adam', loss={'face_detection': 'binary_crossentropy',
                                          'mask_detection': 'binary_crossentropy',
                                          'age_detection': tf.keras.losses.SparseCategoricalCrossentropy(
                                              ignore_class=-1) if version == "classification" else 'mse'},
                  loss_weights={'face_detection': loss_weight_face,
                                'mask_detection': loss_weight_mask,
                                'age_detection': loss_weight_age},
                  metrics={'face_detection': 'accuracy',
                           'mask_detection': 'accuracy',
                           'age_detection': 'accuracy' if version == "classification" else ['mse', 'mae']})
    return model


## create label for multitask

@tf.function
def get_weights(weights):
    '''
    converts the given weighs into a dictionary with a specific format tensorflow.dataset can understand
    :param weights: arrays with weights
    :return: formatted dictionary
    '''
    return {'face_detection': tf.reshape(tf.keras.backend.cast(weights["face_detection"], tf.keras.backend.floatx()), (-1, 1)),
            'mask_detection': tf.reshape(tf.keras.backend.cast(weights["mask_detection"], tf.keras.backend.floatx()), (-1, 1)),
            'age_detection': tf.reshape(tf.keras.backend.cast(weights["age_detection"], tf.keras.backend.floatx()), (-1, 1))}


def cluster_ages(x):
    '''
    cluster the given age into 10 years intervalls
    :param x: age of the person
    :return:  age clustered
    '''
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


@tf.function
def get_label(label):
    """
    formats the ground-truth values in a dictionary tensorflow.datasets can understand
    :param label: arrays of the ground-truth values of the picture for the given tasks
    :return: formatted dictionary
    """
    return {'face_detection': tf.reshape(tf.keras.backend.cast(label[0], tf.keras.backend.floatx()), (-1, 1)),
            'mask_detection': tf.reshape(tf.keras.backend.cast(label[1], tf.keras.backend.floatx()), (-1, 1)),
            'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1, 1))}


@tf.function
def decode_img(img_path):
    """
    function read image from filepath and format it into a tensor
    :param img_path: filepath of the image
    :return: decodes image as tensor
    """
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def process_path(file_path, labels):
    """
    function reads image from filesystem and returns it in specific format
    :param file_path: path of the image file
    :param labels: ground-truth values for the image
    :return: tupel of the image and the label dictionary
    """
    label = get_label(labels)
    img = decode_img(file_path)
    return img, label


def process_path_weighted(file_path, labels, sample_weights):
    '''
    function reads image from filesystem and returns it in specific format
    :param file_path: path of the image file
    :param labels: ground-truth values for the image
    :param sample_weights:
    :return: image, label dictionary and sample_weights for the labels
    '''
    label = get_label(labels)
    img = decode_img(file_path)
    weight = get_weights(sample_weights)
    return img, label, weight


def create_dataset(model_version, category, csv_path, weighted_regression=True):
    '''
    function creates dataset that can be used for training, validation and testing
    :param model_version: if the model is a regression or classification model
    :param category: possible choices: face,mask,age; if the dataset should only contain batches with specific labels (no missing values)
    :param csv_path: path to the csv file, which contains the information for the images
    :param weighted_regression: if sample weights for the regression model should be generated
    :return: tuple of the dataset and a pandas datatable with images information
    '''
    table_data = pd.read_csv(csv_path)
    if category == "mask":
        table_data = table_data[table_data["face"] == 1]
    elif category == "age":
        table_data = table_data[table_data["age"] >= 1]
    table_data = shuffle(table_data, random_state=123)
    if model_version == "regression":
        table_data['face_weights'] = 1
        table_data['mask_weights'] = table_data['face']
        table_data['age_weights'] = table_data["age"].apply(lambda x: 1 if x >= 1 else 0)
        if weighted_regression:
            dict_weighted = {"face_detection": np.array(table_data['face_weights']),
                         "mask_detection": np.array(table_data['mask_weights']),
                         "age_detection": np.array(table_data['age_weights'])}
            data = tf.data.Dataset.from_tensor_slices(
                (table_data["image_path"], table_data[["face", "mask", "age"]], dict_weighted))
            ds = data.map(process_path_weighted)
            ds = ds.batch(32)
            return ds, table_data
        else:
            data = tf.data.Dataset.from_tensor_slices(
                (table_data["image_path"], table_data[["face", "mask", "age"]]))
    elif model_version == "classification":
        table_data['age_clustered'] = table_data["age"].apply(cluster_ages)
        data = tf.data.Dataset.from_tensor_slices(
            (table_data["image_path"], table_data[["face", "mask", "age_clustered"]]))
    else:
        return
    ds = data.map(process_path)
    ds = ds.batch(32)
    return ds, table_data


def change_loss_function_while_training(version, path_to_train_csv, path_to_val_csv, alpha=0.25, dropout=0.2,
                                        epochs_face=10, epochs_mask=10, epochs_age=100, large_version=False, regularizer=False):
    '''
    function creates, compile and train a model with specific parameters, while changing the loss_weights during training.
    After training the model gets saved with a generated name to saved_model/Milestone3
    :param version: classification or regression model
    :param path_to_train_csv: path to the csv file containing information about the train images
    :param path_to_val_csv: path to the csv file containing information about the validation images
    :param alpha: alpah paramter of the feature extractor (mobilenetv1)
    :param dropout: dropout for the age task
    :param epochs_face: number of training epochs for the face task
    :param epochs_mask: number of training epochs for the mask task
    :param epochs_age: number of training epochs for the age task
    :param large_version: true or false: whether if the age head contains more dense and dropout layers or not
    :param regularizer: true or false: whether the dense layers in the age task contains l2 kernel regularization or not
    :return:
    '''
    model = create_model(version, alpha, dropout, large_version, regularizer)
    time = datetime.datetime.now().strftime("%Y%m%d-%H%M_")
    name = time + version + str(epochs_face) + "epochsface_" + str(epochs_mask) + "epochsmask_" + str(epochs_age) + "epochsage_" + str(alpha) + "alpha_" + str(dropout) + "dropout"
    if large_version:
        name += "_largeVersion"
    if regularizer:
        name += "_l2"

    for category in ["face", "mask", "age"]:
        model = compile_model(model,
                              version=version,
                              loss_weight_face=1 if category == "face" else 0,
                              loss_weight_mask=1 if category == "mask" else 0,
                              loss_weight_age=1 if category == "age" else 0)
        train_ds, _ = create_dataset(version, category, path_to_train_csv, weighted_regression=False)
        val_ds, _ = create_dataset(version, category, path_to_val_csv, weighted_regression=False)

        log_dir = "logs/fit/" + name + category + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        tf.debugging.set_log_device_placement(True)
        if category == "face":
            epochs = epochs_face
        elif category == "mask":
            epochs = epochs_mask
        elif category == "age":
            epochs = epochs_age
        model_history = model.fit(train_ds,
                                  epochs=epochs,
                                  validation_data=val_ds,
                                  callbacks=[tensorboard_callback])

    # save model
    model.save("saved_model/Milestone3/" + name)

