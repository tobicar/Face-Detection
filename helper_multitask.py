import tensorflow as tf
import helper
from sklearn.utils import shuffle
import numpy as np
import datetime
import pandas as pd

def add_face(x):
    """

    :param x:
    :return:
    """
    greater = tf.keras.backend.greater_equal(x, 0.5)  # will return boolean values
    greater = tf.keras.backend.cast(greater, dtype=tf.keras.backend.floatx())  # will convert bool to 0 and 1
    return greater


def create_model(version, alpha=0.25, dropout=0.2, large_version=False, regularizer=False):
    """

    :param version:
    :param alpha:
    :param dropout:
    :param large_version:
    :param regularizer:
    :return:
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
    feature_extractor_age = tf.keras.layers.Dense(1)(feature_extractor_age)

    if version == "classification":
        age_detection = tf.keras.layers.Dense(10, activation="softmax", name="age_detection")(feature_extractor_age)
    elif version == "regression":
        face_detection_ground_truth = tf.keras.layers.Lambda(add_face)(face_detection)
        age_detection = tf.keras.layers.multiply([feature_extractor_age, face_detection_ground_truth],
                                                 name="age_detection")
    else:
        return None

    model = tf.keras.Model(inputs=inputs, outputs=[face_detection, mask_detection, age_detection])
    return model


def compile_model(model, version, loss_weight_face=0.33, loss_weight_mask=0.33, loss_weight_age=0.33):
    """

    :param model:
    :param version:
    :param loss_weight_face:
    :param loss_weight_mask:
    :param loss_weight_age:
    :return:
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
def get_label(label):
    """

    :param label:
    :return:
    """
    return {'face_detection': tf.reshape(tf.keras.backend.cast(label[0], tf.keras.backend.floatx()), (-1, 1)),
            'mask_detection': tf.reshape(tf.keras.backend.cast(label[1], tf.keras.backend.floatx()), (-1, 1)),
            'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1, 1))}


@tf.function
def decode_img(img_path):
    """

    :param img_path:
    :return:
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

    :param file_path:
    :param labels:
    :return:
    """
    label = get_label(labels)
    img = decode_img(file_path)
    return img, label


def create_categorical_dataset(model_version, category, csv_path):
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
        dict_weighted = {"face_detection": np.array(table_data['face_weights']),
                         "mask_detection": np.array(table_data['mask_weights']),
                         "age_detection": np.array(table_data['age_weights'])}
        data = tf.data.Dataset.from_tensor_slices(
            (table_data["image_path"], table_data[["face", "mask", "age"]], dict_weighted))
    elif model_version == "classification":
        data = tf.data.Dataset.from_tensor_slices(
            (table_data["image_path"], table_data[["face", "mask", "age_clustered"]]))
    else:
        return
    ds = data.map(process_path)
    ds = ds.batch(32)
    return ds, table_data


def change_loss_function_while_training(version, path_to_train_csv, path_to_val_csv, alpha=0.25, dropout=0.2,
                                        epochs=100, large_version=False, regularizer=False):
    model = create_model(version, alpha, dropout, large_version, regularizer)

    for category in ["face", "mask", "age"]:
        model = compile_model(model,
                              version=version,
                              loss_weight_face=1 if category == "face" else 0,
                              loss_weight_mask=1 if category == "mask" else 0,
                              loss_weight_age=1 if category == "age" else 0)
        train_ds = create_categorical_dataset(version, category, path_to_train_csv)
        val_ds = create_categorical_dataset(version, category, path_to_val_csv)

        name = version + str(epochs) + "epochs_" + str(alpha) + "alpha_" + str(dropout) + "dropout_ValOnlyAge"
        if large_version:
            name += "_largeVersion"
        log_dir = "logs/fit/" + name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        tf.debugging.set_log_device_placement(True)
        model_history = model.fit(train_ds,
                                  epochs=epochs,
                                  validation_data=val_ds,
                                  callbacks=[tensorboard_callback])

    # save model
    model.save("saved_model/Milestone3/" + name)

