import tensorflow as tf
import helper


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

