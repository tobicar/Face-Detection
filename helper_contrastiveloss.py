##
import tensorflow as tf
import helper
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os


def create_model(alpha=1, debug=False, mode="flatten"):
    """
    creates the model structure used for contrastive loss training
    :param alpha: alpha parameter of mobileNet
    :param debug: parameter for printing out the embedding summary
    :param mode: controls if the model uses a Pooling or Flatten Layer after the mobileNet
    :return: the generated tensorflow model
    """
    left_input = tf.keras.Input(shape=(224, 224, 3), name='input_left')
    right_input = tf.keras.Input(shape=(224, 224, 3), name='input_right')
    input = tf.keras.Input(shape=(224, 224, 3), name='input')
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
    model_pretrained.trainable = False
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(input)
    pretrained_head = model_pretrained(feature_extractor, training=False)
    if mode.lower() == "flatten":
        feature_generator = tf.keras.layers.Flatten()(pretrained_head)
    else:
        feature_generator = tf.keras.layers.GlobalAveragePooling2D()(pretrained_head)
    feature_generator = tf.keras.layers.BatchNormalization()(feature_generator)
    feature_generator = tf.keras.layers.Dropout(0.4)(feature_generator)
    feature_generator = tf.keras.layers.Dense(256, activation="relu")(feature_generator)
    feature_generator = tf.keras.layers.BatchNormalization()(feature_generator)
    feature_generator = tf.keras.layers.Dropout(0.4)(feature_generator)
    output = tf.keras.layers.Dense(128)(feature_generator)

    feature_model = tf.keras.Model(input, output, name="feature_generator")
    if debug:
        feature_model.summary()
    encoded_l = feature_model(left_input)
    encoded_r = feature_model(right_input)
    subtracted = tf.keras.layers.Subtract()([encoded_l, encoded_r])
    l1_layer = tf.keras.layers.Lambda(lambda x: abs(x))(subtracted)
    prediction = tf.keras.layers.Dense(1, activation="sigmoid")(l1_layer)
    siamese_net = tf.keras.Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net, feature_model


def contrastive_loss(y_true, y_pred, margin=1):
    """
    calculate the contrastive loss for training
    :param y_true: true labels
    :param y_pred: predicted labels
    :param margin: margin for the maximal contrastive loss value
    :return: calculated contrastive loss
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean((1 - y_true) * square_pred + y_true * margin_square)


def compile_model(model):
    """
    Compile the given model with contrastive loss and the Optimizer Adam
    :param model: tensorflow model
    :return: compiled model
    """
    model.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])
    return model


def decode_image(img_path):
    """
    loads the file from the image_path
    :param img_path: path to image
    :return: images as tf EagerTensor
    """
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def preprocess_binary_array(filepath1, filepath2, label):
    """
    Preprocessed the given file_paths and generates tuples with images and labels
    :param filepath1: path of the first image
    :param filepath2: path of the second image
    :param label: the truth label of the image;
    0: persons on the image are equal
    1: persons on the image are unequal
    :return: tuple: dictionary with the two decoded images,
    """
    return {"input_left": decode_image(filepath1), "input_right": decode_image(filepath2)}, label


def visualize(pic):
    """
    Visualize on pair
    :param pic: dictionary with the two images of the style:
    {"input_left": pic1, "input_right": pic2}
    :return:
    """

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(2, 1))
    axs = fig.subplots(2, 1)
    print()
    show(axs[0], tf.keras.preprocessing.image.array_to_img(pic["input_left"][0]))
    show(axs[1], tf.keras.preprocessing.image.array_to_img(pic["input_right"][0]))


def make_pairs(image_path, image_class, num=22, negative_path="/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/utkcropped", num_random_img=10):
    """
    Generates Pairs of the given image List which are used for training the model
    :param image_path: Pandas Series with all image_paths of images in the database
    :param image_class: Pandas Series with the Names of the images given in image_path
    :param num: Number of positive pictures Pairs used in every iteration
    :param negative_path: Path to image folder containing unknown utk cropped images
    :param num_random_img: number of random utk cropped images used in every iteration
    :return: tuple of (list1, list2). list1: np.array of image_paths; list2: array of matching labels
    """

    # array with names of all class labels
    all_class_labels = image_class.unique().tolist()
    # list with arrays of all indexes of the pictures belonging to one class
    digit_indices = [(np.where(image_class == i)[0], i) for i in all_class_labels]

    pairs = []
    labels = []

    # load random negative face pictures (utk_cropped)
    random_images = sorted([str(negative_path + "/" + f) for f in os.listdir(negative_path)])
    random_images_df = pd.DataFrame(random_images, columns=["path"])
    random_images_df = random_images_df[random_images_df["path"] != "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/utkcropped/.DS_Store"]
    random_images_df = random_images_df.reset_index(drop=True)

    for idx1 in range(len(image_path)):
        # add a matching example
        first_image_path = image_path[idx1]
        # label for the first image
        first_image_class = image_class[idx1]

        # add picture with itself
        pairs += [[first_image_path, first_image_path]]
        labels += [0]

        # find index
        first_image_class_idx = np.where(np.array(all_class_labels) == first_image_class)[0][0]
        # take x random image from pictures with the same class
        idx_list = random.sample(digit_indices[first_image_class_idx][0].tolist(), k=num)
        for sec_image_idx in idx_list:
            sec_image_path = image_path[sec_image_idx]
            pairs += [[first_image_path, sec_image_path]]
            # zero because there are the same people
            labels += [0]

        # add on-matching examples

        # make a copy of the class label list
        sec_class_list = all_class_labels.copy()
        # remove the label of the first image from the list
        sec_class_list.remove(first_image_class)
        # choose x random class labels from the list of people
        class_label2_list = random.sample(sec_class_list, k=(num-num_random_img))
        # for each class choose one random picture
        for class_label2 in class_label2_list:
            class_label2_idx = np.where(np.array(all_class_labels) == class_label2)[0][0]
            idx2 = random.choice(digit_indices[class_label2_idx][0])
            x2 = image_path[idx2]

            pairs += [[first_image_path, x2]]
            labels += [1]

        x_random_negatives = random.sample(random_images_df["path"].to_list(), k=num_random_img)
        for random_pic in x_random_negatives:
            pairs += [[first_image_path, random_pic]]
            labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")
