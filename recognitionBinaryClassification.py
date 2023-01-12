##
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import helper
import os
import random

##
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

##
def create_model(alpha=1, debug=False, mode="flatten"):
    #TODO: train with global average pooling and with flatten layer
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
    #merge_layer = tf.keras.layers.Lambda(euclidean_distance)([encoded_l, encoded_r])
    #normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    #prediction = tf.keras.layers.Dense(1, activation="sigmoid")(merge_layer)
    siamese_net = tf.keras.Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net, feature_model


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean((1 - y_true) * square_pred + y_true * margin_square)


def compile_model(model):
    model.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])
    return model


## CREATE THE DATASET

def decode_image(img_path):
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    #img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def preprocess_triplets(anchor, positive, negative):
    return decode_image(anchor), decode_image(positive), decode_image(negative)


def preprocess_binary(picture1, picture2):
    return decode_image(picture1), decode_image(picture2)

def preprocess_binary_array(filepath1, filepath2, label):
    return {"input_left": decode_image(filepath1), "input_right": decode_image(filepath2)}, label

image_path = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\train"
#image_path = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/archive/Faces/Faces"
#images = sorted([str(image_path +  "/" +  f) for f in os.listdir(image_path)])

images = sorted([(str(image_path +  "/" +  f), str(f.split("_")[0])) for f in os.listdir(image_path)])

images_df = pd.DataFrame(images,columns=["path","class"])

len(images_df.groupby("class").count())

# train test split durchführen

##


def make_pairs(image_path, image_class, num=22, negative_path="/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/utkcropped", num_random_img=10):

    #array with names of all class labels
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
        first_image_class = image_class[idx1] #label for the first image

        #add picture with itself
        pairs += [[first_image_path, first_image_path]]
        labels += [0]

        #find index
        first_image_class_idx = np.where(np.array(all_class_labels) == first_image_class)[0][0]
        # take x random image from pictures with the same class
        idx_list = random.sample(digit_indices[first_image_class_idx][0].tolist(), k=num)
        for sec_image_idx in idx_list:
            sec_image_path = image_path[sec_image_idx]
            pairs += [[first_image_path, sec_image_path]]
            labels += [0] # zero because there are the same people

        # add a non-matching example

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


## generate tensorflow dataset
utk_cropped_path = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\utkCropped\utkcropped"
pairs = make_pairs(images_df["path"],images_df["class"], negative_path=utk_cropped_path, num_random_img=0)
##
np.random.RandomState(seed=32).shuffle(pairs[0])
np.random.RandomState(seed=32).shuffle(pairs[1])

train = pairs[0][0:int(len(pairs[0]) * 0.75)]
val = pairs[0][int(len(pairs[0]) * 0.75):]

train_label = pairs[1][0:int(len(pairs[1]) * 0.75)]
val_label = pairs[1][int(len(pairs[1]) * 0.75):]

data_train = tf.data.Dataset.from_tensor_slices((train[:, 0], train[:, 1], train_label))
data_val = tf.data.Dataset.from_tensor_slices((val[:, 0], val[:, 1], val_label))

ds_train = data_train.map(preprocess_binary_array)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(8)

ds_val = data_val.map(preprocess_binary_array)
ds_val = ds_val.batch(32)
ds_val = ds_val.prefetch(8)


## train the model
EPOCHS = 15
BATCH_SIZE = 32
model, feature_generator = create_model(mode="pooling")
model = compile_model(model)
model.summary()
##
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)
##
model.save("saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_pooling")

## visualize one sample

def visualize(pic, label):
    """Visualize one triplet from the supplied batch."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(2, 1))
    axs = fig.subplots(2, 1)
    print()
    show(axs[0], tf.keras.preprocessing.image.array_to_img(pic["input_left"][0]))
    show(axs[1], tf.keras.preprocessing.image.array_to_img(pic["input_right"][0]))


##

