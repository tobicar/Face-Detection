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
def create_model(alpha=0.25, debug=False):
    #TODO: train with global average pooling and with flatten layer
    left_input = tf.keras.Input(shape=(224, 224, 3), name='input_left')
    right_input = tf.keras.Input(shape=(224, 224, 3), name='input_right')
    input = tf.keras.Input(shape=(224, 224, 3), name='input')
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
    model_pretrained.trainable = False
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(input)
    feature_generator = model_pretrained(feature_extractor, training=False)
    feature_generator = tf.keras.layers.GlobalAveragePooling2D()(feature_generator)
    #feature_generator = tf.keras.layers.Flatten()(feature_generator)
    feature_generator = tf.keras.layers.Dropout(0.2)(feature_generator)
    feature_generator = tf.keras.layers.BatchNormalization()(feature_generator)
    feature_generator = tf.keras.layers.Dense(128, activation='relu')(feature_generator)

    feature_model = tf.keras.Model(input, feature_generator, name="feature_generator")
    if debug:
        feature_model.summary()
    encoded_l = feature_model(left_input)
    encoded_r = feature_model(right_input)
    subtracted = tf.keras.layers.Subtract()([encoded_l, encoded_r])
    l1_layer = tf.keras.layers.Lambda(lambda x: abs(x))(subtracted)
    prediction = tf.keras.layers.Dense(1, activation="sigmoid")(l1_layer)
    siamese_net = tf.keras.Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


def contrastive_loss(y_true, y_pred, margin=1):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


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

image_path = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/archive/Faces/Faces"
#images = sorted([str(image_path +  "/" +  f) for f in os.listdir(image_path)])

images = sorted([(str(image_path +  "/" +  f), str(f.split("_")[0])) for f in os.listdir(image_path)])

images_df = pd.DataFrame(images,columns=["path","class"])

len(images_df.groupby("class").count())

# train test split durchführen

##
def make_pairs(x,y):

    num_classes = y.unique().tolist()
    digit_indices = [(np.where(y == i)[0],i) for i in num_classes]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        label1_idx = np.where(np.array(num_classes) == label1)[0][0]
        idx2 = random.choice(digit_indices[label1_idx][0])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        #label2 = random.randint(0, num_classes - 1)
        label2 = random.choice(num_classes)
        while label2 == label1:
            #label2 = random.randint(0, num_classes - 1)
            label2 = random.choice(num_classes)

        label2_idx = np.where(np.array(num_classes) == label2)[0][0]
        idx2 = random.choice(digit_indices[label2_idx][0])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

## generate tensorflow dataset
pairs = make_pairs(images_df["path"],images_df["class"])
data = tf.data.Dataset.from_tensor_slices((pairs[0][:,0],pairs[0][:,1],pairs[1]))
ds = data.map(preprocess_binary_array)
ds = ds.batch(32)


## train the model
EPOCHS = 10
BATCH_SIZE = 32
model = create_model()
model = compile_model(model)
model.summary()
history = model.fit(ds, epochs=EPOCHS)
