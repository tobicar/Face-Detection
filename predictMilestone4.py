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


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean((1 - y_true) * square_pred + y_true * margin_square)


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
##
PATH_TO_DATABASE = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/database"
PATH_TO_MODEL = "saved_model/Milestone4/binaryClassification_10epochs_alpha1"
PATH_TO_MODEL2 = "saved_model/Milestone4/binaryClassification_10epochs_alpha1_v2"


model = tf.keras.models.load_model(PATH_TO_MODEL, custom_objects={"contrastive_loss": contrastive_loss})
model2 = tf.keras.models.load_model(PATH_TO_MODEL2, custom_objects={"contrastive_loss": contrastive_loss})
##
images = sorted([(str(PATH_TO_DATABASE +  "/" +  f), str(f.split("_")[0])) for f in os.listdir(PATH_TO_DATABASE)])

images_df = pd.DataFrame(images,columns=["path","class"])

images_df = images_df[images_df["class"] != ".DS"]

len(images_df.groupby("class").count())

database_list = [decode_image(f) for f in images_df["path"]]
database_list = np.array(database_list)

## choose image to predict
IMG_TO_PREDICT = "/Users/tobias/Downloads/Bild.jpeg"
img = decode_image(IMG_TO_PREDICT)
image_list = np.array([img]*len(images_df))
##
prediction = model2.predict([image_list, database_list])
images_df["pred"] = prediction
pred_class = images_df.groupby("class").apply(lambda x: x['pred'].sum()/len(x))
top4 = pred_class.nsmallest(4)

fig, ax = plt.subplots()
ax.imshow(tf.keras.preprocessing.image.array_to_img(img))
textstr = '\n'.join((
                r'1. %s: %.2f' % (top4.index[0], top4[0],),
                r'2. %s: %.2f' % (top4.index[1], top4[1],),
                r'3. %s: %.2f' % (top4.index[2], top4[2],),
                r'4. %s: %.2f' % (top4.index[3], top4[3],)
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(240, 112, textstr, fontsize=14, bbox=props, verticalalignment="center")
plt.show()



## predict two images
IMAGE_PATH_1 = "/Users/tobias/Downloads/zac.jpg"
IMAGE_PATH_2 = "/Users/tobias/Downloads/zac.jpg"
img1 = decode_image(IMAGE_PATH_1)
img1 = tf.expand_dims(img1, axis=0)
img2 = decode_image(IMAGE_PATH_2)
img2 = tf.expand_dims(img2, axis=0)
prediction = model2.predict([img1, img2])


## generate database with n random picks from each person
def generate_database(file_path):
    pass






