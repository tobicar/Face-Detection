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
PATH_TO_MODEL = ""


model = tf.keras.models.load_model(PATH_TO_MODEL)
##
images = sorted([(str(PATH_TO_DATABASE +  "/" +  f), str(f.split("_")[0])) for f in os.listdir(PATH_TO_DATABASE)])

images_df = pd.DataFrame(images,columns=["path","class"])

images_df = images_df[images_df["class"] != ".DS"]

len(images_df.groupby("class").count())

database_list = [decode_image(f) for f in images_df["path"]]
database_list = np.array(database_list)

## choose image to predict
IMG_TO_PREDICT = "/Users/tobias/Downloads/tom-cruise-2.jpg"
img = decode_image(IMG_TO_PREDICT)
image_list = np.array([img]*len(images_df))
##
prediction = model.predict([image_list, database_list])
images_df["pred"] = prediction
pred_class = images_df.groupby("class").apply(lambda x: x['pred'].sum()/len(x))
top4 = pred_class.nsmallest(4)

fig, ax = plt.subplots()
ax.imshow(tf.keras.preprocessing.image.array_to_img(img))
textstr = '\n'.join((
                r'1. %s: %.2f' % (top4.index[0], top4[0],),
                r'2. %s: %.2f' % (top4.index[1], top4[1],),
                r'3. %s: %.2f' % (top4.index[2], top4[2],),
                r'4. %s: %.2f' % (top4.index[3], top4[1],)
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(240, 112, textstr, fontsize=14, bbox=props, verticalalignment="center")
plt.show()


