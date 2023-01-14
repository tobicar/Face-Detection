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
import json
##


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean((1 - y_true) * square_pred + y_true * margin_square)


def triplet_loss(y_true, y_pred):
    margin = 1
    ap_distance = y_pred[:, 0, :]
    an_distance = y_pred[:, 1, :]
    loss = ap_distance - an_distance
    loss = tf.maximum(loss + margin, 0.0)
    return loss


def triplet_accuracy(y_true, y_pred):
    # Calculate the distance between the anchor point and the positive point
    pos_dist = tf.reduce_sum(tf.square(y_pred[:, 0, :] - y_pred[:, 1, :]), axis=1)
    # Calculate the distance between the anchor point and the negative point
    neg_dist = tf.reduce_sum(tf.square(y_pred[:, 0, :] - y_pred[:, 2, :]), axis=1)
    # Calculate the accuracy as the percentage of triplets for which the positive distance is less than the negative distance
    accuracy = tf.reduce_mean(tf.cast(pos_dist < neg_dist, dtype=tf.float32))
    return accuracy

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

## generate database with n random picks from each person
def generate_database(file_path, number_of_samples=4):
    images = sorted([(str(file_path + "/" + f), str(f.split("_")[0])) for f in os.listdir(file_path)])
    images_df = pd.DataFrame(images, columns=["path", "class"])
    images_df = images_df[images_df["class"] != ".DS"]

    database = []
    for label,grouped_subframe in images_df.groupby("class"):
        random_rows = grouped_subframe.sample(number_of_samples, random_state=42)
        database.append(random_rows)

    database_frame = pd.concat(database, ignore_index=True)
    database_list = [decode_image(f) for f in database_frame["path"]]
    database_list = np.array(database_list)
    return database_frame, database_list
##
PATH_TO_DATABASE = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/database"
#PATH_TO_MODEL = "saved_model/Milestone4/binaryClassification_10epochs_alpha1"
#PATH_TO_MODEL2 = "saved_model/Milestone4/binaryClassification_20epochs_alpha1_flatten"
PATH_TO_MODEL_TRIPLET = "saved_model/Milestone4/tripletLoss_new"


#model = tf.keras.models.load_model(PATH_TO_MODEL, custom_objects={"contrastive_loss": contrastive_loss})
#model2 = tf.keras.models.load_model(PATH_TO_MODEL2, custom_objects={"contrastive_loss": contrastive_loss})#
model_triplet = tf.keras.models.load_model(PATH_TO_MODEL_TRIPLET, custom_objects={"triplet_loss": triplet_loss, "triplet_accuracy": triplet_accuracy})
## take human generated database
images = sorted([(str(PATH_TO_DATABASE +  "/" +  f), str(f.split("_")[0])) for f in os.listdir(PATH_TO_DATABASE)])

images_db = pd.DataFrame(images,columns=["path","class"])

images_db = images_db[images_db["class"] != ".DS"]

len(images_db.groupby("class").count())

database_list = [decode_image(f) for f in images_db["path"]]
database_list = np.array(database_list)


## random generated database
images_db, database_list = generate_database("/Users/tobias/PycharmProjects/Face-Detection/images/milestone4/train", 5)
#images_db, database_list = generate_database("C:\\Users\\Svea Worms\\PycharmProjects\\Face-Detection\\images\\rawdata4", 5)
## choose image to predict
IMG_TO_PREDICT_PATH = "/Users/tobias/Downloads/anushka.jpg"
#IMG_TO_PREDICT_PATH = r"C:\Users\Svea Worms\Downloads\predictions\mensch1.jpg"
#IMG_TO_PREDICT_PATH = r"C:\Users\Svea Worms\Downloads\predictions\31_0_0_20170104201726242.jpg.chip.jpg"
name = "Alia Bhatt"
img = decode_image(IMG_TO_PREDICT_PATH)
image_list = np.array([img]*len(images_db))
image_list_1 = image_list[0:int(len(image_list)*0.5)]
image_list_2 = image_list[int(len(image_list)*0.5):]
##
database_list_1 = database_list[0:int(len(database_list)*0.5)]
database_list_2 = database_list[int(len(database_list)*0.5):]

TRIPLET_LOSS_MODEL = True
if TRIPLET_LOSS_MODEL:
    prediction_1 = siamese_model.predict([image_list_1, database_list_1, database_list_1])
    prediction_2 = siamese_model.predict([image_list_2, database_list_2, database_list_2])
    prediction = np.append(prediction_1[0], prediction_2[0])
    images_db["pred"] = prediction
else:
    prediction = model2.predict([image_list, database_list])
    images_db["pred"] = prediction
pred_class = images_db.groupby("class").apply(lambda x: x['pred'].sum()/len(x))
pred_class_min = images_db.groupby("class").min()["pred"]
top4_min = pred_class_min.nsmallest(3)
top4 = pred_class.nsmallest(3)

fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(tf.keras.preprocessing.image.array_to_img(img))
textstr = '\n'.join((
                '1. %s: %.2f' % (top4_min.index[0], top4_min[0]),
                '2. %s: %.2f' % (top4_min.index[1], top4_min[1]),
                '3. %s: %.2f' % (top4_min.index[2], top4_min[2]),
                '\n real: %s' % (name)
                #'1. \n mean: %s: %.2f \n min:  %s: %.2f' % (top4.index[0], top4[0], top4_min.index[0], top4_min[0]),
                #'2. \n mean: %s: %.2f \n min: %s: %.2f' % (top4.index[1], top4[1], top4_min.index[1], top4_min[1]),
                #'3. \n mean: %s: %.2f \n min: %s: %.2f' % (top4.index[2], top4[2], top4_min.index[2], top4_min[2])
))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(240, 112, textstr, fontsize=14, bbox=props, verticalalignment="center")
plt.show()



## predict two images
#IMAGE_PATH_1 = "/Users/tobias/Downloads/zac.jpg"
#IMAGE_PATH_2 = "/Users/tobias/Downloads/zac.jpg"
IMAGE_PATH_1 = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\test\Akshay Kumar_8.jpg"
IMAGE_PATH_2 = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\train\Alexandra Daddario_0.jpg"
#IMAGE_PATH_3 = "/Users/tobias/Downloads/Bild.jpeg"
img1 = decode_image(IMAGE_PATH_1)
img1 = tf.expand_dims(img1, axis=0)
img2 = decode_image(IMAGE_PATH_2)
img2 = tf.expand_dims(img2, axis=0)
#img3 = decode_image(IMAGE_PATH_3)
#img3 = tf.expand_dims(img3, axis=0)
prediction = model.predict([img1, img2])
print(prediction)
##
def evaluateTestsetTripletLoss(threshold, images_db, database_list, path_to_testset, mac_os, siamese_model):
    if mac_os:
        images = sorted([(str(path_to_testset + "/" + f), str(f.split("_")[0])) for f in os.listdir(path_to_testset)])
    else:
        images = sorted([(str(path_to_testset + "\\" + f), str(f.split("_")[0])) for f in os.listdir(path_to_testset)])

    images_test = pd.DataFrame(images, columns=["path", "class"])
    pred_sum = 0
    for index, image in images_test.iterrows():
        img = decode_image(image["path"])
        image_list = np.array([img] * len(images_db))
        prediction = siamese_model.predict([image_list, database_list, database_list])
        images_db["pred"] = prediction[0]
        pred_class_min = images_db.groupby("class").min()["pred"]
        top1_min = pred_class_min.nsmallest(1)
        if top1_min[0] <= threshold and top1_min.index[0] == image["class"]:
            pred_sum += 1

    accuracy = pred_sum/images_test["path"].count()
    return accuracy

##
def evaluateTestsetContrastiveLoss(threshold, images_db, database_list, path_to_testset, mac_os, model):
    if mac_os:
        images = sorted([(str(path_to_testset + "/" + f), str(f.split("_")[0])) for f in os.listdir(path_to_testset)])
    else:
        images = sorted([(str(path_to_testset + "\\" + f), str(f.split("_")[0])) for f in os.listdir(path_to_testset)])

    images_test = pd.DataFrame(images, columns=["path", "class"])
    pred_sum = 0
    for index, image in images_test.iterrows():
        img = decode_image(image["path"])
        image_list = np.array([img] * len(images_db))
        prediction = model.predict([image_list, database_list])
        images_db["pred"] = prediction
        # pred_class = images_db.groupby("class").apply(lambda x: x['pred'].sum() / len(x))
        pred_class_min = images_db.groupby("class").min()["pred"]
        top1_min = pred_class_min.nsmallest(1)
        if top1_min[0] <= threshold and top1_min.index[0] == image["class"]:
            pred_sum += 1

    accuracy = pred_sum / images_test["path"].count()
    return accuracy

##
PATH_TO_TESTSET = "images/milestone4/test"
load_status = siamese_model.load_weights("saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain_utk/siamese_net")
predictions_triplet_utk = []
for t in [0.2, 0.4, 0.6]:
    #pred = evaluateTestsetContrastiveLoss(t, images_db, database_list, r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\test", False, model)
    pred = evaluateTestsetTripletLoss(t, images_db, database_list, PATH_TO_TESTSET, True, siamese_model)
    predictions_triplet_utk.append([t, pred])

with open('evaluation/Milestone4/prediction_triplet_utk.json', 'w') as file:
    # write the list to the file in json format
    json.dump(predictions_triplet_utk, file)

load_status = siamese_model.load_weights("saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain/siamese_net")
predictions_triplet = []
for t in [0.2, 0.4, 0.6]:
    #pred = evaluateTestsetContrastiveLoss(t, images_db, database_list, r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\test", False, model)
    pred = evaluateTestsetTripletLoss(t, images_db, database_list, PATH_TO_TESTSET, True, siamese_model)
    predictions_triplet.append([t, pred])
with open('evaluation/Milestone4/prediction_triplet.json', 'w') as file:
    # write the list to the file in json format
    json.dump(predictions_triplet, file)

PATH_TO_MODEL2 = "saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_utk_pooling"
model2 = tf.keras.models.load_model(PATH_TO_MODEL2, custom_objects={"contrastive_loss": contrastive_loss})
prediction_contrastive_utk = []
for t in [0.2, 0.4, 0.6]:
    pred = evaluateTestsetContrastiveLoss(t, images_db, database_list, PATH_TO_TESTSET, True, model2)
    prediction_contrastive_utk.append([t, pred])
with open('evaluation/Milestone4/prediction_contrastive_utk.json', 'w') as file:
    # write the list to the file in json format
    json.dump(prediction_contrastive_utk, file)

PATH_TO_MODEL2 = "saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_pooling"
model2 = tf.keras.models.load_model(PATH_TO_MODEL2, custom_objects={"contrastive_loss": contrastive_loss})
prediction_contrastive = []
for t in [0.2, 0.4, 0.6]:
    pred = evaluateTestsetContrastiveLoss(t, images_db, database_list, PATH_TO_TESTSET, True, model2)
    prediction_contrastive.append([t, pred])

with open('evaluation/Milestone4/prediction_contrastive.json', 'w') as file:
    # write the list to the file in json format
    json.dump(prediction_contrastive, file)

