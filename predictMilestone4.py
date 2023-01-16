##
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import helper_tripletloss
import helper_contrastiveloss
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

##
# load model triplet loss
# create model
siamese_net = helper_tripletloss.create_model()
# reform it to siamese model
siamese_model = helper_tripletloss.SiameseModel(siamese_net)
siamese_model.compile(optimizer='adam',
                      loss=helper_tripletloss.triplet_loss,
                      metrics=["accuracy",
                               "precision",
                               helper_tripletloss.triplet_accuracy])
# load weights
load_status = siamese_model.load_weights(
    "saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain_utk/siamese_net")

##
# load model contrastive loss
model = tf.keras.models.load_model("saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_utk_pooling",
                                   custom_objects={"contrastive_loss": helper_contrastiveloss.contrastive_loss})


##
def generate_database(file_path, number_of_samples=4):
    """
    generate database with n random picks from each person
    :param file_path: path to train images
    :param number_of_samples: number of samples for each class
    :return: dataset of images, dataset of images as numpy array
    """
    # image preprocessing
    images = sorted([(str(file_path + "/" + f), str(f.split("_")[0])) for f in os.listdir(file_path)])
    images_df = pd.DataFrame(images, columns=["path", "class"])
    images_df = images_df[images_df["class"] != ".DS"]

    # generate database
    database = []
    for label, grouped_subframe in images_df.groupby("class"):
        random_rows = grouped_subframe.sample(number_of_samples, random_state=42)
        database.append(random_rows)

    # decode database as nupy array
    database_frame = pd.concat(database, ignore_index=True)
    database_list = [helper_tripletloss.decode_image(f) for f in database_frame["path"]]
    database_list = np.array(database_list)
    return database_frame, database_list


## take human generated database
PATH_TO_DATABASE = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/database"
images = sorted([(str(PATH_TO_DATABASE + "/" + f), str(f.split("_")[0])) for f in os.listdir(PATH_TO_DATABASE)])
images_db = pd.DataFrame(images, columns=["path", "class"])
images_db = images_db[images_db["class"] != ".DS"]
len(images_db.groupby("class").count())

database_list = [helper_tripletloss.decode_image(f) for f in images_db["path"]]
database_list = np.array(database_list)

## random generated database
images_db, database_list = generate_database("images/milestone4/train", 5)


##
def pred_image(name, image_path, images_db, database_list, use_triplet_loss=True):
    """
    predict one image with all images in database and plots top three suitable prediction
    :param use_triplet_loss: bool to choose model version: True -> triplet loss; False -> contrastive loss
    :param name: person name
    :param image_path: path to image
    :param images_db: dataset containing all images and classes of database as string
    :param database_list: dataset containing all images of database decoded as numpy array
    :return: -
    """
    # image preprocessing
    img = helper_tripletloss.decode_image(image_path)
    image_list = np.array([img] * len(images_db))
    image_list_1 = image_list[0:int(len(image_list) * 0.5)]
    image_list_2 = image_list[int(len(image_list) * 0.5):]

    # split database
    database_list_1 = database_list[0:int(len(database_list) * 0.5)]
    database_list_2 = database_list[int(len(database_list) * 0.5):]
    # predict image
    if use_triplet_loss:
        prediction_1 = siamese_model.predict([image_list_1, database_list_1, database_list_1])
        prediction_2 = siamese_model.predict([image_list_2, database_list_2, database_list_2])
        prediction = np.append(prediction_1[0], prediction_2[0])
        images_db["pred"] = prediction
    else:
        prediction = model.predict([image_list, database_list])
        images_db["pred"] = prediction
    # get best fitting predictions
    pred_class = images_db.groupby("class").apply(lambda x: x['pred'].sum() / len(x))
    pred_class_min = images_db.groupby("class").min()["pred"]
    top_min = pred_class_min.nsmallest(3)
    top = pred_class.nsmallest(3)
    # create plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(tf.keras.preprocessing.image.array_to_img(img))
    textstr = '\n'.join((
        '1. %s: %.2f' % (top_min.index[0], top_min[0]),
        '2. %s: %.2f' % (top_min.index[1], top_min[1]),
        '3. %s: %.2f' % (top_min.index[2], top_min[2]),
        '\n real: %s' % (name)
        # '1. \n mean: %s: %.2f \n min:  %s: %.2f' % (top4.index[0], top4[0], top4_min.index[0], top4_min[0]),
        # '2. \n mean: %s: %.2f \n min: %s: %.2f' % (top4.index[1], top4[1], top4_min.index[1], top4_min[1]),
        # '3. \n mean: %s: %.2f \n min: %s: %.2f' % (top4.index[2], top4[2], top4_min.index[2], top4_min[2])
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(240, 112, textstr, fontsize=14, bbox=props, verticalalignment="center")
    plt.show()


## choose image to predict
IMG_TO_PREDICT_PATH = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\test\Anushka Sharma_26.jpg"
name = "Anushka Sharma"
pred_image(name, IMG_TO_PREDICT_PATH, images_db, database_list)

## predict two images
IMAGE_PATH_1 = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\test\Akshay Kumar_8.jpg"
IMAGE_PATH_2 = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\train\Alexandra Daddario_0.jpg"
# IMAGE_PATH_3 = "/Users/tobias/Downloads/Bild.jpeg"
img1 = helper_tripletloss.decode_image(IMAGE_PATH_1)
img1 = tf.expand_dims(img1, axis=0)
img2 = helper_tripletloss.decode_image(IMAGE_PATH_2)
img2 = tf.expand_dims(img2, axis=0)
# img3 = decode_image(IMAGE_PATH_3)
# img3 = tf.expand_dims(img3, axis=0)
prediction = model.predict([img1, img2])


##
def evaluateTestset(threshold, images_db, database_list, path_to_testset, mac_os, use_triplet_loss, model):
    """
    evaluate model with triplet loss with test dataset
    :param use_triplet_loss: bool True -> using triplet loss; False -> using contrastive loss
    :param threshold: threshold to decide if prediction is valid
    :param images_db: dataset containing all images and classes of database as string
    :param database_list: dataset containing all images of database decoded as numpy array
    :param path_to_testset: path to test images
    :param mac_os: bool if using mac_os
    :param model: model used to predict image
    :return: accuracy
    """
    if mac_os:
        images = sorted([(str(path_to_testset + "/" + f), str(f.split("_")[0])) for f in os.listdir(path_to_testset)])
    else:
        images = sorted([(str(path_to_testset + "\\" + f), str(f.split("_")[0])) for f in os.listdir(path_to_testset)])

    images_test = pd.DataFrame(images, columns=["path", "class"])
    pred_sum = 0
    count = 0
    for index, image in images_test.iterrows():
        # preprocessing image
        img = helper_tripletloss.decode_image(image["path"])
        image_list = np.array([img] * len(images_db))
        # generate prediction
        if use_triplet_loss:
            prediction = model.predict([image_list, database_list, database_list])
            images_db["pred"] = prediction[0]
        else:
            prediction = model.predict([image_list, database_list])
            images_db["pred"] = prediction
        # get best fitting
        pred_class_min = images_db.groupby("class").min()["pred"]
        top1_min = pred_class_min.nsmallest(1)
        # person in database
        if top1_min[0] <= threshold and top1_min.index[0] == image["class"]:
            pred_sum += 1
        # image from utk (person not in database)
        elif top1_min[0] > threshold and len(image["class"]) <= 3:
            pred_sum += 1
        count += 1
        print(str(count) + " of " + str(images_test["path"].count()))
    # calculate accuracy
    accuracy = pred_sum / images_test["path"].count()
    return accuracy


## Evaluate multiple variants
# Test database size
PATH_TO_TESTSET = "images/milestone4/test"
load_status = siamese_model.load_weights("saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain/siamese_net")
predictions_triplet = []
for t in [8]:
    images_db, database_list = generate_database(
        "C:\\Users\\Svea Worms\\PycharmProjects\\Face-Detection\\images\\milestone4\\train", t)
    pred = evaluateTestset(0.2, images_db, database_list, PATH_TO_TESTSET, True, True, siamese_model)
    predictions_triplet.append([t, pred])
with open('evaluation/Milestone4/prediction_databasesize_triplet.json', 'w') as file:
    # write the list to the file in json format
    json.dump(predictions_triplet, file)

PATH_TO_TESTSET = "images/milestone4/test"
load_status = siamese_model.load_weights(
    "saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain_utk/siamese_net")
predictions_triplet_utk = []
for t in [8]:
    images_db, database_list = generate_database(
        "C:\\Users\\Svea Worms\\PycharmProjects\\Face-Detection\\images\\milestone4\\train", t)
    pred = evaluateTestset(0.6, images_db, database_list, PATH_TO_TESTSET, True, True, siamese_model)
    predictions_triplet_utk.append([t, pred])

with open('evaluation/Milestone4/prediction_databasesize_triplet_utk.json', 'w') as file:
    # write the list to the file in json format
    json.dump(predictions_triplet_utk, file)
##
PATH_TO_TESTSET = "images/milestone4/test"
PATH_TO_MODEL2 = "saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_utk_pooling"
model2 = tf.keras.models.load_model(PATH_TO_MODEL2,
                                    custom_objects={"contrastive_loss": helper_contrastiveloss.contrastive_loss})
prediction_contrastive_utk = []
for t in [8]:
    images_db, database_list = generate_database(
        "C:\\Users\\Svea Worms\\PycharmProjects\\Face-Detection\\images\\milestone4\\train", t)
    pred = evaluateTestset(0.2, images_db, database_list, PATH_TO_TESTSET, True, False, model2)
    prediction_contrastive_utk.append([t, pred])

with open('evaluation/Milestone4/prediction_databasesize_contrastive_utk.json', 'w') as file:
    # write the list to the file in json format
    json.dump(prediction_contrastive_utk, file)
##
PATH_TO_MODEL2 = "saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_pooling"
model2 = tf.keras.models.load_model(PATH_TO_MODEL2,
                                    custom_objects={"contrastive_loss": helper_contrastiveloss.contrastive_loss})
prediction_contrastive = []
for t in [1, 3, 8]:
    images_db, database_list = generate_database(
        "C:\\Users\\Svea Worms\\PycharmProjects\\Face-Detection\\images\\milestone4\\train", t)
    pred = evaluateTestset(0.2, images_db, database_list, PATH_TO_TESTSET, True, False, model2)
    prediction_contrastive.append([t, pred])

with open('evaluation/Milestone4/prediction_databasesize_contrastive.json', 'w') as file:
    # write the list to the file in json format
    json.dump(prediction_contrastive, file)
