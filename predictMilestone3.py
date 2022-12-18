##
# OpenFileDialog and predict one Image
import helper
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import helper_multitask
##
PATH_TO_MODEL = "saved_model/Milestone3/20221210-1618_classification10epochsface_10epochsmask_50epochsage_0.25alpha_0.5dropout_categoricalLoss_l2"
PATH_TO_MODEL_REGRESSION = "saved_model/Milestone3/20221211-2234_regression10epochsface_10epochsmask_50epochsage_0.25alpha_0.2dropout"

model = tf.keras.models.load_model(PATH_TO_MODEL)
model_regression = tf.keras.models.load_model(PATH_TO_MODEL_REGRESSION)

##
def get_string_for_clustered_age(age_pred):
    if age_pred == 0:
        return "0 - 10 years"
    elif age_pred == 1:
        return "11 - 20 years"
    elif age_pred == 2:
        return "21 - 30 years"
    elif age_pred == 3:
        return "31 - 40 years"
    elif age_pred == 4:
        return "41 - 50 years"
    elif age_pred == 5:
        return "51 - 60 years"
    elif age_pred == 6:
        return "61 - 70 years"
    elif age_pred == 7:
        return "71 - 80 years"
    elif age_pred == 8:
        return "81 - 90 years"
    elif age_pred == 9:
        return "91 - 100 years"



##
def predict_image(path, model, show_image=True,regression_age=True,age_csv_file_path=None):
    """
        create and plot prediction to input image
        :param path: path to file
        :param model: trained model which is used to generate prediction
        :param show_image: bool if image with prediction is plotted or only printed in console
        :param regression_age: bool whether the model has a regression or classification age head
        :param age_csv_file_path: path to csv file to get real age of face
        :return: -
        """
    # extract and prepare image
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    img.set_shape((224, 224, 3))
    img = tf.expand_dims(img, axis=0)
    # generate prediction
    predictions = model.predict(img)
    # extract age
    true_age = -1
    if age_csv_file_path:
        path_for_csv = ""
        if path.__contains__("images"):
            path_for_csv = "images" + path.split("images")[1]
            print(path_for_csv)
        pd.read_csv(age_csv_file_path)
        table_data = pd.read_csv(age_csv_file_path)
        row = table_data[table_data["image_path"] == path_for_csv]
        if not row.empty:
            true_age = row["age"].values[0]
        else:
            true_age = -1

    if show_image:
        # generate plot
        fig, ax = plt.subplots()
        ax.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
        if true_age < 0:
            age_text = r'real age: unknown'
        else:
            age_text = r'real age: %.2f' % true_age
        if regression_age:
            textstr = '\n'.join((
                r'face: %.2f' % (predictions[0],),
                r'mask: %.2f' % (predictions[1],), r'age: %.2f' % (predictions[2]),age_text))
        else:
            age_idx = predictions[2].argmax()
            age_pred = predictions[2].max()
            age_pred_text = get_string_for_clustered_age(age_idx)
            textstr = '\n'.join((
                r'face: %.2f' % (predictions[0],),
                r'mask: %.2f' % (predictions[1],),
                r'age: %s with %.2f ' % (age_pred_text, age_pred),
                age_text))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(240, 112, textstr, fontsize=14, bbox=props, verticalalignment="center")
        plt.show()




##
# open file dialog

# change whether you want to use regression or classification model
use_regression_model = False
root = tk.Tk()
root.withdraw()

filetypes = [('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if use_regression_model:
    # regression
    m = model_regression
    reg = True
else:
    # classification
    m = model
    reg = False
# check file_path and predict image
if file_path:
    # helper.predict_image(file_path, model)
    print(file_path)
    predict_image(file_path,m,True,reg,"images/featureTableTest.csv")
