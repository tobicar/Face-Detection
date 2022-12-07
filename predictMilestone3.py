##
# OpenFileDialog and predict one Image
import helper
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
##
PATH_TO_MODEL = "saved_model/milestone2_regression_100epochs_025alpha_0.2dropout_mse"

model = tf.keras.models.load_model(PATH_TO_MODEL)

##
def trunc(values, decs=2):
    """
    cuts decimal numbers
    :param values: number (float)
    :param decs: decimals to cut
    :return: number
    """
    return np.trunc(values * 10 ** decs) / (10 ** decs)

##
path = "/Users/tobias/Downloads/bild.png"
def predict_image(path, model, show_image=True,regression_age=True,age_csv_file_path = None):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (224, 224), method="bilinear")
    img.set_shape((224, 224, 3))
    img = tf.expand_dims(img,axis=0)
    predictions = model.predict(img)
    true_age = ""
    if age_csv_file_path:
        path_for_csv = ""
        if path.__contains__("images"):
            path_for_csv = "images" + path.split("images")[1]
            print(path_for_csv)
        pd.read_csv(age_csv_file_path)
        table_data = pd.read_csv(age_csv_file_path)
        row = table_data[table_data["image_path"] == path_for_csv]
        if not row.empty:
            true_age =row["age"].values[0]
        else:
            true_age = ""
    if show_image:
        fig, ax = plt.subplots()
        ax.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
        if type(true_age) == str:
            age_text = r'real age: unknown'
        else:
            age_text = r'real age: %.2f' % true_age
        textstr = '\n'.join((
            r'face: %.2f' % (predictions[0],),
            r'mask: %.2f' % (predictions[1],), r'age: %.2f' % (predictions[2]),age_text))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(240, 112, textstr,  fontsize=14, bbox=props,verticalalignment="center")
        plt.show()




##
root = tk.Tk()
root.withdraw()

filetypes = [('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if file_path:
    #helper.predict_image(file_path, model)
    print(file_path)
    predict_image(file_path,model,True,True,"images/featureTableTest.csv")
