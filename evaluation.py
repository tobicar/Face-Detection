## imports
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import date
import MobileNet
import os
## load test dataset with batchsize 1
test_ds = MobileNet.import_test_images("images/test",1)
## evaluate all current models and save name, loss and acuraccy to array
data = []
directory = "saved_model"
for model_path in os.listdir(directory):
    # because of MAC OS DS_Store folder
    if model_path == ".DS_Store":
        continue
    model = tf.keras.models.load_model(directory + "/" +model_path)
    evaluation = model.evaluate(test_ds)
    row = {"name": model_path, "loss": evaluation[0], "acc": evaluation[1]}
    data.append(row)
## save data to pandas Dataframe and to file
df = pd.DataFrame(data)
time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df.to_csv("evaluation/" + str(time) + ".csv")



## load one model (old)
PATH_TO_MODEL = "saved_model/model_transfer_10epochs_32batch"
model = tf.keras.models.load_model(PATH_TO_MODEL)
##
#TODO: Visualisierung der Daten

##
## OpenFileDialog and predict one Image

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

filetypes =[('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if file_path:
    MobileNet.predict_image(file_path, model)
