## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import MobileNet

## load model
PATH_TO_MODEL = "saved_model/model_50epochs"
model = tf.keras.models.load_model(PATH_TO_MODEL)
## load test Dataset
test_ds = MobileNet.import_test_images("dir")

## evaluierung des Models
#TODO: Implementierung und model.evaluate anschauen
evaluation = model.evaluate(test_ds)
##
#TODO: Visualisierung der Dateb

##
## OpenFileDialog

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

filetypes =[('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if file_path:
    loadImagePredict(file_path, model)
