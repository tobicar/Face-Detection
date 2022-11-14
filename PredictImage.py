## OpenFileDialog and predict one Image
import MobileNet
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
##
PATH_TO_MODEL = "saved_model/model_transfer_10epochs_32batch"
model = tf.keras.models.load_model(PATH_TO_MODEL)

## TODO: Funktioniert noch nicht richtig
root = tk.Tk()
root.withdraw()

filetypes =[('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if file_path:
    MobileNet.predict_image(file_path, model)