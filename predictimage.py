##
# OpenFileDialog and predict one Image
import helper
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
##
PATH_TO_MODEL = "saved_model/modelv1_scratch_75epochs_32batch_025alpha_1depthMultiplier"
#PATH_TO_MODEL = "saved_model/modelv1_scratch_75epochs_32batch_1alpha_1depthMultiplier"
model = tf.keras.models.load_model(PATH_TO_MODEL)

##
root = tk.Tk()
root.withdraw()

filetypes = [('image files', '.png .jpg .jpeg .jfif')]
file_path = filedialog.askopenfilename(parent=root, filetypes=filetypes)

if file_path:
    helper.predict_image(file_path, model)
