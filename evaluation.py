##
import pandas as pd
import tensorflow as tf
from datetime import datetime
import MobileNet
import os
##
# load test dataset with batch size 1
test_ds = MobileNet.import_test_images("images/test", 1)
##
# evaluate all current models and save name, loss and accuracy to array
data = []
directory = "saved_model"
for model_path in os.listdir(directory):
    # because of macOS DS_Store folder
    if model_path == ".DS_Store" or not model_path.__contains__("v1"):
        continue
    model = tf.keras.models.load_model(directory + "/" + model_path)
    evaluation = model.evaluate(test_ds)
    row = {"name": model_path, "loss": evaluation[0], "acc": evaluation[1],
           "epochs": model_path.split("_")[2].split("e")[0], "batch": model_path.split("_")[3].split("b")[0]}
    data.append(row)
##
# save data to pandas Dataframe and to file
df = pd.DataFrame(data)
time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df.to_csv("evaluation/" + str(time) + ".csv")
