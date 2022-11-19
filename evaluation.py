##
import pandas as pd
import tensorflow as tf
from datetime import datetime
import helper
import os
import time
##
# load test dataset with batch size 1
test_ds = helper.import_test_images("images/test", 1,)
##
# evaluate all current models and save name, loss and accuracy to array
data = []
directory = "saved_model"
for model_path in os.listdir(directory):
    # because of macOS DS_Store folder
    if model_path == ".DS_Store" or not model_path.__contains__("v1") or model_path.__contains__("inputSize"):
        continue
    #test_ds = helper.import_test_images("images/test", 1, image_size=int(model_path.split("_")[5].split("i")[0]))
    model = tf.keras.models.load_model(directory + "/" + model_path)
    evaluation = model.evaluate(test_ds)
    row = {"name": model_path, "loss": evaluation[0], "acc": evaluation[1],
           "epochs": model_path.split("_")[2].split("e")[0], "batch": model_path.split("_")[3].split("b")[0],
           "alpha": model_path.split("_")[4].split("a")[0] if len(model_path.split("_")) >= 5 else "",
           "depth_multiplier": model_path.split("_")[5].split("d")[0] if len(model_path.split("_")) >= 6 else ""}
    data.append(row)
##
# save data to pandas Dataframe and to file
df = pd.DataFrame(data)
time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df.to_csv("evaluation/" + str(time) + ".csv")


## evaluate time for prediction of specific models
# load test dataset with batch size 1
test_ds = helper.import_test_images("images/test", 1,)
data = []
models = ["saved_model/modelv1_scratch_75epochs_32batch_1alpha_1depthMultiplier",
          "saved_model/modelv1_scratch_75epochs_32batch_075alpha_1depthMultiplier",
          "saved_model/modelv1_scratch_75epochs_32batch_05alpha_1depthMultiplier",
          "saved_model/modelv1_scratch_75epochs_32batch_025alpha_1depthMultiplier",
          "saved_model/modelv1_transfer_10epochs_128batch",
          "saved_model/modelv1_transfer_10epochs_128batch_075alpha",
          "saved_model/modelv1_transfer_10epochs_128batch_05alpha",
          "saved_model/modelv1_transfer_10epochs_128batch_025alpha"
          ]
for m in models:
    model = tf.keras.models.load_model(m)
    begin = time.time()
    pred = model.predict(test_ds)
    end = time.time()
    row = {"name": m, "time": end-begin}
    data.append(row)
##
# save data to pandas Dataframe and to file
df = pd.DataFrame(data)
time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df.to_csv("evaluation/time_measurement_" + str(time) + ".csv")

## evaluate time for prediction of models with different input size (width multiplier)
data = []
models = ["saved_model/modelv1_scratch_30epochs_32batch_10alpha_128inputSize",
          "saved_model/modelv1_scratch_30epochs_32batch_10alpha_160inputSize",
          "saved_model/modelv1_scratch_30epochs_32batch_10alpha_192inputSize",
          "saved_model/modelv1_scratch_30epochs_32batch"
          ]
for m in models:
    model = tf.keras.models.load_model(m)
    if len(m.split("_")) == 7:
        test_ds = helper.import_test_images("images/test", 1, int(m.split("_")[6].split("i")[0]))
    else:
        test_ds = helper.import_test_images("images/test", 1, )
    begin = time.time()
    pred = model.predict(test_ds)
    end = time.time()
    row = {"name": m, "time": end-begin}
    data.append(row)

##
# save data to pandas Dataframe and to file
df = pd.DataFrame(data)
timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df.to_csv("evaluation/time_measurement_" + str(timestamp) + ".csv")
