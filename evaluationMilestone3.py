##
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os

import helper_multitask

##
# datasets for regression models
test_ds_face, test_table_face = helper_multitask.create_dataset("regression", "face", "images/featureTableTest.csv", weighted_regression=False)
test_ds_age, test_table_age = helper_multitask.create_dataset("regression", "age", "images/featureTableTest.csv", weighted_regression=False)
test_ds_mask, test_table_mask = helper_multitask.create_dataset("regression", "mask", "images/featureTableTest.csv", weighted_regression=False)

# datasets for the classification models
test_ds_face_class, test_table_face_class = helper_multitask.create_dataset("classification", "face", "images/featureTableTest.csv")
test_ds_age_class, test_table_age_class = helper_multitask.create_dataset("classification", "age", "images/featureTableTest.csv")
test_ds_mask_class, test_table_mask_class = helper_multitask.create_dataset("classification", "mask", "images/featureTableTest.csv")

##
# evaluate through all models

data_classification = []
data_regression = []
directory = "saved_model/Milestone3"

for model_path in os.listdir(directory):
    # because of macOS DS_Store folder
    if model_path == ".DS_Store":
        continue
    if model_path.__contains__("regression"):
        model = tf.keras.models.load_model(directory + "/" + model_path)
        face = model.evaluate(test_ds_face)
        mask = model.evaluate(test_ds_mask)
        age = model.evaluate(test_ds_age)
        row = {"name": model_path, "loss_all_face": face[0], "loss_all_mask": mask[0], "loss_all_age":age[0],"loss_face":face[1],"loss_mask":mask[2],"loss_age":age[3],"face_acc":face[4],"mask_acc":mask[5],"age_mse":age[6],"age_mae":age[7]}
        data_regression.append(row)
    #elif model_path.__contains__("classification"):
    #    model = tf.keras.models.load_model(directory + "/" + model_path)
    #    face = model.evaluate(test_ds_face_class)
    #    mask = model.evaluate(test_ds_mask_class)
    #    age = model.evaluate(test_ds_age_class)
    #    row = {"name": model_path, "loss_all_face": face[0], "loss_all_mask": mask[0], "loss_all_age":age[0],"loss_face":face[1],"loss_mask":mask[2],"loss_age":age[3],"face_acc":face[4],"mask_acc":mask[5],"age_acc":age[6]}
    #    data_classification.append(row)
    #else:
    #    raise IOError


##
df_regression = pd.DataFrame(data_regression)
time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df_regression.to_csv("evaluation/Milestone3/" + "regression_" + str(time) + ".csv")

##
df_classification = pd.DataFrame(data_classification)
time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
df_classification.to_csv("evaluation/Milestone3/" + "classification_" + str(time) + ".csv")