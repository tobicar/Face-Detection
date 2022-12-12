##
import pandas as pd
import tensorflow as tf
from datetime import datetime
import helper
import os
import time
from sklearn.utils import shuffle
import numpy as np

import helper_multitask

##
test_ds_face, test_table_face = helper_multitask.create_dataset("regression", "face", "images/featureTableTest.csv")
test_ds_age, test_table_age = create_dataset_regression("images/featureTableTest.csv", only_age=True)
test_ds_mask, test_table_mask = create_dataset_regression("images/featureTableTest.csv", only_mask=True)

test_ds_face_class, test_table_face_class = create_dataset_classification("images/featureTableTest.csv")
test_ds_age_class, test_table_age_class = create_dataset_classification("images/featureTableTest.csv", only_age=True)
test_ds_mask_class, test_table_mask_class = create_dataset_classification("images/featureTableTest.csv", only_mask=True)
##

helper_multitask.create_dataset()

## evaluate through all models

data = []
directory = "saved_model"

for model_path in os.listdir(directory):
    # because of macOS DS_Store folder
    if model_path == ".DS_Store":
        continue
    model = tf.keras.models.load_model(directory + "/" + model_path)
    evaluation = model.evaluate(test_ds_face)