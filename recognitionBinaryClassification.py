##
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import helper_contrastiveloss
## load all images from the train folder into a list and generate pd.Dataframe from it
image_path = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\milestone4\train"
# image_path = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/archive/Faces/Faces"

images = sorted([(str(image_path + "/" + f), str(f.split("_")[0])) for f in os.listdir(image_path)])

images_df = pd.DataFrame(images, columns=["path", "class"])

len(images_df.groupby("class").count())

## generate the pairs
utk_cropped_path = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\utkCropped\utkcropped"
pairs = helper_contrastiveloss.make_pairs(images_df["path"], images_df["class"], negative_path=utk_cropped_path, num_random_img=0)
## generate the train and the validation dataset
np.random.RandomState(seed=32).shuffle(pairs[0])
np.random.RandomState(seed=32).shuffle(pairs[1])

train = pairs[0][0:int(len(pairs[0]) * 0.75)]
val = pairs[0][int(len(pairs[0]) * 0.75):]

train_label = pairs[1][0:int(len(pairs[1]) * 0.75)]
val_label = pairs[1][int(len(pairs[1]) * 0.75):]

data_train = tf.data.Dataset.from_tensor_slices((train[:, 0], train[:, 1], train_label))
data_val = tf.data.Dataset.from_tensor_slices((val[:, 0], val[:, 1], val_label))

ds_train = data_train.map(helper_contrastiveloss.preprocess_binary_array)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(8)

ds_val = data_val.map(helper_contrastiveloss.preprocess_binary_array)
ds_val = ds_val.batch(32)
ds_val = ds_val.prefetch(8)


## generate the model
EPOCHS = 15
BATCH_SIZE = 32
model, feature_generator = helper_contrastiveloss.create_model(mode="pooling")
model = helper_contrastiveloss.compile_model(model)
model.summary()
## train the model
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS)
## save the trained model to file system
NAME_TO_SAVE = "saved_model/Milestone4/binaryClassification_15epochs_alpha1_onlyTrain_pooling"
model.save(NAME_TO_SAVE)
