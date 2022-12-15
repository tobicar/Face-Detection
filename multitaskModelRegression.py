##
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import helper_multitask


##
def create_dataset(csv_path, only_age=False):
    """
    function creates dataset from table given as csv file
    :param csv_path: path to csv file
    :param only_age: bool to only extract data with given age
    :return: dataset and data as table
    """
    # read csv file
    table_data = pd.read_csv(csv_path)
    if only_age:
        # extract only data with given age
        table_data = table_data[table_data["age"] >= 1]

    table_data = shuffle(table_data, random_state=123)
    # create new columns in data table with weights of data in each row
    table_data['face_weights'] = 1
    table_data['mask_weights'] = table_data['face']
    table_data['age_weights'] = table_data["age"].apply(lambda x: 1 if x >= 1 else 0)
    # create dictionary of all weights
    dict_weighted = {"face_detection": np.array(table_data['face_weights']),
                     "mask_detection": np.array(table_data['mask_weights']),
                     "age_detection": np.array(table_data['age_weights'])}
    # transform data table to tensorflow dataset
    data = tf.data.Dataset.from_tensor_slices(
        (table_data["image_path"], table_data[["face", "mask", "age"]], dict_weighted))
    ds = data.map(helper_multitask.process_path)
    ds = ds.batch(32)

    return ds, table_data


# generate datasets
train_ds, train_table = create_dataset("images/featureTableTrain.csv")
val_ds, val_table = create_dataset("images/featureTableVal.csv")
test_ds, test_table = create_dataset("images/featureTableTest.csv")

val_ds_age, val_table_age = create_dataset("images/featureTableVal.csv", only_age=True)

## generate regression training loop
train_ds, train_table = helper_multitask.create_dataset("regression","face","images/featureTableTrain.csv",weighted_regression=True)
val_ds_age, val_table_age = helper_multitask.create_dataset("regression","age","images/featureTableVal.csv")

##
# Regression
EPOCHS = [50]
ALPHAS = [0.25]
LOSS = ['mse']
DROPOUTS = [0.2]
LARGE_VERSION = [False, True]

for alpha in ALPHAS:
    for dropout in DROPOUTS:
        for largeVersion in LARGE_VERSION:
            for loss in LOSS:
                for epochs in EPOCHS:
                    model = helper_multitask.create_model("regression",
                                                          alpha=alpha,
                                                          dropout=dropout,
                                                          large_version=largeVersion)
                    model = helper_multitask.compile_model(model, "regression")
                    time = datetime.datetime.now().strftime("%Y%m%d-%H%M_")
                    name = r"" + time + "regression_" + str(epochs) + "epochs_" + \
                           str(alpha) + "alpha_" + str(dropout) + "dropout" + loss + "_ValOnlyAge"

                    if largeVersion:
                        name += "_largeVersion"

                    log_dir = "logs/fit/" + name + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
                    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
                    tf.debugging.set_log_device_placement(True)

                    model_history = model.fit(train_ds,
                                              epochs=epochs,
                                              validation_data=val_ds_age,
                                              callbacks=[tensorboard_callback])

                    # save model
                    model.save("saved_model/Milestone3/" + name)

## generate History and plot it
def plot_history(model_history):
    epochs_range = range(len(model_history.epoch))

    complete_loss = model_history.history['loss']
    face_detection_loss = model_history.history["face_detection_loss"]
    mask_detection_loss = model_history.history["mask_detection_loss"]
    age_detection_loss = model_history.history["age_detection_loss"]

    face_detection_accuracy = model_history.history['face_detection_accuracy']
    mask_detection_accuracy = model_history.history['mask_detection_accuracy']
    age_detection_mse = model_history.history["age_detection_mse"]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, face_detection_accuracy, label='Face Accuracy')
    plt.plot(epochs_range, mask_detection_accuracy, label='Mask Accuracy')
    # plt.plot(epochs_range, age_detection_mse, label='Age MSE')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, complete_loss, label='Training Loss')
    plt.plot(epochs_range, face_detection_loss, label='Face Loss')
    plt.plot(epochs_range, mask_detection_loss, label='Mask Loss')
    plt.plot(epochs_range, age_detection_loss, label='Age Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


## generate scatter plot
model_large = tf.keras.models.load_model(
    "saved_model/Milestone3/regression_100epochs_0.25alpha_0.2dropoutmse_largeVersion")
model_small = tf.keras.models.load_model("saved_model/Milestone3/regression_100epochs_0.25alpha_0.2dropoutmse")

pred_small_train = model_small.predict(train_ds)
pred_small_val = model_small.predict(val_ds)
pred_small_test = model_small.predict(test_ds)
pred_large_train = model_large.predict(train_ds)
pred_large_val = model_large.predict(val_ds)
pred_large_test = model_large.predict(test_ds)
# plt.scatter(train_table["age"], pred_train[2], s=70, alpha=0.2)
# plt.scatter(train_table['age'], train_table['age'], s=10)
# plt.scatter(val_table["age"], pred_val[2])
# plt.scatter(val_table['age'], val_table['age'])

plt.scatter(test_table["age"], pred_small_test[2], s=70, alpha=0.2)
plt.scatter(test_table['age'], test_table['age'], s=10)
plt.scatter(test_table['age'], pred_large_test[2], s=70, alpha=0.2)
