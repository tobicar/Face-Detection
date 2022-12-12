##
import pandas as pd
import tensorflow as tf
import helper_multitask
import datetime
from sklearn.utils import shuffle


##
def clusterAges(x):
    """
    function clusters ages into 10 classes.
    In class -1 are all images with no ages given.
    :param x: age of face in image
    :return: class id
    """
    if x < 1:
        return -1
    if 0 < x <= 10:
        return 0
    if 10 < x <= 20:
        return 1
    if 20 < x <= 30:
        return 2
    if 30 < x <= 40:
        return 3
    if 40 < x <= 50:
        return 4
    if 50 < x <= 60:
        return 5
    if 60 < x <= 70:
        return 6
    if 70 < x <= 80:
        return 7
    if 80 < x <= 90:
        return 8
    if 90 < x <= 100:
        return 9


def create_dataset(csv_path, only_age=False):
    """
    function creates dataset from table given as csv file
    :param csv_path: path to csv file
    :param only_age: bool to only extract data with given age
    :return: dataset and data as table
    """
    # read csv file
    table_data = pd.read_csv(csv_path)
    # add new column (age_clustered) to table of csv file which includes the class created in clusterAges(x)
    table_data['age_clustered'] = table_data["age"].apply(clusterAges)
    if only_age:
        table_data = table_data[table_data["age"] >= 1]

    # transform data table to tensorflow dataset
    table_data = shuffle(table_data, random_state=123)
    data = tf.data.Dataset.from_tensor_slices((table_data["image_path"], table_data[["face", "mask", "age_clustered"]]))
    ds = data.map(helper_multitask.process_path)
    ds = ds.batch(32)

    return ds, table_data


## generate datasets
train_ds, train_table = create_dataset("images/featureTableTrain.csv")
val_ds, val_table = create_dataset("images/featureTableVal.csv")

val_ds_age, val_table_age = create_dataset("images/featureTableVal.csv", only_age=True)

## generate classification trainings loop
EPOCHS = [100]
ALPHAS = [0.25]
DROPOUTS = [0.2]
LARGE_VERSION = [False]
L2 = [True]

for large in LARGE_VERSION:
    for alpha in ALPHAS:
        for dropout in DROPOUTS:
            for epochs in EPOCHS:
                for l2 in L2:
                    model = helper_multitask.create_model("classification",
                                                          alpha=alpha,
                                                          dropout=dropout,
                                                          large_version=large,
                                                          regularizer=l2)
                    model = helper_multitask.compile_model(model, "classification")
                    name = r"classification" + str(epochs) + "epochs_" + \
                           str(alpha) + "alpha_" + str(dropout) + "dropout_ValOnlyAge"
                    if large:
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

## load test data
ONLY_AGE = True
test_table = pd.read_csv("images/featureTableVal.csv")  # oder featureTableTest.csv
test_table['age_clustered'] = test_table["age"].apply(clusterAges)
if ONLY_AGE:
    test_table = test_table[test_table["age"] >= 1]
test_table = shuffle(test_table, random_state=123)
test_data = tf.data.Dataset.from_tensor_slices(
    (test_table["image_path"], test_table[["face", "mask", "age_clustered"]]))
ds_test = test_data.map(helper_multitask.process_path)
ds_test = ds_test.batch(32)
## augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])

batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, shuffle=False, augment=False):
    if shuffle:
        ds = ds.shuffle(1000)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)
    # Use buffered prefetching on all datasets.
    return ds
