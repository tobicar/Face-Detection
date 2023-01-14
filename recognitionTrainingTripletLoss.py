import helper_tripletloss as helper
import pandas as pd
import numpy as np
import tensorflow as tf
import os

##
# generate tensorflow dataset

# path to images
image_path = r"C:\Users\svea\PycharmProjects\Face-Detection\images\milestone4\train"
#MAC OS
#image_path = "/Users/tobias/PycharmProjects/Face-Detection/images/rawdata4/archive/Faces/Faces"

images = sorted([(str(image_path + "\\" + f), str(f.split("_")[0])) for f in os.listdir(image_path)])
#MAC OS
#images = sorted([(str(image_path + "/" + f), str(f.split("_")[0])) for f in os.listdir(image_path)])

images_df = pd.DataFrame(images, columns=["path", "class"])
directory_path = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\utkCropped\utkcropped\utkcropped"
subfiles = [directory_path + "/" + f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

# generate triplets
triplets = helper.make_triplets_utk(images_df["path"], subfiles, images_df["class"], num=30)

##
# split train and validation data using online strategy
np.random.seed(123)
np.random.shuffle(triplets)
train = triplets[0:int(len(triplets) * 0.75)]
val = triplets[int(len(triplets) * 0.75):]

# creating batched dataset for train
data_train = tf.data.Dataset.from_tensor_slices((train[:, 0], train[:, 1], train[:, 2]))
ds_train = data_train.map(helper.preprocess_triplets_array)
ds_train = ds_train.batch(32)
# creating batched dataset for validation
data_val = tf.data.Dataset.from_tensor_slices((val[:, 0], val[:, 1], val[:, 2]))
ds_val = data_val.map(helper.preprocess_triplets_array)
ds_val = ds_val.batch(32)

##
# create model
siamese_net = helper.create_model()
# reform it to siamese model
siamese_model = helper.SiameseModel(siamese_net)
siamese_model.compile(optimizer='adam', loss=helper.triplet_loss, metrics=["accuracy", "precision", helper.triplet_accuracy])

##
# train model
history = siamese_model.fit(ds_train, epochs=15, validation_data=ds_val)
##
# save weights
siamese_model.save_weights("saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain/siamese_net")

##
# load weights
# have to create model before
load_status = siamese_model.load_weights("saved_model/Milestone4/tripletLoss_15epochs_alpha1_weights_onlyTrain_utk/siamese_net")
## visualize one sample

def visualize(number_of_sample, anchor, positive, negative):
    """Visualize one triplet from the supplied batch."""

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(3, 1))
    axs = fig.subplots(3, 1)
    show(axs[0], tf.keras.preprocessing.image.array_to_img(anchor[number_of_sample]))
    show(axs[1], tf.keras.preprocessing.image.array_to_img(positive[number_of_sample]))
    show(axs[2], tf.keras.preprocessing.image.array_to_img(negative[number_of_sample]))
