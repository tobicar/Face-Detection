## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

##
def load_model_for_training(version, classes,dropout=0.2,pre_trained=False):
    """
    :param version: v3Large oder v3Small kann geladen werden
    :param classes: Anzahl der Klassen wenn die Gewichte nicht vortrainiert geladen werden
    :param dropout: Dropout Rate im letzten Dense Layer (default: 20%)
    :param preTrained: Laden der vortrainierten Gewichte des ImageNets oder randomisiert initialisierte Gewichte
    :return: Instanz des Keras Models
    """
    version = version.lower()
    if version == "v3large":
        return tf.keras.applications.MobileNetV3Large(
            input_shape=(224, 224, 3),  # default (224,224,3)
            include_top= not pre_trained, # if preTrained no Top
            weights='imagenet' if pre_trained else None,
            classes=1000 if pre_trained else classes,
            dropout_rate=dropout,
            classifier_activation="softmax",
            include_preprocessing=True)
    elif version == "v3small":
        return tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=not pre_trained,
            weights='imagenet' if pre_trained else None,
            classes=1000 if pre_trained else classes,
            dropout_rate=dropout,
            classifier_activation='softmax',
            include_preprocessing=True
        )

## cut decimals
def trunc(values, decs=2):
    """ cuts decimal numbers
    :param values: number (float)
    :param decs: decimals to cut
    :return: number
    """
    return np.trunc(values*10**decs)/(10**decs)

## config output
#TODO: Falls Training mit binaryCrossEntropy -> anderes prediction array
def get_prediction_text(prediction_array):
    """ get label to a specific prediction
    :param prediction_array: array of prediction
    :return: string with max prediction and label
    """
    labels = ["no face", "face"]
    sort_array = np.argsort(prediction_array)[::-1]
    print(sort_array)
    print(prediction_array)
    text = ""
    for e in sort_array:
        text += labels[e] + ": " + str(trunc(prediction_array[e], 4)) + "\n"
    return text

## import images from directory
def import_train_images(directory,seed=1, batch_size=32):
    """
    load training dataset from directory
    :param directory: path of directory
    :param seed: seed to generate random shuffle for val split
    :param batch_size: Size of the batches of data
    :return: tuple with training and validation tf.data.Dataset
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary", #oder mode binary
        class_names=["no_face", "face"], #umbenannt
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=seed,
        validation_split=0.8235,
        subset="both",
        crop_to_aspect_ratio=True)

##
def import_test_images(directory,batch_size=32):
    """
    import test images from test directory
    :param directory: path to the directory
    :param batch_size: Size of the batches of Data
    :return: tf.data.Dataset with test images and labels
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode="binary",
        class_names=["no_face", "face"],
        batch_size=batch_size,
        image_size=(224,224),
        shuffle=True,
        crop_to_aspect_ratio=True)

## show image
def print_original_image(path, text=""):
    """ show image with original size
    :param path: path to image file
    :return: -
    """
    image = tf.keras.preprocessing.image.load_img(path)
    figure, ax = plt.subplots()
    ax.imshow(image)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    ax.set_xticklabels([])
    ax.set_xlabel(text, loc='left')

##
def predict_image(path, model,show_image=True):
    """
    predict single image and calculate probability if a face is in the picture or not
    :param path: path to image
    :param model: trained model
    :param show_image: shows image with prediction probability in additional window
    :return: text with prediction probability
    """
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert image to single batch.
    predictions = model.predict(input_arr)
    text = get_prediction_text(predictions[0])
    if show_image:
        print_original_image(path, text)
    return text

##
#ds_size = len(ds.file_paths)
#train_size = int(0.7 * ds_size)
#val_size = int(0.15 * ds_size)
#test_size = int(0.15 * ds_size)

#full_dataset = ds
#full_dataset = full_dataset.shuffle()
#train_dataset = full_dataset.take(train_size)
#test_dataset = full_dataset.skip(train_size)
#val_dataset = test_dataset.skip(val_size)
#test_dataset = test_dataset.take(test_size)


