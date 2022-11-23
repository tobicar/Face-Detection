import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime


def load_model_for_training(version, classes, input_size=224, dropout=0.2, pre_trained=False, alpha=1,
                            depth_multiplier=1):
    """
    load MobileNet version to train on
    :param input_size:
    :param depth_multiplier: depth multiplier of v1
    :param alpha: alpha of v1
    :param version: v3Large, v3Small or v1 can be loaded
    :param classes: count of classes (if weights are not trained)
    :param dropout: dropout rate of last dense layer (default: 20%)
    :param pre_trained: True = load pretrained weights of imagenet; False = random initialized weights
    :return: instance of keras model
    """
    version = version.lower()
    if version == "v3large":
        return tf.keras.applications.MobileNetV3Large(
            input_shape=(input_size, input_size, 3),  # default (224,224,3)
            include_top=not pre_trained,  # if preTrained no Top
            weights='imagenet' if pre_trained else None,
            classes=1000 if pre_trained else classes,
            dropout_rate=dropout,
            classifier_activation="sigmoid",
            include_preprocessing=True)
    elif version == "v1":
        return tf.keras.applications.MobileNet(
            input_shape=(input_size, input_size, 3),
            include_top=not pre_trained,
            weights="imagenet" if pre_trained else None,
            classes=1000 if pre_trained else classes,
            dropout=dropout,
            classifier_activation="sigmoid",
            alpha=alpha,
            depth_multiplier=depth_multiplier)
    elif version == "v3small":
        return tf.keras.applications.MobileNetV3Small(
            input_shape=(input_size, input_size, 3),
            include_top=not pre_trained,
            weights='imagenet' if pre_trained else None,
            classes=1000 if pre_trained else classes,
            dropout_rate=dropout,
            classifier_activation='softmax',
            include_preprocessing=True
        )


def trunc(values, decs=2):
    """
    cuts decimal numbers
    :param values: number (float)
    :param decs: decimals to cut
    :return: number
    """
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def get_prediction_text(prediction_array):
    """
    get label to a specific prediction
    :param prediction_array: array of prediction
    :return: string with prediction text
    """
    face = prediction_array.flatten()[0]
    if face > 0.5:
        text = "face with a probability of: " + str(trunc(face, 4)) + "\n"
    else:
        text = "no face with a probability of: " + str(trunc(1 - face, 4)) + "\n"
    return text


def import_train_images(directory, seed=123, batch_size=32, imagesize=224):
    """
    load training dataset from directory
    :param imagesize: import size of images
    :param directory: path of directory
    :param seed: seed to generate random shuffle for val split
    :param batch_size: Size of the batches of data
    :return: tuple with training and validation tf.data.Dataset
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["no_face", "face"],
        batch_size=batch_size,
        image_size=(imagesize, imagesize),
        shuffle=True,
        seed=seed,
        validation_split=0.1765,
        subset="both",
        crop_to_aspect_ratio=True)


def import_test_images(directory, batch_size=32, image_size=224):
    """
    import test images from test directory
    :param image_size: import size of images
    :param directory: path to the directory
    :param batch_size: Size of the batches of Data
    :return: tf.data.Dataset with test images and labels
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode="binary",
        class_names=["no_face", "face"],
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=True,
        crop_to_aspect_ratio=True)


def print_original_image(path, text=""):
    """
    show image with original size
    :param path: path to image file
    :param text: probabilities
    :return: -
    """
    image = tf.keras.preprocessing.image.load_img(path)
    figure, ax = plt.subplots()
    ax.imshow(image)
    # hide y-axis
    ax.get_yaxis().set_visible(False)
    ax.set_xticklabels([])
    ax.set_xlabel(text, loc='left', size=20)


def predict_image(path, model, show_image=True):
    """
    predict single image and calculate probability if a face is in the picture or not
    :param path: path to image
    :param model: trained model
    :param show_image: shows image with prediction probability in additional window
    :return: text with prediction probability
    """
    image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert image to single batch.
    predictions = model.predict(input_arr)
    text = get_prediction_text(predictions[0])
    if show_image:
        print_original_image(path, text)
    return text


def train_model(model, epochs, train_ds, val_ds, save_file_name):
    """
    Method compile a model with the binary cross entropy as loss function for binary classification problems.
    As metric accuracy is used.
    :param model: model to train
    :param epochs: number of epochs to train
    :param train_ds: dataset used for training the model
    :param val_ds: dataset used for calculating validation values
    :param save_file_name: filename of the saved model
    :return: the training history
    """
    # compile Model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.summary()

    # train the model
    tf.debugging.set_log_device_placement(True)
    history = model.fit(train_ds,
                        epochs=epochs,
                        validation_data=val_ds, callbacks=[tensorboard_callback], )

    # save model
    model.save("saved_model/" + save_file_name)
    return history


def generate_history_and_save(history, name):
    """
    generates figure with training and validation accuracy and loss and saves it to file
    :param history: the history of the trained model
    :param name: filename of the saved figure
    :return: -
    """
    # parts from https://www.tensorflow.org/tutorials/images/transfer_learning
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig("plots/" + name)
