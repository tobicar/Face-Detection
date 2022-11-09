## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import MobileNet

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Trainingspipeline:

BATCH_SIZES = [32,64,128,256]
#BATCH_SIZE = 64 #TODO: Hier varieren mit 32, 64, 128, 256
EPOCHS = [10,30] #TODO: Hier variieren

#TODO: Crossvalidation
#TODO: Learning Rate Decay ???
#TODO: Early Stopping (Overfitting vermeiden)

##
for batch in BATCH_SIZES:
    for epoch in EPOCHS:
        # load train and validation data
        (train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=batch)
        #buffer datasets in RAM to prevent I/0 Blocking
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        # pipeline pretrained
        model_pretrained = MobileNet.load_model_for_training("v3Large", 1000,
                                                             pre_trained=True)  # TODO: Dropout: 0.1 0.2 0.3
        model_pretrained.trainable = False
        # Create Own Model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
        x = model_pretrained(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        # x = tf.keras.layers.Dense(1000)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model_transfer = tf.keras.Model(inputs, outputs)
        name = "model_transfer_" + str(epoch) + "epochs_" + str(batch) + "batch"
        history = MobileNet.train_model(model_transfer, epoch, train_ds, val_ds, name)
        MobileNet.generate_history_and_save(history, name)





## load train and validation data
(train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=BATCH_SIZE)

## buffer datasets in RAM to prevent I/0 Blocking
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

## pipeline pretrained
model_pretrained = MobileNet.load_model_for_training("v3Large", 1000, pre_trained=True) #TODO: Dropout: 0.1 0.2 0.3
model_pretrained.trainable = False
model_pretrained.summary()

## Create Own Model
inputs = tf.keras.Input(shape=(224, 224, 3))
#x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
x = model_pretrained(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.Dense(1000)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_transfer = tf.keras.Model(inputs, outputs)
##
history = MobileNet.train_model(model_transfer, EPOCHS, train_ds, val_ds, "model_transfer_" + str(EPOCHS) + "epochs_"
                                + str(BATCH_SIZE) + "batch")
