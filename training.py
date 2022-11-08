## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import MobileNet

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Trainingspipeline:
BATCH_SIZE = 32 #TODO: Hier varieren mit 32, 64, 128, 256

## load train and validation data
(train_ds, val_ds) = MobileNet.import_train_images()

## load net architecture
model_scratch = MobileNet.load_model_for_training("v3Large",1)

# Um BinaryCrossEntropy verwenden zu können, muss classes wahrscheinlich auf 1 stehen.

## Ausgabe des Models
model.summary()

## pipeline scratch model

#buffer datasets in RAM to prevent I/0 Blocking
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


## pipeline pretrained
model_pretrained = MobileNet.load_model_for_training("v3Large",1000,pre_trained=True)
model_pretrained.trainable = False
model_pretrained.summary()

## Create Own Model
inputs = tf.keras.Input(shape=(224,224,3))
#x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
x = model_pretrained(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
#x = tf.keras.layers.Dense(1000)(x)
outputs = tf.keras.layers.Dense(1)(x)
model_transfer = tf.keras.Model(inputs, outputs)

## compile Model
model_transfer.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_transfer.summary()

## train the model
tf.debugging.set_log_device_placement(True)
history = model_transfer.fit(train_ds,
                    epochs=10,
                    validation_data=val_ds, callbacks=[tensorboard_callback],)
## save model
model_transfer.save("saved_model/model_transfer")







