## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import MobileNet

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Trainingspipeline:
BATCH_SIZE = 64 #TODO: Hier varieren mit 32, 64, 128, 256
EPOCHS = 50 #TODO: Hier variieren

#TODO: Crossvalidation
#TODO: Learning Rate Decay ???
#TODO: Early Stopping (Overfitting vermeiden)

## load train and validation data
(train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=BATCH_SIZE)

## buffer datasets in RAM to prevent I/0 Blocking
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

## load net architecture
model_scratch = MobileNet.load_model_for_training("v3Large", 1) #TODO: Dropout: 0.1 0.2 0.3

# Um BinaryCrossEntropy verwenden zu können, muss classes wahrscheinlich auf 1 stehen.

## Ausgabe des Models
model_scratch.summary()

## pipeline scratch model
history = MobileNet.train_model(model_scratch, EPOCHS, train_ds, val_ds, "model_scratch_" + str(EPOCHS) + "epochs_"
                                + str(BATCH_SIZE) + "batch")

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

history = MobileNet.train_model(model_transfer, EPOCHS, train_ds, val_ds, "model_transfer_" + str(EPOCHS) + "epochs_"
                                + str(BATCH_SIZE) + "batch")

## plot
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
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()






