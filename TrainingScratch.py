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

## print summary of the model
model_scratch.summary()
## load,train and save model
history = MobileNet.train_model(model_scratch, EPOCHS, train_ds, val_ds, "model_scratch_" + str(EPOCHS) + "epochs_"
                                + str(BATCH_SIZE) + "batch")
