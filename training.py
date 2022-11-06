## imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import MobileNet

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Trainingspipeline:
BATCH_SIZE = 32 #TODO: Hier varieren mit 32, 64, 128, 256

## load Data
(train_ds, val_ds) = MobileNet.loadTrainImages()
test_ds = MobileNet.loadTestImages()

#diese Funktion verwenden
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=(224, 224))


## Lade die Netzarchitektur
model = MobileNet.load_model("v3Large",2)
# Um BinaryCrossEntropy verwenden zu können, muss classes wahrscheinlich auf 1 stehen.

## Ausgabe des Models
model.summary()



