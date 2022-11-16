##
# imports
import tensorflow as tf
import helper

# train pipeline:
BATCH_SIZE = 64
EPOCHS = 50

# TODO: Cross-validation
# TODO: Learning Rate Decay ???
# TODO: Early Stopping (avoid Overfitting)

##
# load train and validation data
(train_ds, val_ds) = helper.import_train_images("images/train", batch_size=BATCH_SIZE, seed=123)

##
# buffer datasets in RAM to prevent I/0 Blocking
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

##
# load net architecture
model_scratch = helper.load_model_for_training("v3Large", 1)  # TODO: Dropout: 0.1 0.2 0.3

##
# print model
model_scratch.summary()

##
# pipeline scratch model
history = helper.train_model(model_scratch, EPOCHS, train_ds, val_ds, "model_scratch_" + str(EPOCHS) + "epochs_"
                                + str(BATCH_SIZE) + "batch")

##
# pipeline pretrained
model_pretrained = helper.load_model_for_training("v3Large", 1000, pre_trained=True)  # TODO: Dropout: 0.1 0.2 0.3
model_pretrained.trainable = False
model_pretrained.summary()

##
# create own Model
inputs = tf.keras.Input(shape=(224, 224, 3))
# x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
x = model_pretrained(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
# x = tf.keras.layers.Dense(1000)(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model_transfer = tf.keras.Model(inputs, outputs)

history = helper.train_model(model_transfer, EPOCHS, train_ds, val_ds, "model_transfer_" + str(EPOCHS) + "epochs_"
                                + str(BATCH_SIZE) + "batch")
