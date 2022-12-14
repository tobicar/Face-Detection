##
import tensorflow as tf
import helper

# train pipeline:
BATCH_SIZES = [128]
EPOCHS = [10]
ALPHAS = [0.75, 0.5, 0.25]

# TODO: Cross-validation
# TODO: Learning Rate Decay
# TODO: Early Stopping (avoid Overfitting)

##
for batch in BATCH_SIZES:
    for epoch in EPOCHS:
        for alpha in ALPHAS:
            # load train and validation data
            (train_ds, val_ds) = helper.import_train_images("images/train", batch_size=batch)
            # buffer datasets in RAM to prevent I/0 Blocking
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
            # pipeline pretrained
            model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
            model_pretrained.trainable = False
            # create own model
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = tf.keras.applications.mobilenet.preprocess_input(inputs)
            x = model_pretrained(x, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            # x = tf.keras.layers.Dense(1000)(x)
            outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
            model_transfer = tf.keras.Model(inputs, outputs)
            name = "modelv1_transfer_" + str(epoch) + "epochs_" + str(batch) + "batch"
            if alpha != 1:
                name += "_" + str(alpha).split(".")[0] + str(alpha).split(".")[1] + "alpha"
            history = helper.train_model(model_transfer, epoch, train_ds, val_ds, name)
            helper.generate_history_and_save(history, name)
