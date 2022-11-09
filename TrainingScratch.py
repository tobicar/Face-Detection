## imports
import tensorflow as tf
import MobileNet

# Trainingspipeline:
BATCH_SIZES = [32, 64, 128, 256]
# BATCH_SIZE = 64  # TODO: Hier varieren mit 32, 64, 128, 256
EPOCHS = [10, 30, 50, 100]

# TODO: Crossvalidation
# TODO: Learning Rate Decay ???
# TODO: Early Stopping (Overfitting vermeiden)

##
for batch in BATCH_SIZES:
    for epoch in EPOCHS:
        # load train and validation data
        (train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=batch)
        # buffer datasets in RAM to prevent I/0 Blocking
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        # load net architecture
        model_scratch = MobileNet.load_model_for_training("v3Large", 1)  # TODO: Dropout: 0.1 0.2 0.3
        # load,train and save model
        name = "model_scratch_" + str(epoch) + "epochs_" + str(batch) + "batch"
        history = MobileNet.train_model(model_scratch, epoch, train_ds, val_ds, name)
        MobileNet.generate_history_and_save(history, name)
