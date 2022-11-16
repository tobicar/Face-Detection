##
import tensorflow as tf
import MobileNet

# train pipeline:
BATCH_SIZES = [64]
EPOCHS = [30, 10]
ALPHAS = [1]  # [1, 0.75, 0.5, 0.25, 1.25]
# DEPTH_MULTIPLIERS = [1, 0.75, 0.5]  # ERROR: scheint nicht richtig implementiert zu sein in MobileNet
INPUT_SIZES = [224]  # [192, 160, 128]
FULL_NAME = False

# TODO: Cross-validation
# TODO: Learning Rate Decay ???
# TODO: Early Stopping (avoid overfitting)

##
alpha = 1.0
for batch in BATCH_SIZES:
    for epoch in EPOCHS:
        for alpha in ALPHAS:
            for size in INPUT_SIZES:
                # load train and validation data
                (train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=batch, imagesize=size)
                # buffer datasets in RAM to prevent I/0 Blocking
                AUTOTUNE = tf.data.AUTOTUNE
                train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
                val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
                # load net architecture
                model_scratch = MobileNet.load_model_for_training("v1", 1)
                # load,train and save model
                name = "modelv1_scratch_" + str(epoch) + "epochs_" + str(batch) + "batch"
                if FULL_NAME:
                    name += "_" + str(alpha).split(".")[0] + str(alpha).split(".")[1] + "alpha_" + str(size) + "inputSize"
                history = MobileNet.train_model(model_scratch, epoch, train_ds, val_ds, name)
                MobileNet.generate_history_and_save(history, name)

##
# train on ResNet50

# for batch in BATCH_SIZES:
#    for epoch in [30, 50]:
#        # load train and validation data
#        (train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=batch)
#        # buffer datasets in RAM to prevent I/0 Blocking
#        AUTOTUNE = tf.data.AUTOTUNE
#        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
#        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
#        # load net architecture
#        model_scratch = MobileNet.load_resnet_model()
#        # load,train and save model
#        name = "model_resnet_" + str(epoch) + "epochs_" + str(batch) + "batch"
#        history = MobileNet.train_model(model_scratch, epoch, train_ds, val_ds, name)
#        MobileNet.generate_history_and_save(history, name)
