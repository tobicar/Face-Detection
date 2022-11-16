##
import tensorflow as tf
import helper

# train pipeline:
BATCH_SIZES = [32]
EPOCHS = [30, 10]
ALPHAS = [1, 0.75, 0.5, 0.25]
# DEPTH_MULTIPLIERS = [1, 0.75, 0.5]  # ERROR: scheint nicht richtig implementiert zu sein in MobileNet daher inputsize
INPUT_SIZES = [224]  # [192, 160, 128]

# TODO: Cross-validation
# TODO: Learning Rate Decay
# TODO: Early Stopping (avoid overfitting)

##
for batch in BATCH_SIZES:
    for epoch in EPOCHS:
        for alpha in ALPHAS:
            for size in INPUT_SIZES:
                # load train and validation data
                (train_ds, val_ds) = helper.import_train_images("images/train", batch_size=batch, imagesize=size)
                # buffer datasets in RAM to prevent I/0 Blocking
                AUTOTUNE = tf.data.AUTOTUNE
                train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
                val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
                # load net architecture
                model_scratch = helper.load_model_for_training("v1", 1)
                # load,train and save model
                name = "modelv1_scratch_" + str(epoch) + "epochs_" + str(batch) + "batch"
                if alpha != 1:
                    name += "_" + str(alpha).split(".")[0] + str(alpha).split(".")[1] + "alpha"
                if size != 224:
                    name += "_" + str(size) + "inputSize"
                history = helper.train_model(model_scratch, epoch, train_ds, val_ds, name)
                helper.generate_history_and_save(history, name)
