##
import tensorflow as tf
import MobileNet

# train pipeline:
BATCH_SIZES = [32]
EPOCHS = [75]

ALPHAS = [1, 0.75, 0.5, 0.25, 1.25, 1.5, 1.75, 2]

DEPTH_MULTIPLIERS = [1, 0.75, 0.5]  # ERROR: scheint nicht richtig implementiert zu sein in MobileNet

# TODO: Cross-validation
# TODO: Learning Rate Decay ???
# TODO: Early Stopping (avoid overfitting)

##
depth_multiplier = 1
alpha = 1
for batch in BATCH_SIZES:
    for epoch in EPOCHS:
        for alpha in ALPHAS:
        #for depth_multiplier in DEPTH_MULTIPLIERS:
            if depth_multiplier == 1 and alpha == 1:
                continue
            print("alpha: " + str(alpha) + "\ndepth multiplier: " + str(depth_multiplier))
            # load train and validation data
            (train_ds, val_ds) = MobileNet.import_train_images("images/train", batch_size=batch)
            # buffer datasets in RAM to prevent I/0 Blocking
            AUTOTUNE = tf.data.AUTOTUNE
            train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
            val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
            # load net architecture
            model_scratch = MobileNet.load_model_for_training("v1", 1, alpha=alpha,
                                                              depth_multiplier=depth_multiplier)
            # load,train and save model
            name = "modelv1_scratch_" + str(epoch) + "epochs_" + str(batch) + "batch_" + str(alpha).split(".")[0] +\
                   str(alpha).split(".")[1] + "alpha_" + str(depth_multiplier) + "depthMultiplier"
            history = MobileNet.train_model(model_scratch, epoch, train_ds, val_ds, name)
            MobileNet.generate_history_and_save(history, name)

##
# train on ResNet50

#for batch in BATCH_SIZES:
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
