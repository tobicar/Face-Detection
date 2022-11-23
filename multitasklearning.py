##
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##
train_data = pd.read_csv('/Users/tobias/Downloads/MTFL/training.txt', sep=' ', header=None, skipinitialspace=True, nrows=10000)
test_data = pd.read_csv('/Users/tobias/Downloads/MTFL/testing.txt', sep=' ', header=None, skipinitialspace=True, nrows=2995)
##
train_data.iloc[:, 0] = train_data.iloc[:, 0].apply(lambda s: s.replace('\\', '/')) # Needed for filename convention
test_data.iloc[:, 0] = test_data.iloc[:, 0].apply(lambda s: s.replace('\\', '/')) # Needed for filename convention
##
# Load filenames and labels
filenames = tf.constant(train_data.iloc[:, 0].tolist())
labels = tf.constant(train_data.iloc[:, 1:].values)

# Add to a dataset object
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# We can debug using eager execution
for img, labels in dataset.batch(4).take(1):
  print(img)
  print(labels)
##
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.io.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=3) # Channels needed because some test images are b/w
  image_resized = tf.image.resize(image_decoded, [40, 40])
  image_shape = tf.cast(tf.shape(image_decoded), tf.float32)
  label = tf.concat([label[0:5] / image_shape[0], label[5:10] / image_shape[1], label[10:]], axis=0)
  return {"x": image_resized}, label
##
# This snippet is adapted from here: https://www.tensorflow.org/guide/datasets
def input_fn(dataframe, is_eval=False):
    # Load the list of files
    filenames = tf.constant(dataframe.iloc[:, 0].tolist())

    # Load the labels
    labels = tf.constant(dataframe.iloc[:, 1:].values.astype(np.float32))

    # Build the dataset with image processing on top of it
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)

    # Add shuffling and repeatition if training
    if is_eval:
        dataset = dataset.batch(64)
    else:
        dataset = dataset.repeat().shuffle(1000).batch(64)

    return dataset
##
#check the image
for (imgs, labels) in input_fn(train_data, is_eval=True).take(1):
  plt.imshow(imgs['x'][0] / 255)
  print(labels[0])
##
def extract_features(features):
    # Input layer
    input_layer = tf.reshape(features["x"], [-1, 40, 40, 3])

    # First convolutive layer
    conv1 = tf.keras.layers.Conv2D(inputs=input_layer, filters=16, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.keras.layers.MaxPooling2D(inputs=conv1, pool_size=[2, 2], strides=2)

    # Second convolutive layer
    conv2 = tf.keras.layers.Conv2D(inputs=pool1, filters=48, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool2 = tf.keras.layers.MaxPooling2D(inputs=conv2, pool_size=[2, 2], strides=2)

    # Third convolutive layer
    conv3 = tf.keras.layers.Conv2D(inputs=pool2, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    pool3 = tf.keras.layers.MaxPooling2D(inputs=conv3, pool_size=[2, 2], strides=2)

    # Fourth convolutive layer
    conv4 = tf.keras.layers.Conv2D(inputs=pool3, filters=64, kernel_size=[2, 2], padding="same", activation=tf.nn.relu)

    # Dense Layer
    flat = tf.reshape(conv4, [-1, 5 * 5 * 64])
    dense = tf.keras.layers.Dense(inputs=flat, units=100, activation=tf.nn.relu)

    return dense

##
# Adapted from here: https://www.tensorflow.org/tutorials/layers
def single_task_cnn_model_fn(features, labels, mode):
    # Get features
    dense = extract_features(features)

    # Make predictions
    predictions = tf.keras.layers.Dense(inputs=dense, units=2)

    outputs = {
        "predictions": predictions
    }

    # We just want the predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=outputs)

    # If not in mode.PREDICT, compute the loss (mean squared error)
    loss = tf.losses.mean_squared_error(labels=labels[:, 2:8:5], predictions=predictions)

    # Single optimization step
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.optimizers.Adam()#tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # If not PREDICT or TRAIN, then we are evaluating the model
    eval_metric_ops = {
        "rmse": tf.metrics.root_mean_squared_error(
            labels=labels[:, 2:8:5], predictions=outputs["predictions"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

