##
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import helper

##
def create_model(alpha=0.25, debug=False):
    #TODO: train with global average pooling and with flatten layer
    left_input = tf.keras.Input(shape=(224, 224, 3), name='input_left')
    right_input = tf.keras.Input(shape=(224, 224, 3), name='input_right')
    input = tf.keras.Input(shape=(224, 224, 3), name='input')
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
    model_pretrained.trainable = False
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(input)
    feature_generator = model_pretrained(feature_extractor)
    feature_generator = tf.keras.layers.GlobalAveragePooling2D()(feature_generator)
    #feature_generator = tf.keras.layers.Flatten()(feature_generator)
    feature_generator = tf.keras.layers.Dropout(0.2)(feature_generator)
    feature_generator = tf.keras.layers.BatchNormalization()(feature_generator)
    feature_generator = tf.keras.layers.Dense(128, activation='relu')(feature_generator)

    feature_model = tf.keras.Model(input, feature_generator, name="feature_generator")
    if debug:
        feature_model.summary()
    encoded_l = feature_model(left_input)
    encoded_r = feature_model(right_input)
    subtracted = tf.keras.layers.Subtract()([encoded_l, encoded_r])
    l1_layer = tf.keras.layers.Lambda(lambda x: abs(x))(subtracted)
    prediction = tf.keras.layers.Dense(1, activation="sigmoid")(l1_layer)
    siamese_net = tf.keras.Model(inputs=[left_input, right_input], outputs=prediction)
    return siamese_net


def contrastive_loss(y_true, y_pred, margin=1):
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - y_pred, 0))
    return tf.math.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)


def compile_model(model):
    model.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])
    return model


## train the model
EPOCHS = 10
BATCH_SIZE = 32
model = create_model()
model = compile_model(model)
model.summary()
history = model.fit([x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)