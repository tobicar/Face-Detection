##
import tensorflow as tf
import helper

#TODO: training triplet loss umstellen


##
class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)

##
def create_model(alpha=0.25, debug=False):
    #TODO: train with global average pooling and with flatten layer
    anchor_input = tf.keras.Input(name="anchor", shape=(224, 224, 3))
    positive_input = tf.keras.Input(name="positive", shape=(224, 224, 3))
    negative_input = tf.keras.Input(name="negative", shape=(224, 224, 3))
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

    distances = DistanceLayer()(
        feature_model(anchor_input),
        feature_model(positive_input),
        feature_model(negative_input),
    )

    siamese_net = tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
    return siamese_net


def triplet_loss(ap_distance, an_distance, margin=1):
    loss = ap_distance - an_distance
    loss = tf.maximum(loss + margin, 0.0)
    return loss


def compile_model(model):
    model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
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