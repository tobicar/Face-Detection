##
import random
import numpy as np
import tensorflow as tf
import helper

##
# functions for triplet loss model

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


class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.metrics.Mean(name="loss")

    def call(self, inputs, training=None, mask=None):
        #self.build(inputs.shape)
        return self.siamese_network(inputs, training=training, mask=mask)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def compile(
            self,
            optimizer="rmsprop",
            loss=None,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
            **kwargs,
    ):
        if metrics is None:
            metrics = []
        metrics.append(triplet_accuracy)
        super(SiameseModel, self).compile(optimizer, loss, metrics, **kwargs)

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


@tf.function
def triplet_loss(y_true, y_pred):
    """
    define triplet loss
    :param y_true: real value (not used in triplet loss)
    :param y_pred: predicted value
    :return: loss
    """
    margin = 1
    ap_distance = y_pred[:, 0, :]
    an_distance = y_pred[:, 1, :]
    loss = ap_distance - an_distance
    loss = tf.maximum(loss + margin, 0.0)
    return loss


@tf.function
def triplet_accuracy(y_true, y_pred):
    """
    define accuracy for triplet loss (does not work)
    :param y_true: real value (not used)
    :param y_pred: predicted value
    :return: accuracy
    """
    # Calculate the distance between the anchor point and the positive point
    pos_dist = tf.reduce_sum(tf.square(y_pred[:, 0, :] - y_pred[:, 1, :]), axis=1)
    # Calculate the distance between the anchor point and the negative point
    neg_dist = tf.reduce_sum(tf.square(y_pred[:, 0, :] - y_pred[:, 2, :]), axis=1)
    # Calculate the accuracy as the percentage of triplets for which the positive distance is less than the negative distance
    accuracy = tf.reduce_mean(tf.cast(pos_dist < neg_dist, dtype=tf.float32))
    return accuracy


def compile_model(model):
    """
    compile the given model with triplet attributes
    :param model: model to compile
    :return: compiled model
    """
    model.compile(optimizer='adam', loss=triplet_loss, metrics=[triplet_accuracy])
    return model


# CREATE THE DATASET

def decode_image(img_path):
    """
    decoding and preprocessing of image
    :param img_path: path to image
    :return: image in shape (244,244,3)
    """
    image_size = (224, 224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    # img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def preprocess_triplets(anchor, positive, negative):
    """
    generate triplets out of three inputs
    :param anchor: anchor
    :param positive: positive image
    :param negative: negative image
    :return: triplet / tuple with inputs
    """
    return decode_image(anchor), decode_image(positive), decode_image(negative)


def preprocess_triplets_array(filepath_anchor, filepath_positive, filepath_negative):
    """
    generate dictionary with paths to triplet components
    :param filepath_anchor: path to anchor
    :param filepath_positive: path to positive image
    :param filepath_negative: path to negative image
    :return: dictionary of paths
    """
    return {"anchor": decode_image(filepath_anchor),
            "positive": decode_image(filepath_positive),
            "negative": decode_image(filepath_negative)}

##


def make_triplets_utk(image_paths, image_paths_utk, image_classes, num=5, probability_array=[True, False]):
    """
    generate triplets from images image_paths. Create (num+1)-count triplets for each image class
    :param image_paths_utk: list of path to each image of utk
    :param image_paths: list of path to each image
    :param image_classes: list of image classes
    :param num: count of triplets (num+1)
    :param probability_array: probability of using utk image as negative. True = not use utk; False = use utk
    :return: numpy array of triplets containing path to images
    """
    # extract all classes and indices of images in it
    if probability_array is None:
        probability_array = [True, False]
    num_classes = image_classes.unique().tolist()
    digit_indices = [(np.where(image_classes == image_class)[0], image_class) for image_class in num_classes]

    triplets = []
    # loop over each class
    for anchor_id in range(len(image_paths)):
        # find anchor
        anchor_path = image_paths[anchor_id]
        anchor_class = image_classes[anchor_id]
        anchor_class_id = int(np.where(np.array(num_classes) == anchor_class)[0][0])

        # find matching example
        # one time anchor self
        positive_id_list = [anchor_id]
        # random images of class
        positive_id_list.extend(random.choices(digit_indices[anchor_class_id][0], k=num))

        # find non-matching example
        negative_path_list = []
        for i in range(num + 1):
            # decide random if image from utk or not
            if random.choice(probability_array):
                # get random negative class
                negative_class_id = int(random.choice(np.where(np.array(num_classes) == random.choice(num_classes))))
                while anchor_class_id == negative_class_id:
                    negative_class_id = int(
                        random.choice(np.where(np.array(num_classes) == random.choice(num_classes))))
                # get random image of random negative class
                negative_id = random.choice(digit_indices[negative_class_id][0])
                negative_path_list.append(image_paths[negative_id])
            else:
                # get random image of utk
                negative_path_list.append(random.choice(image_paths_utk))

        # create triplet
        for i in range(len(positive_id_list)):
            positive_path = image_paths[positive_id_list[i]]
            negative_path = negative_path_list[i]
            triplets += [[anchor_path, positive_path, negative_path]]

    return np.array(triplets)


@tf.function
def normalize(x):
    """
    define L2 regularisation
    """
    return tf.math.l2_normalize(x, axis=-1)


def create_model():
    """
    create model for triplet loss using mobilenet as base model
    :return: model
    """
    anchor_input = tf.keras.Input(name="anchor", shape=(224, 224, 3))
    positive_input = tf.keras.Input(name="positive", shape=(224, 224, 3))
    negative_input = tf.keras.Input(name="negative", shape=(224, 224, 3))
    input = tf.keras.Input(shape=(224, 224, 3), name='input')
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=1)
    model_pretrained.trainable = False
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(input)
    feature_generator = model_pretrained(feature_extractor)
    feature_generator = tf.keras.layers.GlobalAveragePooling2D()(feature_generator)
    # feature_generator = tf.keras.layers.Flatten()(feature_generator)
    feature_generator = tf.keras.layers.BatchNormalization()(feature_generator)
    feature_generator = tf.keras.layers.Dropout(0.4)(feature_generator)
    feature_generator = tf.keras.layers.Dense(256, activation='relu')(feature_generator)
    feature_generator = tf.keras.layers.BatchNormalization()(feature_generator)
    feature_generator = tf.keras.layers.Dropout(0.4)(feature_generator)
    feature_generator = tf.keras.layers.Dense(128, activation='relu')(feature_generator)
    feature_generator = tf.keras.layers.Lambda(normalize)(feature_generator)

    feature_model = tf.keras.Model(input, feature_generator, name="feature_generator")

    # distantf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    distances = DistanceLayer()(
        feature_model(anchor_input),
        feature_model(positive_input),
        feature_model(negative_input),
    )

    return tf.keras.Model(inputs=[anchor_input, positive_input, negative_input], outputs=distances)
