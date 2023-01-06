##
import os
import random

import pandas as pd
import numpy as np
import tensorflow as tf
import helper


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

    def call(self, inputs):
        return self.siamese_network(inputs)

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

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker]


##
def create_model(alpha=0.25, debug=False):
    # TODO: train with global average pooling and with flatten layer
    anchor_input = tf.keras.Input(name="anchor", shape=(224, 224, 3))
    positive_input = tf.keras.Input(name="positive", shape=(224, 224, 3))
    negative_input = tf.keras.Input(name="negative", shape=(224, 224, 3))
    input = tf.keras.Input(shape=(224, 224, 3), name='input')
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=alpha)
    model_pretrained.trainable = False
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(input)
    feature_generator = model_pretrained(feature_extractor)
    feature_generator = tf.keras.layers.GlobalAveragePooling2D()(feature_generator)
    # feature_generator = tf.keras.layers.Flatten()(feature_generator)
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

def triplet_accuracy(ap_distance, an_distance):
    margin = 0
    pred = (ap_distance - an_distance - margin).cpu().data
    return (pred > 0).sum()*1.0/ap_distance.size()[0]


def compile_model(model):
    model.compile(optimizer='adam', loss=triplet_loss, metrics=triplet_accuracy)
    return model


## CREATE THE DATASET

def decode_image(img_path):
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
    return decode_image(anchor), decode_image(positive), decode_image(negative)


def preprocess_triplets_array(filepath_anchor, filepath_positive, filepath_negative):
    return {"anchor": decode_image(filepath_anchor),
            "positive": decode_image(filepath_positive),
            "negative": decode_image(filepath_negative)}


#image_path = r"C:\Users\Svea Worms\PycharmProjects\Face-Detection\images\rawdata4"
image_path = r"C:\Users\svea\PycharmProjects\Face-Detection\images\rawdata4"
# images = sorted([str(image_path +  "/" +  f) for f in os.listdir(image_path)])

images = sorted([(str(image_path + "\\" + f), str(f.split("_")[0])) for f in os.listdir(image_path)])

images_df = pd.DataFrame(images, columns=["path", "class"])

len(images_df.groupby("class").count())


# train test split durchführen


##
def make_triplets(image_paths, image_classes, num=5):
    num_classes = image_classes.unique().tolist()
    digit_indices = [(np.where(image_classes == image_class)[0], image_class) for image_class in num_classes]
    triplets = []

    for anchor_id in range(len(image_paths)):
        # find anchor
        anchor_path = image_paths[anchor_id]
        anchor_class = image_classes[anchor_id]
        anchor_class_id = int(np.where(np.array(num_classes) == anchor_class)[0][0])

        # find matching example
        positive_id_list = [anchor_id]
        positive_id_list.extend(random.choices(digit_indices[anchor_class_id][0], k=num))

        # find non-matching example
        negative_class_list = random.choices(num_classes, k=num+1)
        while anchor_class in negative_class_list:
            negative_class_list = random.choices(num_classes, k=num+1)

        negative_id_list = []
        for negative_class in negative_class_list:
            negative_class_id = int(np.where(np.array(num_classes) == negative_class)[0][0])
            negative_id_list.append(random.choice(digit_indices[negative_class_id][0]))

        for i in range(len(positive_id_list)):
            positive_path = image_paths[positive_id_list[i]]
            negative_path = image_paths[negative_id_list[i]]
            triplets += [[anchor_path, positive_path, negative_path]]

            print(anchor_path)
            print(positive_path)
            print(negative_path + "\n")

        #print(len(triplets))

        # all triplets (ca. 6 Mio.)
        #
        # find matching example
        # for positive_id in digit_indices[anchor_class_id][0]:
        #    #positive_id = random.choice(digit_indices[anchor_class_id][0])
        #    if anchor_id == positive_id:
        #        continue
        #    positive_path = image_paths[positive_id]
        #
        # find non-matching example
        #    for negative_class in num_classes:
        #        if negative_class == anchor_class:
        #            continue
        #        negative_class_id = int(np.where(np.array(num_classes) == negative_class)[0][0])
        #        negative_id = random.choice(digit_indices[negative_class_id][0])
        #        negative_path = image_paths[negative_id]
        #        triplets += [[anchor_path, positive_path, negative_path]]

        #        print(anchor_path)
        #        print(positive_path)
        #        print(negative_path + "\n")

    return np.array(triplets)


##
# generate tensorflow dataset
triplets = make_triplets(images_df["path"], images_df["class"])

## online strategy

random.seed(0)
random.shuffle(triplets)
train_large = triplets[0:int(len(triplets) * 0.70)]
test_large = triplets[int(len(triplets) * 0.70):int(len(triplets) * 0.85)]
val_large = triplets[int(len(triplets) * 0.85):]

#train_small = train_large[0:int(len(train_large) * 0.1)]
#test_small = test_large[0:int(len(test_large) * 0.1)]
#val_small = val_large[0:int(len(val_large) * 0.1)]

#train = train_small
#test = test_small
#val = val_small

train = train_large
test = test_large
val = val_large

data_train = tf.data.Dataset.from_tensor_slices((train[:, 0], train[:, 1], train[:, 2]))
ds_train = data_train.map(preprocess_triplets_array)
ds_train = ds_train.batch(32)

data_test = tf.data.Dataset.from_tensor_slices((test[:, 0], test[:, 1], test[:, 2]))
ds_test = data_test.map(preprocess_triplets_array)
ds_test = ds_test.batch(32)

data_val = tf.data.Dataset.from_tensor_slices((val[:, 0], val[:, 1], val[:, 2]))
ds_val = data_val.map(preprocess_triplets_array)
ds_val = ds_val.batch(32)

##
EPOCHS = 10
model = create_model()
siamese_model = SiameseModel(model)
siamese_model = compile_model(siamese_model)
history = siamese_model.fit(ds_train, epochs=10, validation_data=ds_val)
