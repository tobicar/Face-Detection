##
import pandas as pd
import tensorflow as tf
import helper
import numpy as np

##
def add_face(x):
    greater = tf.keras.backend.greater_equal(x, 0.5) #will return boolean values
    greater = tf.keras.backend.cast(greater, dtype=tf.keras.backend.floatx()) #will convert bool to 0 and 1
    return greater



def createModel():
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=0.25)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)

    # face detection
    face_detection = tf.keras.layers.Dense(1, activation="sigmoid", name='face_detection')(feature_extractor)

    # mask detecion
    mask_detection = tf.keras.layers.Dense(1,activation="sigmoid", name='mask_detection')(feature_extractor)

    # age detecion
    face_detection_ground_truth = tf.keras.layers.Lambda(add_face)(face_detection)
    age_detection = tf.keras.layers.Dense(250, activation="relu")(feature_extractor)
    age_detection = tf.keras.layers.Dense(1)(age_detection)
    age_detection = tf.keras.layers.multiply([age_detection, face_detection_ground_truth],name="age_detection")

    model = tf.keras.Model(inputs = inputs, outputs = [face_detection, mask_detection, age_detection])
    return model

##
def create_model_age():
    #TODO: Klassifizierung für Altersklassen (z.B. 10-20, 20-30, etc.)
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=0.25)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)
    age_detection = tf.keras.layers.Dense(102, activation="softmax", name="age_detection")(feature_extractor)
    model = tf.keras.Model(inputs=inputs, outputs=age_detection)
    return model

def create_model_age_regression():
    #TODO: alpha variieren
    #TODO: learning rate
    #TODO: validierung hinzufügen
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=0.25)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)
    #TODO: mehr Dense Layer
    feature_extractor = tf.keras.layers.Dense(1000)(feature_extractor)
    feature_extractor = tf.keras.layers.Dense(500)(feature_extractor)
    feature_extractor = tf.keras.layers.Dense(250)(feature_extractor)
    age_detection = tf.keras.layers.Dense(1, name="age_detection")(feature_extractor)
    model = tf.keras.Model(inputs=inputs, outputs=age_detection)
    return model

def compile_model_age_regression(model):
    model.compile(optimizer='adam', loss='mse',
                  metrics=['mse'])
    return model

def custom_sparse_categorical_crossentropy(y_true,y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred, ignore_class=-1)
def compile_model_age(model):
    model.compile(optimizer='adam', loss=custom_sparse_categorical_crossentropy,
                  metrics='accuracy')
    return model


def createModelV2():
    model_pretrained = helper.load_model_for_training("v1", 1000, pre_trained=True, alpha=0.25)
    model_pretrained.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    feature_extractor = tf.keras.applications.mobilenet.preprocess_input(inputs)
    feature_extractor = model_pretrained(feature_extractor, training=False)
    feature_extractor = tf.keras.layers.GlobalAveragePooling2D()(feature_extractor)
    feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)

    # face detection
    face_detection = tf.keras.layers.Dense(1, activation="sigmoid", name='face_detection')(feature_extractor)

    # mask detecion
    mask_detection = tf.keras.layers.Dense(1, activation="sigmoid", name='mask_detection')(feature_extractor)

    # age detecion
    # one Class for no age = 0
    # faces with unknown age = -1 --> ignored
    #  91 classes for ages between 10 and 100
    age_detection = tf.keras.layers.Dense(92, activation="softmax", name="age_detection")(feature_extractor)

    model = tf.keras.Model(inputs=inputs, outputs=[face_detection, mask_detection, age_detection])
    return model

##

def custom_accuracy(y_true,y_pred):
    pass


def compileModelV2(model):
    model.compile(optimizer='adam', loss={'face_detection': 'binary_crossentropy',
                                          'mask_detection': 'binary_crossentropy',
                                          'age_detection': custom_sparse_categorical_crossentropy},
                  loss_weights={'face_detection': 0.33, 'mask_detection': 0.33, 'age_detection': 0.33},
                  metrics={'face_detection': 'accuracy',
                           'mask_detection': 'accuracy',
                           'age_detection': 'accuracy'})
    return model
##

def customMSE(y_true,y_pred):
    mask = tf.keras.backend.cast(tf.keras.backend.not_equal(y_true,-1), tf.keras.backend.floatx())
    return tf.keras.losses.mse(y_true * mask, y_pred * mask)

def compileModel(model):
    model.compile(optimizer='adam', loss={'face_detection': 'binary_crossentropy',
                                          'mask_detection': 'binary_crossentropy',
                                          'age_detection': customMSE},
                  loss_weights={'face_detection': 0.33, 'mask_detection':0.33, 'age_detection': 0.33},
                  metrics={'face_detection': 'accuracy',
                                          'mask_detection': 'accuracy',
                                          'age_detection': 'mse'})
    return model

##
path3 = "images/test/no_face/00000000_(5).jpg"
path = "images/test/face/10_0_0_20161220222308131.jpg"
path2 = "/Users/tobias/Downloads/29177_1_mundschutz-typll-gruen_1.jpg"
image = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
image2 = tf.keras.preprocessing.image.load_img(path2, target_size=(224, 224))
image3 = tf.keras.preprocessing.image.load_img(path3, target_size=(224, 224))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr2 = tf.keras.preprocessing.image.img_to_array(image2)
input_arr3 = tf.keras.preprocessing.image.img_to_array(image3)
x_train = np.array([input_arr,input_arr2,input_arr3])
labels_face = np.array([1,1,0],dtype=np.float32)
labels_age = np.array([10,30,0],dtype=np.float32)
labels_mask = np.array([0,1,0],dtype=np.float32)
dataset = tf.data.Dataset.from_tensor_slices((x_train, {'face_detection':labels_face, 'mask_detection':labels_mask,'age_detection':labels_age})).batch(2)

##
@tf.function
def get_label(label):
    #if label[2] == 0:
    #    age = label[2]
    #else:
    #    age = label[2]-9
    #if only_age:
    #return {'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1,1))}
    #else:
    return {'face_detection': tf.reshape(tf.keras.backend.cast(label[0], tf.keras.backend.floatx()), (-1,1)),
               'mask_detection':  tf.reshape(tf.keras.backend.cast(label[1], tf.keras.backend.floatx()), (-1,1)),
                'age_detection': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1,1))} # -9 because there are no classes between age 0 and 9

##
@tf.function
def decode_img(img_path):
    image_size = (224,224)
    num_channels = 3
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=num_channels, expand_animations=False
    )
    img = tf.image.resize(img, image_size, method="bilinear")
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img
    #img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    #return tf.keras.preprocessing.image.img_to_array(img)

##
@tf.function
def process_path(file_path,labels):
    label = get_label(labels)
    #label = {'face_detection': 1,'mask_detection': 2,'age_detection':3}
    # Load the raw data from the file as a string
    #img = tf.io.read_file(file_path)
    img = decode_img(file_path)
    #img = file_path
    return img, label

##
data = pd.read_csv("images/featureTableTrain.csv")
train = tf.data.Dataset.from_tensor_slices((data["image_path"], data[["face","mask","age"]]))
train_ds = train.map(process_path)
train_ds = train_ds.shuffle(16437, seed=123, reshuffle_each_iteration=False).batch(64)
##
data_val = pd.read_csv("images/featureTableVal.csv")
val = tf.data.Dataset.from_tensor_slices((data_val["image_path"], data_val[["face","mask","age"]]))
val_ds = val.map(process_path)
val_ds = val_ds.shuffle(4097, seed=123, reshuffle_each_iteration=False).batch(64)
##

##
train_age = data[data["age"] > 10]
train_age = train_age[train_age["age"] <= 100]
dataset_age = tf.data.Dataset.from_tensor_slices((train_age["image_path"], train_age[["face","mask","age"]]))
train_ds_age = dataset_age.map(process_path)


##
model = createModelV2()
model = compileModelV2(model)
model_history = model.fit(train_ds,epochs=10, validation_data=val_ds)
#model_history = model.fit({'input':x_train},
#                          {'face_detection': y_train_1,'mask_detection': y_train_2,'age_detection':y_train_3}, epochs=15)



