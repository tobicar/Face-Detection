##
import pandas as pd
import tensorflow as tf
import helper
import numpy as np

##
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
    age_detetion = tf.keras.layers.Dense(1, name="age_detection")(feature_extractor)

    model = tf.keras.Model(inputs = inputs, outputs = [face_detection, mask_detection, age_detetion])
    return model

##
def compileModel(model):
    model.compile(optimizer='adam', loss={'face_detection': 'binary_crossentropy',
                                          'mask_detection': 'binary_crossentropy',
                                          'age_detection': 'mse'},
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
model = createModel()
model = compileModel(model)
model_history = model.fit(dataset,epochs=15)
#model_history = model.fit({'input':x_train},
#                          {'face_detection': y_train_1,'mask_detection': y_train_2,'age_detection':y_train_3}, epochs=15)


##
tf.data.Dataset.from_tensor_slices()
## plot images from csv

#visualising the dataset

##
def get_label(label):
    #row =label_csv[label_csv["image_path"] == file_path]
    return {'face_detection': tf.reshape(label[0], (-1,1)),
            'mask_detection':  tf.reshape(label[1], (-1,1)),
            'age_detection': tf.reshape(label[2], (-1,1))}

##
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
def process_path(file_path,labels):
    label = get_label(labels)
    #label = {'face_detection': 1,'mask_detection': 2,'age_detection':3}
    # Load the raw data from the file as a string
    #img = tf.io.read_file(file_path)
    img = decode_img(file_path)
    #img = file_path
    return img, label

##
data = pd.read_csv("images/featureTable.csv")
dataset = tf.data.Dataset.from_tensor_slices((data["image_path"], data[["face","mask","age"]]))
##
train_ds = dataset.map(process_path)
train_ds = train_ds.shuffle(1).batch(32)

