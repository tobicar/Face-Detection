from keras.models import load_model
import coremltools as ct
import tensorflow as tf


PATH_TO_MODEL = "saved_model/model_transfer_10epochs_32batch"
model = load_model(PATH_TO_MODEL)


image_input = ct.ImageType(shape=(1, 224, 224, 3,))#,
                           #bias=[-1,-1,-1], scale=1/127)

class_labels = ["face"]
classifier_config = ct.ClassifierConfig(class_labels)

coremlmodel = ct.convert(model,inputs=[image_input], classifier_config=classifier_config)

coremlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageClassifier"
coremlmodel.input_description["input_2"] = "Input image to be classified"
#coremlmodel.output_description["classLabel"] = "Most likely image category"

coremlmodel.author = 'Tobias Kaps & Svea Worms'
coremlmodel.short_description = 'Face Detection with MobileNetV3'
coremlmodel.version = "1.0"
coremlmodel.save('FaceDetection21.mlmodel')
