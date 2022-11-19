from keras.models import load_model
import coremltools as ct
import tensorflow as tf


#PATH_TO_MODEL = "saved_model/modelv1_transfer_10epochs_128batch_025alpha"
PATH_TO_MODEL = "saved_model/modelv1_scratch_75epochs_32batch_025alpha_1depthMultiplier"
model = load_model(PATH_TO_MODEL)


image_input = ct.ImageType(shape=(1, 224, 224, 3,))#,
                           #bias=[-1,-1,-1], scale=1/127)

class_labels = ["face"]
classifier_config = ct.ClassifierConfig(class_labels)

coremlmodel = ct.convert(model,inputs=[image_input], classifier_config=classifier_config)

coremlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageClassifier"
coremlmodel.input_description["input_3"] = "Input image to be classified"
#coremlmodel.output_description["classLabel"] = "Most likely image category"

coremlmodel.author = 'Tobias Kaps & Svea Worms'
coremlmodel.short_description = 'Face Detection with MobileNet 32 batch and 0.25 alpha'
coremlmodel.version = "1.0"
coremlmodel.save('FaceDetectionScratch.mlmodel')
## example prediction
from PIL import Image
example_image = Image.open("/Users/tobias/Downloads/affe.jpg").resize((224, 224))
out_dict = coremlmodel.predict({"input_2": example_image})
print(out_dict["classLabel"])
