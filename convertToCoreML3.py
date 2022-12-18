##
from keras.models import load_model
import coremltools as ct
import json
import tensorflow as tf
import helper_multitask


#PATH_TO_MODEL = "saved_model/modelv1_transfer_10epochs_128batch_025alpha"
#PATH_TO_MODEL = "saved_model/Milestone3/20221210-1618_classification10epochsface_10epochsmask_50epochsage_0.25alpha_0.5dropout_categoricalLoss_l2"
PATH_TO_MODEL = "saved_model/Milestone3/20221211-2234_regression10epochsface_10epochsmask_50epochsage_0.25alpha_0.2dropout"
model = load_model(PATH_TO_MODEL)


image_input = ct.ImageType(shape=(1, 224, 224, 3,))#,
                           #bias=[-1,-1,-1], scale=1/127)
#outputs=[ct.TensorType(name="face"),ct.TensorType(name="mask"),ct.TensorType(name="age")]
coremlmodel = ct.convert(model, inputs=[image_input], )#, classifier_config=classifier_config)

coremlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageClassifier"
coremlmodel.input_description["input"] = "Input image to be classified"
#coremlmodel.output_description["classLabel"] = "Most likely image category"
coremlmodel.author = 'Tobias Kaps & Svea Worms'
coremlmodel.short_description = 'Face Detection,Mask Detection and Age prediction with MobileNet 32 batch and 0.25 alpha'
coremlmodel.version = "1.0"
##
coremlmodel.save('Milestone3regression.mlmodel')
## example prediction
from PIL import Image
example_image = Image.open("/Users/tobias/Downloads/affe.jpg").resize((224, 224))
out_dict = coremlmodel.predict({"input": example_image})
print(out_dict)
