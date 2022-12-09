##
import pandas as pd
import matplotlib.pyplot as plt

## import data from files

train_face_acc = pd.read_csv(
    "presentation/milestone3/face_detection_accuracy_train.csv")
val_face_acc = pd.read_csv(
    "presentation/milestone3/face_detection_accuracy_validation.csv")

train_face_loss = pd.read_csv(
    "presentation/milestone3/face_detection_loss_train.csv")
val_face_loss = pd.read_csv(
    "presentation/milestone3/face_detection_loss_validation.csv")

train_mask_acc = pd.read_csv(
    "presentation/milestone3/mask_detection_accuracy_train.csv")
val_mask_acc = pd.read_csv(
    "presentation/milestone3/mask_detection_accuracy_validation.csv")
train_mask_loss = pd.read_csv(
    "presentation/milestone3/mask_detection_loss_train.csv")
val_mask_loss = pd.read_csv(
    "presentation/milestone3/mask_detection_loss_validation.csv")

train_age_acc_classification = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/age_detection_accuracy_train.csv")
val_age_acc_classification = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/age_detection_accuracy_validation.csv")
train_age_loss_classification = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/age_detection_loss_train.csv")
val_age_loss_classification = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/age_detection_loss_validation.csv")

train_age_acc_classification_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/L2_regularization/age_detection_accuracy_train.csv")
val_age_acc_classification_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/L2_regularization/age_detection_accuracy_validation.csv")
train_age_loss_classification_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/L2_regularization/age_detection_loss_train.csv")
val_age_loss_classification_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge/L2_regularization/age_detection_loss_validation.csv")

train_age_acc_classification_largeVersion = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/age_detection_accuracy_train.csv")
val_age_acc_classification_largeVersion = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/age_detection_accuracy_validation.csv")
train_age_loss_classification_largeVersion = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/age_detection_loss_train.csv")
val_age_loss_classification_largeVersion = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/age_detection_loss_validation.csv")

train_age_acc_classification_largeVersion_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/L2_regularization/age_detection_accuracy_train.csv")
val_age_acc_classification_largeVersion_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/L2_regularization/age_detection_accuracy_validation.csv")
train_age_loss_classification_largeVersion_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/L2_regularization/age_detection_loss_train.csv")
val_age_loss_classification_largeVersion_L2 = pd.read_csv(
    "presentation/milestone3/classification_ValOnlyAge_largeVersion/L2_regularization/age_detection_loss_validation.csv")

train_age_loss_regression = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge/age_detection_loss_train.csv")
val_age_loss_regression = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge/age_detection_loss_validation.csv")
train_age_mse_regression = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge/age_detection_mse_train.csv")
val_age_mse_regression = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge/age_detection_mse_validation.csv")
train_age_mae_regression = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge/age_detection_mae_train.csv")
val_age_mae_regression = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge/age_detection_mae_validation.csv")

train_age_loss_regression_largeVersion = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge_largeVersion/age_detection_loss_train.csv")
val_age_loss_regression_largeVersion = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge_largeVersion/age_detection_loss_validation.csv")
train_age_mse_regression_largeVersion = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge_largeVersion/age_detection_mse_train.csv")
val_age_mse_regression_largeVersion = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge_largeVersion/age_detection_mse_validation.csv")
train_age_mae_regression_largeVersion = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge_largeVersion/age_detection_mae_train.csv")
val_age_mae_regression_largeVersion = pd.read_csv(
    "presentation/milestone3/regression_ValOnlyAge_largeVersion/age_detection_mae_validation.csv")

## age classification large-small plots
fig, ax = plt.subplots()
ax.plot(train_age_acc_classification["Step"], train_age_acc_classification["Value"], label="training")
ax.plot(val_age_acc_classification["Step"], val_age_acc_classification["Value"], label="validation")
ax.plot(train_age_acc_classification_largeVersion["Step"], train_age_acc_classification_largeVersion["Value"], label="training_large")
ax.plot(val_age_acc_classification_largeVersion["Step"], val_age_acc_classification_largeVersion["Value"], label="validation_large")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.legend()
plt.ylim([0.0, 0.5])
plt.title("Train and Validation Accuracy")
plt.show()
plt.savefig("plots/milestone3/classification-small-large-accuracy.png")



## age classification accuracy multiple plots
fig, ax = plt.subplots()
ax.plot(train_age_acc_classification["Step"], train_age_acc_classification["Value"], label="training")
ax.plot(val_age_acc_classification["Step"], val_age_acc_classification["Value"], label="validation")
ax.plot(train_age_acc_classification_L2["Step"], train_age_acc_classification_L2["Value"], label="training_l2")
ax.plot(val_age_acc_classification_L2["Step"], val_age_acc_classification_L2["Value"], label="validation_l2")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.legend()
plt.ylim([0.0, 0.5])
plt.title("Train and Validation Accuracy")
plt.show()
plt.savefig("plots/milestone3/classification-multiple-accuracy-smallVersion.png")

## age classification loss multiple plots

fig, ax = plt.subplots()
ax.plot(train_age_loss_classification_largeVersion["Step"], train_age_loss_classification_largeVersion["Value"],
        label="training_large")
ax.plot(val_age_loss_classification_largeVersion["Step"], val_age_loss_classification_largeVersion["Value"],
        label="validation_large")
ax.plot(train_age_loss_classification_largeVersion_L2["Step"], train_age_loss_classification_largeVersion_L2["Value"],
        label="training_large_l2")
ax.plot(val_age_loss_classification_largeVersion_L2["Step"], val_age_loss_classification_largeVersion_L2["Value"],
        label="validation_large_l2")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
plt.ylim([1, 1.7])
plt.title("Train and Validation Loss")
plt.show()
plt.savefig("plots/milestone3/classification-multiple-loss-largeVersion.png")

## age regression loss multiple plots

fig, ax = plt.subplots()
ax.plot(train_age_loss_regression["Step"], train_age_loss_regression["Value"], label="training_small")
ax.plot(val_age_loss_regression["Step"], val_age_loss_regression["Value"], label="validation_small")
ax.plot(train_age_loss_regression_largeVersion["Step"], train_age_loss_regression_largeVersion["Value"],
        label="training_large")
ax.plot(val_age_loss_regression_largeVersion["Step"], val_age_loss_regression_largeVersion["Value"],
        label="validation_large")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
plt.ylim([1, 2.4])
plt.title("Train and Validation Loss")
plt.show()
plt.savefig("plots/milestone3/regression-multiple-loss.png")

## age regression mse multiple plots

fig, ax = plt.subplots()
ax.plot(train_age_mse_regression["Step"], train_age_mse_regression["Value"], label="training_small")
ax.plot(val_age_mse_regression["Step"], val_age_mse_regression["Value"], label="validation_small")
ax.plot(train_age_mse_regression_largeVersion["Step"], train_age_mse_regression_largeVersion["Value"],
        label="training_large")
ax.plot(val_age_mse_regression_largeVersion["Step"], val_age_mse_regression_largeVersion["Value"],
        label="validation_large")
ax.set_xlabel("epoch")
ax.set_ylabel("mse")
plt.legend()
# plt.ylim([1, 1.7])
plt.title("Train and Validation MSE")
plt.show()
plt.savefig("plots/milestone3/regression-multiple-mse.png")

## age regression mae multiple plots

fig, ax = plt.subplots()
ax.plot(train_age_mae_regression["Step"], train_age_mae_regression["Value"], label="training_small")
ax.plot(val_age_mae_regression["Step"], val_age_mae_regression["Value"], label="validation_small")
ax.plot(train_age_mae_regression_largeVersion["Step"], train_age_mae_regression_largeVersion["Value"],
        label="training_large")
ax.plot(val_age_mae_regression_largeVersion["Step"], val_age_mae_regression_largeVersion["Value"],
        label="validation_large")
ax.set_xlabel("epoch")
ax.set_ylabel("mae")
plt.legend()
# plt.ylim([0.8, 1.2])
plt.title("Train and Validation MAE")
plt.show()
plt.savefig("plots/milestone3/regression-multiple-mae.png")

## mask plot
fig, ax = plt.subplots()
ax.plot(train_mask_acc["Step"], train_mask_acc["Value"], label="training")
ax.plot(val_mask_acc["Step"], val_mask_acc["Value"], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.legend()
plt.ylim([0.9, 1])
plt.title("Train and Validation Accuracy")
plt.show()
plt.savefig("plots/milestone3/mask-detection-accuracy.png")

## face detection accuracy plot
fig, ax = plt.subplots()
ax.plot(train_face_acc["Step"], train_face_acc["Value"], label="training")
ax.plot(val_face_acc["Step"], val_face_acc["Value"], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.legend()
plt.ylim([0.9, 1])
plt.title("Train and Validation Accuracy")
plt.show()
plt.savefig("plots/milestone3/face-detection-accuracy.png")

## face detection loss plot
fig, ax = plt.subplots()
ax.plot(train_face_loss["Step"], train_face_loss["Value"], label="training")
ax.plot(val_face_loss["Step"], val_face_loss["Value"], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
plt.ylim([0.0, 0.1])
plt.title("Train and Validation Loss")
plt.show()
plt.savefig("plots/milestone3/face-detection-loss.png")
