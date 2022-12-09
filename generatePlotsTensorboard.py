##
import pandas as pd
import matplotlib.pyplot as plt

## modelv1_transfer_30epochs
train_acc = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/32batches/run-20221116-082410_train-tag-epoch_accuracy.csv")
val_acc = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/32batches/run-20221116-082410_validation-tag-epoch_accuracy.csv")
train_acc_64 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/64batches/run-20221116-083644_train-tag-epoch_accuracy.csv")
val_acc_64 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/64batches/run-20221116-083644_validation-tag-epoch_accuracy.csv")
train_acc_128 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/128batches/run-20221116-085356_train-tag-epoch_accuracy.csv")
val_acc_128 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/128batches/run-20221116-085356_validation-tag-epoch_accuracy.csv")
train_acc_256 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/256batches/run-20221116-090730_train-tag-epoch_accuracy.csv")
val_acc_256 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/256batches/run-20221116-090730_validation-tag-epoch_accuracy.csv")
train_loss = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/32batches/run-20221116-082410_train-tag-epoch_loss.csv")
val_loss = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/32batches/run-20221116-082410_validation-tag-epoch_loss.csv")
train_loss_64 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/64batches/run-20221116-083644_train-tag-epoch_loss.csv")
val_loss_64 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/64batches/run-20221116-083644_validation-tag-epoch_loss.csv")
train_loss_128 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/128batches/run-20221116-085356_train-tag-epoch_loss.csv")
val_loss_128 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/128batches/run-20221116-085356_validation-tag-epoch_loss.csv")
train_loss_256 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/256batches/run-20221116-090730_train-tag-epoch_loss.csv")
val_loss_256 = pd.read_csv(
    "presentation/milestone2/modelv1_transfer_30epochs/256batches/run-20221116-090730_validation-tag-epoch_loss.csv")
##
fig, ax = plt.subplots()
ax.plot(train_acc_256["Step"], train_acc_256["Value"], label="training")
ax.plot(val_acc_256["Step"], val_acc_256["Value"], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
plt.ylim([0.9,1.0])
plt.legend()
plt.title("train and validation accuracy")
plt.show()
#plt.savefig("presentation/modelv1_transfer_30epochs_256batches_run-20221116-090730-accuracy" + ".png")
##
fig, ax = plt.subplots()
ax.plot(train_loss_256["Step"], train_loss_256["Value"], label="training")
ax.plot(val_loss_256["Step"], val_loss_256["Value"], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
plt.title("train and validation loss")
plt.show()
plt.savefig("presentation/modelv1_transfer_30epochs_256batches_run-20221116-090730-loss" + ".png")



## modelv1_scratch_100epochs
train_acc = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/32batches/run-20221115-200829_train-tag-epoch_accuracy.csv")
val_acc = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/32batches/run-20221115-200829_validation-tag-epoch_accuracy.csv")
train_acc_64 = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/64batches/run-20221115-212035_train-tag-epoch_accuracy.csv")
val_acc_64 = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/64batches/run-20221115-212035_validation-tag-epoch_accuracy.csv")
train_loss = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/32batches/run-20221115-200829_train-tag-epoch_loss.csv")
val_loss = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/32batches/run-20221115-200829_validation-tag-epoch_loss.csv")
train_loss_64 = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/64batches/run-20221115-212035_train-tag-epoch_loss.csv")
val_loss_64 = pd.read_csv(
    "presentation/milestone2/modelv1_scratch_100epochs/64batches/run-20221115-212035_validation-tag-epoch_loss.csv")

##
fig, ax = plt.subplots()
ax.plot(train_acc_64["Step"], train_acc_64["Value"], label="training_64_batch", color="#cacaca")
ax.plot(val_acc_64["Step"], val_acc_64["Value"], label="validation_64_batch", color="#cacaca")
ax.plot(train_acc["Step"], train_acc["Value"], label="training_32_batch")
ax.plot(val_acc["Step"], val_acc["Value"], label="validation_32_batch")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
#plt.ylim([0.9,1.0])
plt.legend()
plt.title("train and validation accuracy")
plt.show()
plt.savefig("presentation/modelv1_scratch_100epochs-accuracy" + ".png")
##
fig, ax = plt.subplots()
ax.plot(train_loss_64["Step"], train_loss_64["Value"], label="training_64_batch", color="#cacaca")
ax.plot(val_loss_64["Step"], val_loss_64["Value"], label="validation_64_batch", color="#cacaca")
ax.plot(train_loss["Step"], train_loss["Value"], label="training_32_batch")
ax.plot(val_loss["Step"], val_loss["Value"], label="validation_32_batch")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
plt.title("train and validation loss")
plt.show()
plt.savefig("presentation/modelv1_scratch_100epochs-loss" + ".png")
