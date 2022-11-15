##
import pandas as pd
import matplotlib.pyplot as plt

##
# change name of the run here, to generate plots
RUN_NAME = "run-20221111-163248"
TRAIN_ACC_PATH = "presentation/" + RUN_NAME + "_train-tag-epoch_accuracy.csv"
VAL_ACC_PATH = "presentation/" + RUN_NAME + "_validation-tag-epoch_accuracy.csv"
TRAIN_LOSS_PATH = "presentation/" + RUN_NAME + "_train-tag-epoch_loss.csv"
VAL_LOSS_PATH = "presentation/" + RUN_NAME + "_validation-tag-epoch_loss.csv"
train_acc = pd.read_csv(TRAIN_ACC_PATH)
val_acc = pd.read_csv(VAL_ACC_PATH)
train_loss = pd.read_csv(TRAIN_LOSS_PATH)
val_loss = pd.read_csv(VAL_LOSS_PATH)
##
fig, ax = plt.subplots()
ax.plot(train_acc["Step"], train_acc["Value"], label="train")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.plot(val_acc["Step"], val_acc["Value"], label="val")
plt.legend()
plt.title("train and validation accuracy")
plt.show()
plt.savefig("presentation/" + RUN_NAME + "-accuracy" + ".png")
##
fig, ax = plt.subplots()
ax.plot(train_loss["Step"], train_loss["Value"], label="train")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.plot(val_loss["Step"], val_loss["Value"], label="val")
plt.legend()
plt.title("train and validation loss")
plt.show()
plt.savefig("presentation/" + RUN_NAME + "-loss" + ".png")
