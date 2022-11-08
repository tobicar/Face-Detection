import os
import random
import shutil


def split_image_directory():
    """
    split folder of images in train (85%) and test (15%)
    :return: -
    """
    directory = "images/rawdata"
    for folder in os.listdir(directory):
        files = os.listdir(directory+"/"+folder)
        random.seed(0)
        random.shuffle(files)
        train = files[0:int(len(files)*0.85)]
        test = files[int(len(files)*0.85):]
        for file in train:
            shutil.copy(directory + "/" + folder + "/" + file, "images/train/" + folder + "/" + file)
        for file in test:
            shutil.copy(directory + "/" + folder + "/" + file, "images/test/" + folder + "/" + file)
