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
        files = os.listdir(directory + "/" + folder)
        random.seed(0)
        random.shuffle(files)
        train = files[0:int(len(files) * 0.85)]
        test = files[int(len(files) * 0.85):]
        for file in train:
            shutil.copy(directory + "/" + folder + "/" + file, "images/train/" + folder + "/" + file)
        for file in test:
            shutil.copy(directory + "/" + folder + "/" + file, "images/test/" + folder + "/" + file)


split_image_directory()


##

def create_feature_table(directory):
    folder_list = os.listdir(directory)
    for folder in folder_list:
        subdirectory = directory + "/" + folder
        for file in os.listdir(subdirectory):
            # prove if file is a folder
            if file.split(".").__len__() == 1:
                # add to folderList
                folder_list.append(file)
            else:
                # set feature parameter
                filename = file
                image_path = subdirectory + "/" + file
                face = 0
                mask = 0
                age = -1
                if subdirectory.__contains__("face"):
                    face = 1
                    if folder == "mask":
                        mask = 1
                    elif folder == "face":
                        try:
                            age = int(file.split("_")[0])
                        except:
                            age = -1
                #TODO: save in csv data
