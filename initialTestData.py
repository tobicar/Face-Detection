##
import os
import random
import shutil
import pandas as pd

DIRECTORY = "images/rawdata"


def split_image_directory(directory):
    """
    split folder of images in train (85%) and test (15%)
    :param directory: directory of image folders
    :return: -
    """
    for folder in os.listdir(directory):
        files = os.listdir(directory + "/" + folder)
        random.seed(0)
        random.shuffle(files)
        train = files[0:int(len(files) * 0.85)]
        test = files[int(len(files) * 0.85):]
        for file in train:
            shutil.copy(directory + "/" + folder + "/" + file, "images/train2/" + folder + "/" + file)
        for file in test:
            shutil.copy(directory + "/" + folder + "/" + file, "images/test2/" + folder + "/" + file)


def split_image_directory_hierarchical(directory):
    """
    split folder with hierarchical order of images in train (85%) and test (15%)
    :param directory: directory of image folders
    :return: -
    """
    folders = os.listdir(directory)
    folder_list = []
    for folder in folders:
        folder_list.append(directory + "/" + folder)
    for subdirectory in folder_list:
        # split files and folders
        file_list = []
        for file in os.listdir(subdirectory):
            # prove if file is a folder
            if file.split(".").__len__() == 1:
                # add folder to folder_list
                folder_list.append(subdirectory + "/" + file)
            else:
                # add file to file_list
                file_list.append(file)
        # split list of files into train and test
        random.seed(0)
        random.shuffle(file_list)
        train = file_list[0:int(len(file_list) * 0.85)]
        test = file_list[int(len(file_list) * 0.85):]
        for file in train:
            shutil.copy(subdirectory + "/" + file, "images/train2" + subdirectory[directory.__len__():] + "/" + file)
        for file in test:
            shutil.copy(subdirectory + "/" + file, "images/test2" + subdirectory[directory.__len__():] + "/" + file)


##

def create_feature_table(directory, path):
    """
    create csv data with table of features in images
    :param directory: directory of image folders
    :param path: directory of csv data
    :return: -
    """
    data = []
    folders = os.listdir(directory)
    folder_list = []
    for folder in folders:
        folder_list.append(directory + "/" + folder)
    for subdirectory in folder_list:
        for file in os.listdir(subdirectory):
            # prove if file is a folder
            if file.split(".").__len__() == 1:
                # add to folderList
                folder_list.append(subdirectory + "/" + file)
            else:
                # set feature parameter
                image_path = subdirectory + "/" + file
                face = 0
                mask = 0
                age = -1
                if subdirectory.__contains__("/face/"):
                    face = 1
                    parts = subdirectory.split("/")
                    if parts[parts.__len__() - 1] == "face":
                        try:
                            filename_parts = file.split("_")
                            if filename_parts.__len__() > 1:
                                age = int(filename_parts[0])
                        except:
                            age = -1
                if subdirectory.__contains__("/mask/"):
                    mask = 1
                # create row in csv data
                row = {"filename": file, "image_path": image_path, "face": face, "mask": mask, "age": age}
                data.append(row)
    # save data to pandas Dataframe and to file
    df = pd.DataFrame(data)
    df.to_csv(path + ".csv")
