import os
import random
import shutil
import pandas as pd


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

def create_feature_table(directory, path):
    data = []
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
                image_path = subdirectory + "/" + file
                face = 0
                mask = 0
                age = -1
                if subdirectory.__contains__("face"):
                    face = 1
                    if folder == "face":
                        try:
                            age = int(file.split("_")[0])
                        except:
                            age = -1
                if subdirectory.__contains__("mask"):
                    mask = 1
                # create row in csv data
                row = {"filename": file, "image_path": image_path, "face": face, "mask": mask, "age": age}
                data.append(row)
    # save data to pandas Dataframe and to file
    df = pd.DataFrame(data)
    df.to_csv(path + ".csv")
