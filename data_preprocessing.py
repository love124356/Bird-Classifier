import os
import cv2
import numpy as np


def extract_test_data(dir_test):
    """
        Read all the pictures in the testing_images folder
        and put the values into an array.
    """
    img_array = []
    for filename in os.listdir(dir_test):
        img = cv2.imread(dir_test + "/" + filename)
        img_array.append(img)

    return img_array


def extract_training_data_label(dir_label, dir_train):
    """
        First read the txt file and set the picture name to key
        and the category to value using the directory data type.
        Then read all the pictures in the training_images folder
        ,put the values into an array and find the corresponding
        category in the directory and put it into another array.
    """
    file = open(dir_label, "r")
    train_dict = {}
    for line in file.readlines():
        img_name, label = line.strip('\n').split(' ')
        train_dict[img_name] = label
    file.close()

    train_data = []
    train_label = []
    for filename in os.listdir(dir_train):
        img = cv2.imread(dir_train + "/" + filename)
        train_data.append(img)
        train_label.append(train_dict[filename])

    return train_data, train_label

DATA_ROOT = r"./Data/"
test = extract_test_data(DATA_ROOT + "testing_images")
train, trainLabel = extract_training_data_label(
    DATA_ROOT + "training_labels.txt", DATA_ROOT + "training_images")

if not os.path.isdir("Data"):
    os.mkdir("Data")

np.save('Data/test', test)
np.save('Data/train', train)
np.save('Data/trainLabel', trainLabel)

print("End data preprocessing.")
