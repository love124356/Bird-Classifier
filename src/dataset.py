from __future__ import print_function, division
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATA_ROOT = r"./data/"
TRAIN_RATIO = 0.8


def train_test_split(images, labels):

    data = dict()
    train_image, train_label, val_image, val_label = [], [], [], []

    for index, label in enumerate(labels):
        if label not in data:
            data[label] = [images[index]]
        else:
            data[label].append(images[index])

    for key in data:
        train_image += (data[key][:int(len(data[key])*TRAIN_RATIO)])
        train_label += [key]*len(data[key][:int(len(data[key])*TRAIN_RATIO)])
        val_image += (data[key][int(len(data[key])*TRAIN_RATIO):])
        val_label += [key]*len(data[key][int(len(data[key])*TRAIN_RATIO):])

    return train_image, train_label, val_image, val_label


def get_data(mode):

    if mode == "test":
        with open(DATA_ROOT + 'testing_img_order.txt') as f:
            test_name = [x.strip().split(' ')[0] for x in f.readlines()]

        return test_name, None

    file = open(DATA_ROOT + "training_labels.txt", "r")
    train_name = []
    train_label = []
    for line in file.readlines():
        img_name, label = line.strip('\n').split(' ')
        train_name.append(img_name)
        label = int(label.split('.')[0]) - 1
        train_label.append(label)
    file.close()

    train_name, train_label, val_name, val_label = train_test_split(
                                            train_name, train_label)

    if mode == "train":
        return train_name, train_label

    elif mode == "val":
        return val_name, val_label


# Create Dataset
class BirdDataset(Dataset):
    """Bird species dataset."""

    def __init__(self, root_dir, mode, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            mode (string): which set (training, validation, or testing).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_names, self.img_labels = get_data(mode)
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.img_labels is not None:
            img_name = os.path.join(self.root_dir, "training_images",
                                    self.img_names[idx])
            self.img = Image.open(img_name)
            self.label = self.img_labels[idx]
            if self.transform:
                self.img = self.transform(self.img)

            return self.img, self.label

        else:
            img_name = os.path.join(self.root_dir, "testing_images",
                                    self.img_names[idx])
            self.img = Image.open(img_name)
            if self.transform:
                self.img = self.transform(self.img)

            return self.img
