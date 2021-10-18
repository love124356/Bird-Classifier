from __future__ import print_function, division
import os
import cv2
from PIL import Image
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


DATA_ROOT = r"./Data/"


def get_data(mode):
    VAL_RATIO = 0.2
    file = open(DATA_ROOT + "training_labels.txt", "r")
    train_name = []
    train_label = []
    for line in file.readlines():
        img_name, label = line.strip('\n').split(' ')
        train_name.append(img_name)
        label = int(label.split('.')[0]) - 1
        train_label.append(label)
    file.close()

    if mode == "train":
        percent = int(len(train_name) * (1 - VAL_RATIO))
        train_name = train_name[:percent]
        train_label = train_label[:percent]

        return train_name, train_label
    
    elif mode == "val":
        percent = int(len(train_name) * (1 - VAL_RATIO))
        train_name = train_name[percent:]
        train_label = train_label[percent:]

        return train_name, train_label


# Create Dataset
class BirdDataset(Dataset):
    """Bird species dataset."""

    def __init__(self, root_dir, mode, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_names, self.img_labels = get_data("train")
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,"training_images",
                                self.img_names[idx])
        self.img = Image.open(img_name)
        self.label = self.img_labels[idx]

        if self.transform:
            self.img = self.transform(self.img)

        return self.img, self.label



if __name__ == "__main__":
    img,label = get_data('train')
    # print(img)
    # print(label)
    print(max(label))
    img1,label1 = get_data('val')
    print(max(label1))
    BATCH_SIZE = 64
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        # transforms.RandomHorizontalFlip(1.0),
        # transforms.RandomVerticalFlip(1.0),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = BirdDataset(DATA_ROOT, "train", transform = train_transform)
    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
    for inputs, labels in train_loader:
        print(inputs)
        print(labels)