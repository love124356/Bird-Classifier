from __future__ import print_function, division
import torch
import numpy as np
import torchvision.transforms.functional as TF
from dataset import BirdDataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

DATA_ROOT = r"./data/"
BATCH_SIZE = 4 * 3
IMG_SIZE_W = 375
IMG_SIZE_H = 375
MODEL_PATH = "./model/ResNeXt101_32x8d.pt"


def pad(img, fill=0, size_max=500):
    """
    Pads images to the specified size (height x width).
    Fills up the padded area with value(s) passed to the `fill` parameter.
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    return TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
# pad images to 500 x 500 pixels
max_padding = transforms.Lambda(lambda x: pad(x, fill=fill))

with open(DATA_ROOT + 'testing_img_order.txt') as f:
    # all the testing images
    test_images = [x.strip().split(' ')[0] for x in f.readlines()]

with open(DATA_ROOT + 'classes.txt') as f:
    # all the classes
    classes = [x.strip().split(' ')[0] for x in f.readlines()]

test_transform = transforms.Compose([
    max_padding,
    transforms.CenterCrop((IMG_SIZE_H, IMG_SIZE_W)),
    transforms.ToTensor(),
])

test_set = BirdDataset(DATA_ROOT, "test", transform=test_transform)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


test_model = torch.load(MODEL_PATH, map_location=torch.device(device))
test_model.eval()


predict = []
test_model.eval()  # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = test_model(inputs)
        # get the index of the class with the highest probability
        _, test_pred = torch.max(outputs, 1)
        for y in test_pred.cpu().numpy():
            predict.append(y)


submission = []
for i, y in enumerate(predict):
    submission.append([test_images[i], classes[y]])

np.savetxt(DATA_ROOT + 'answer.txt', submission, fmt='%s')
