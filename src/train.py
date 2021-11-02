from __future__ import print_function, division
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import time
import copy
from dataset import BirdDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

BATCH_SIZE = 4
DATA_ROOT = r"./data/"
LR = 0.001
NUM_CLASSES = 200
IMG_SIZE_H = 375
IMG_SIZE_W = 375
MODEL_PATH = "./model/ResNeXt101_32x8d.pt"
GRAD_CLIP = 1
NUM_EPOCHS = 200


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if GRAD_CLIP:
                            nn.utils.clip_grad_value_(model.parameters(),
                                                      GRAD_CLIP)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                print(epoch, scheduler.get_last_lr()[0])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, MODEL_PATH)
                print('Save model. Its acc is {:.4f}'.format(epoch_acc))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # torch.save(model, model_path)

    return model


# fill padded area with ImageNet's mean pixel value converted to range [0, 255]
fill = tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406)))
# pad images to 500 pixels
max_padding = transforms.Lambda(lambda x: pad(x, fill=fill))

data_transforms = {
    'train': transforms.Compose([
        max_padding,
        transforms.CenterCrop((IMG_SIZE_H, IMG_SIZE_W)),
        transforms.RandomOrder([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        max_padding,
        transforms.CenterCrop((IMG_SIZE_H, IMG_SIZE_W)),
        transforms.ToTensor(),
    ]),
}

image_datasets = {x: BirdDataset(DATA_ROOT, x, transform=data_transforms[x])
                  for x in ["train", "val"]}

dataloaders = {}
dataloaders["train"] = DataLoader(image_datasets["train"],
                                  batch_size=BATCH_SIZE, shuffle=True)
dataloaders["val"] = DataLoader(image_datasets["val"],
                                batch_size=BATCH_SIZE*3, shuffle=False)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

model_ft = models.resnext101_32x8d(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, NUM_CLASSES))
model_ft = model_ft.to(device)

for param in model_ft.parameters():
    # All layers unfreeze
    param.requires_grad = True

optimizer = optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9,
                      weight_decay=3e-4)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

# fix random seed
same_seeds(0)

model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)
