from __future__ import print_function, division
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import copy
from dataset import BirdDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {device}')

BATCH_SIZE = 64
DATA_ROOT = r"./data/"
LR = 1e-4
NUM_CLASSES = 200
IMG_SIZE = 224
MODEL_PATH = "./model/resnet50.pt"
NUM_EPOCHS = 200


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

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


# transform_options = [
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
#     transforms.RandomRotation(degrees=[-15, 15]),
#     transforms.GaussianBlur(kernel_size=3),
#     transforms.RandomAffine(0, shear=20)
# ]

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(IMG_SIZE),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #      transforms.RandomChoice(transform_options)
        # ], p = 0.9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# train_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

image_datasets = {x: BirdDataset(DATA_ROOT, x, transform=data_transforms[x])
                  for x in ["train", "val"]}

dataloaders = {}
dataloaders["train"] = DataLoader(image_datasets["train"],
                                  batch_size=BATCH_SIZE, shuffle=True)
dataloaders["val"] = DataLoader(image_datasets["val"],
                                batch_size=BATCH_SIZE, shuffle=True)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model_ft = model_ft.to(device)

for name, child in model_ft.named_children():
    if name in ['layer4', 'fc']:
        # print(name + ' is unfrozen')
        for param in child.parameters():
            param.requires_grad = True
    else:
        # print(name + ' is frozen')
        for param in child.parameters():
            param.requires_grad = False

print(model_ft)

optimizer = optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# fix random seed
same_seeds(0)

model_ft = train_model(model_ft, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=NUM_EPOCHS)
