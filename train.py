import torch
from data_preprocessing_tmp import BirdDataset
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {device}')

BATCH_SIZE = 64
DATA_ROOT = r"./Data/"
LR = 1e-4
NUM_CLASSES = 200
IMG_SIZE = 224

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


train_set = BirdDataset(DATA_ROOT, "train", transform = data_transforms['train'])
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True)
val_set = BirdDataset(DATA_ROOT, "val", transform = data_transforms['val'])
val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = False)

resnet50 = models.resnet50(pretrained = True)
num_ft = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ft, NUM_CLASSES)
# resnet50 = resnet50.to(device)

if torch.cuda.is_available():
    resnet50.cuda(0)
# print(f'DEVICE: {device}')

optimizer = optim.Adam(resnet50.parameters(), lr = LR)
criterion = nn.CrossEntropyLoss()
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

same_seeds(0)

# summary(resnet50, (3, 256, 256))
print(resnet50)

for name,child in resnet50.named_children():
    if name in ['layer4','fc']:
        # print(name + ' is unfrozen')
        for param in child.parameters():
            param.requires_grad = True
    else:
        # print(name + ' is frozen')
        for param in child.parameters():
            param.requires_grad = False

# ct = 0
# for name, child in model_init.named_children():
#     print(f'name={name}, child={child}')
#     if ct < NUM_FREEZE_LAYERS:
#         print(f'Freeze->{ct} layer')
#         for name2, params in child.named_parameters():
#             params.requires_grad = False
#     ct += 1
model_path = "./Model/resnet50.pt"

def train_model(model, criterion, optimizer, num_epochs = 25):
    # pass
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # training
        model.train() # set the model to training mode
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            print(outputs.shape)
            print(labels.shape)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward() 
            optimizer.step() 

            train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
            train_loss += batch_loss.item()

        # validation
        if len(val_set) > 0:
            model.eval() # set the model to evaluation mode
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels) 
                    _, val_pred = torch.max(outputs, 1) 
                
                    val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                    val_loss += batch_loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, num_epochs, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    # torch.save(model.state_dict(), model_path)
                    torch.save(resnet50, model_path)
                    print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epochs, train_acc/len(train_set), train_loss/len(train_loader)
            ))

    # if not validating, save the last epoch
    if len(val_set) == 0:
        # torch.save(resnet50.state_dict(), model_path)
        torch.save(resnet50, model_path)
        print('saving model at last epoch')

    # torch.save(model, model_path)

NUM_EPOCHS = 200
model_ft = train_model(resnet50, criterion, optimizer,
                        num_epochs = NUM_EPOCHS)