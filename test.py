import torch
from data_preprocessing_tmp import BirdDataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {device}')

DATA_ROOT = r"./Data/"
BATCH_SIZE = 64

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = BirdDataset(DATA_ROOT, "test", transform = test_transform)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = True)

model = torch.load("./Model/resnet50.pt")
model.eval()

predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)


# import os
# import numpy as np

# test_images = os.listdir('testing_images/')  # all the testing images

# submission = []
# for img in test_images:  # image order is important to your result
#     predicted_class = your_model(img)  # the predicted category
#     submission.append([img, predicted_class])

# np.savetxt('answer.txt', submission, fmt='%s')