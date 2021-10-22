import torch
import numpy as np
from data_preprocessing_tmp import BirdDataset
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {device}')

DATA_ROOT = r"./Data/"
BATCH_SIZE = 64


with open(DATA_ROOT + 'testing_img_order.txt') as f:
    test_images = [x.strip().split(' ')[0] for x in f.readlines()]  # all the testing images
    # print(test_images)
with open(DATA_ROOT + 'classes.txt') as f:
    classes = [x.strip().split(' ')[0] for x in f.readlines()]  # all the testing images
    # print(classes)


test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_set = BirdDataset(DATA_ROOT, "test", transform = test_transform)
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE, shuffle = False)


test_model = torch.load("./Model/resnet50_lessTRANS.pt", map_location = torch.device("cpu"))
test_model.eval()


predict = []
test_model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        # print(test_set[i])
        inputs = data
        inputs = inputs.to(device)
        outputs = test_model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        for y in test_pred.cpu().numpy():
            # print(y)
            predict.append(y)
        # break


submission = []
for i, y in enumerate(predict):
    # print('{},{}\n'.format(test_images[i], classes[y]))
    submission.append([test_images[i], classes[y]])
print(submission)


np.savetxt('answer.txt', submission, fmt='%s')