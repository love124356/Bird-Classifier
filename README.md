# BirdClassifier

This repository gathers the code for bird image classification from the [in-class CodaLab competition](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07).

Use ResNeXt-101 32x8d with data augmentation and SGD with weight_decay optimizer + cosine annealing learning rate scheduler.

## Reproducing Submission
Our model achieve 78.5031% accuracy in testing set.

To reproduce my submission without retrainig, do the following steps:
1. [Requirements](#Requirements)
2. [Inference](#Inference)

## Hardware

Ubuntu 18.04.5 LTS

Intel® Core™ i7-3770 CPU @ 3.40GHz × 8

GeForce GTX 1080/PCIe/SSE2


## Requirements

All requirements should be detailed in requirements.txt.

```env
$ virtualenv venv --python=3.6
$ source ./venv/bin/activate
$ cd BirdClassifier
$ pip install -r requirements.txt
```

Official images can be downloaded from [CodaLab competition](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit)


## Repository Structure

Run the following command to build the directory.
```
$ mkdir model
$ mkdir data
```

The repository structure is:
```
BirdClassifier
  +- data                     # all file used in the program 
  +- model                    # all trained models
  +- result                   # confusion matrix of two models and answer.txt save here
  +- src            
  ∣- dataset.py               # set a dataset class for loading imgs
  ∣- inference.py             # reproduce my submission file or test your model
  ∣- train.py                 # for training model
  - requirements.txt          # txt file for establishing the environment
```

## Dataset Preparation
After downloading images from [CodaLab competition](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit), we expect the data directory is structured as:
```
data
  +- training_data         # all training data from CodaLab
  ∣- 0003.jpg
  ∣- 0008.jpg
  ∣- 0010.jpg
  ∣- ...................
  +- testing_data          # all testing data from CodaLab
  ∣- 0001.jpg
  ∣- 0002.jpg  
  ∣- 0004.jpg
  ∣- ...................
  - classes.txt            # txt file contain 200 class in the dataset
  - testing_img_order.txt  # txt file about testing img submitting order
  - training_labels.txt    # txt file contain training img's class and label
```


## Training

To train the model, run this command:

```train
python ./src/train.py
```

Notice that the MODEL_PATH is correct or not.

All experiments will be written in [Results](#Results).

You can use the table info to adjust the parameters to get a similar model weights.

Trained model will be saved as ```model/model_name.pth```

## Inference

Please download [this model](https://reurl.cc/Rb2ZD6) if you want to reproduce my submission file, and put it in the 'model' folder.

To reproduce my submission file or test the model you trained, run:

```inference
python ./src/inference.py
```

Notice that the MODEL_PATH is correct or not.

Prediction file will be saved as ```result/answer.txt```

## Results

Our model achieves the following performance on :


| **Model name**   | **Accuracy** | **LR** | **Optimizer**                       | **Batch size**     | **Scheduler**              |**Img size** | **Other**                                                                     |
|-------------------|--------------|--------|-------------------------------------|--------------------|----------------------------|--------------|-------------------------------------------------------------------------------|
| ResNeXt-101 32x8d | 78.5031% | 0.001 | SGD, momentum=0.9,weight_decay=3e-4 | train: 4, test:12  | CosineAnnealing, T_max=200 | 375 x 375    | max_padding, CenterCrop, (HorizontalFlip,or VerticalFlip), unfreeze all layer |
| ResNet152         | 76.2941% | 0.001 | SGD, momentum=0.9,weight_decay=3e-4 | train: 4, test:12  | CosineAnnealing, T_max=200 | 375 x 375    | max_padding, CenterCrop, (HorizontalFlip,or VerticalFlip), unfreeze all layer |
| resnext50_32x4d   | 63.6004% | 0.01 | SGD, momentum=0.9,weight_decay=3e-4 | All: 32            | CosineAnnealing, T_max=200 | 224 x 224    | Horizontal, Rotation, unfreeze last 3 layer                                   |
| Resnet50          | 56.0831% | 0.001 | SGD, momentum=0.9                   | All: 64            | CosineAnnealing, T_max=4   | 224 x 224    | Horizontal, training set *2, unfreeze 'layer4' and  'fc'                      |
