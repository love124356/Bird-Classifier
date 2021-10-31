# BirdClassifier

This repository gathers the code for bird image classification from the [in-class CodaLab competition](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07).

Use ResNeXt-101 32x8d with data augmentation and SGD with weight_decay optimizer + 1 cosine annealing learning rate scheduler.

## Reproducing Submission
Our model achieve 78.503% accuracy in testing set.

To reproduce my submission without retrainig, do the following steps:
1. [Requirements](#Requirements)
2. [Inference](#Inference)

## Requirements

All requirements should be detailed in requirements.txt.

```env
virtualenv .
source bin/activate
pip3 install -r requirements.txt
```

Official images can be downloaded from [CodaLab competition](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit)

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

## Repository Structure

Run the following command to build the directory.
```
mkdir model
mkdir data
```

The repository structure is:
```
BirdClassifier
  +- data             # all file used in the program and the prediction(answer.txt)
  +- model            # all trained models
  +- src            
  ∣- dataset.py        # set a dataset class for loading imgs
  ∣- inference.py      # reproduce my submission file
  ∣- train.py          # training model
  - requirements.txt  # txt file for establishing the environment
```

## Training

To train the model, run this command:

```train
python train.py
```

All experiments will be written in [Results](#Results).
You can use the table info to adjust the parameters to get a similar model weights.

Trained model will be save as ```model/model_name.pth```

## Inference

Please download [this model]() if you want to reproduce my submission file, and put it in the 'model' folder.
To reproduce my submission file or test the model you trained, run:

```inference
python inference.py
```

Prediction file will be save as ```data/answer.txt```

## Results

Our model achieves the following performance on :


| Model name          | Top 1 Accuracy  |
| ------------------  |---------------- |
| ResNeXt-101 32x8d   |     78.503%     |
| ResNet152           |     --%         |
