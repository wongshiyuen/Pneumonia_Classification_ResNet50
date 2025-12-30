# Pneumonia_Classification_ResNet50
Medical image classification project: Chest X‑rays (Normal vs Pneumonia) using ResNet50.

## Repository Contents
|Folder     |File                  |Key Features
|:----------|:---------------------|:---------------------------------------------------------------------------------
|experiments|resnet50_1.py         |ResNet50 model, augmentations (crop, flip, rotation, color jitter), early stopping
|experiments|resnet50_2.py         |ResNet50 model, augmentations (crop, flip, affine, color jitter), early stopping
|experiments|resnet50_3.py         |Same as v2, plus validation loss as tiebreaker when ≥2 epochs have max accuracy
|experiments|resnet50_3smoothing.py|Same as v3, plus label smoothing
|experiments|resnet50_4.py         |Same as v3, plus Mixup augmentation with constant alpha values {0.05, 0.1, 0.2}
|experiments|resnet50_5.py         |Same as v3, plus Mixup augmentation with linear‑decaying alpha (max=0.2, decay=2)
|deployment |chosenModel.py        |Chosen model (resnet50_3.py) with highest accuracy out of all tested models
|deployment |testTransform.py      |Image transforms for images in testing dataset for deployment purposes

**Notes:**
* Early stopping is implemented in all versions.
* Validation loss is used as a tiebreaker in versions 3–5.
* Mixup augmentation is introduced in versions 4 and 5 with different alpha strategies.
* Files include code for creating confusion matrices, classification reports, and ROC curves.

## Requirements
* Python ≥ 3.8 (tested with version 3.11.9)
* PyTorch ≥ 1.10 (tested with version 2.5.1+cu121)
* torchvision ≥ 0.11 (tested with version 0.20.1+cu121)
* numpy (tested with version 2.1.3)
* matplotlib (tested with version 3.10.3)
* seaborn (tested with version 0.13.2)
* psutil (tested with version 7.1.3)
* scikit‑learn (tested with version 1.7.1)

## Getting Started
The following are the procedures for downloading and training the models in this repository using the Chest X-ray Images (Pneumonia) dataset:

1. Download the Chest X-ray Images (Pneumonia) dataset from Kaggle (or any other available repositories). The dataset contains 3 main folders ('train', 'test', and 'val'), with 2 subfolders ('normal' and 'pneumonia') in each folder.
2. Download the '.py' file of interest (or all of them) from the 'experiments' section, as well as 'requirements.txt', from this repository. Ensure the file(s) are downloaded into the same folder containing the 'train', 'test', and 'val' subfolders.
3. Open cmd/terminal and change directory to dataset folder.
4. Install required packages by typing 'pip install -r requirements.txt' into cmd.
5. Run the '.py' file of interest by typing 'python <insert '.py' filename here>' into cmd.

## Results
|File                                  |Test Data Accuracy (%)
|:-------------------------------------|-----------------------:
|resnet50_1.py                         |87.98
|resnet50_2.py                         |89.74
|resnet50_3.py                         |**91.35**
|resnet50_3smoothing.py                |90.38
|resnet50_4.py (alpha=0.05)            |91.03
|resnet50_4.py (alpha=0.10)            |89.26
|resnet50_4.py (alpha=0.20)            |89.74
|resnet50_5.py (max_alpha=0.2, decay=2)|90.06

NOTE: In order to reproduce results here for resnet50_4.py, please alter the value of alpha as necessary.

## Data Attribution
This project uses chest X-ray images (normal and with pneumonia) obtained from Guangzhou Women and Children’s Medical Center, Guangzhou. Dataset available on Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). Licensed under **CC BY 4.0**.
