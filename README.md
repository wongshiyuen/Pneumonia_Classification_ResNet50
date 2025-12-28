# Pneumonia_Classification_ResNet50
Medical image classification project: Chest X‑rays (Normal vs Pneumonia) using ResNet50.

## Repository Contents
|File                  |Key Features
|:---------------------|:---------------------------------------------------------------------------------
|resnet50_1.py         |ResNet50 model, augmentations (crop, flip, rotation, color jitter), early stopping
|resnet50_2.py         |ResNet50 model, augmentations (crop, flip, affine, color jitter), early stopping
|resnet50_3.py         |Same as v2, plus validation loss as tiebreaker when ≥2 epochs have max accuracy
|resnet50_3smoothing.py|Same as v3, plus label smoothing
|resnet50_4.py         |Same as v3, plus Mixup augmentation with constant alpha values {0.05, 0.1, 0.2}
|resnet50_5.py         |Same as v3, plus Mixup augmentation with linear‑decaying alpha (max=0.2, decay=2)

Note:
* Early stopping is implemented in all versions.
* Validation loss is used as a tiebreaker in versions 3–5.
* Mixup augmentation is introduced in versions 4 and 5 with different alpha strategies.
* Files include code for creating confusion matrices, classification reports, and ROC curves.

## Requirements
* Python ≥ 3.8 (3.11.9 for reproducibility)
* PyTorch ≥ 1.10 (2.5.1+cu121 for reproducibility)
* torchvision ≥ 0.11 (0.20.1+cu121 for reproducibility)
* numpy
* matplotlib
* seaborn
* psutil
* scikit‑learn

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

## Data Attribution
This project uses chest X-ray images (normal and with pneumonia) obtained from Guangzhou Women and Children’s Medical Center, Guangzhou. The whole dataset can be downloaded from Kaggle (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). The dataset is licensed under CC BY 4.0.
