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
* Python ≥ 3.8
* PyTorch ≥ 1.10
* torchvision ≥ 0.11
* numpy
* matplotlib
* seaborn
* psutil
* scikit‑learn

## Results
|File                  |Memory |Test Data Accuracy
|:---------------------|:------|--------:
|resnet50_1.py         |ResNet50 model, augmentations (crop, flip, rotation, color jitter), early stopping
|resnet50_2.py         |ResNet50 model, augmentations (crop, flip, affine, color jitter), early stopping
|resnet50_3.py         |Same as v2, plus validation loss as tiebreaker when ≥2 epochs have max accuracy
|resnet50_3smoothing.py|Same as v3, plus label smoothing
|resnet50_4.py         |Same as v3, plus Mixup augmentation with constant alpha values {0.05, 0.1, 0.2}
|resnet50_5.py         |Same as v3, plus Mixup augmentation with linear‑decaying alpha (max=0.2, decay=2)
