import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os
import numpy as np
import psutil
import seaborn as sns
import matplotlib.pyplot as plt
import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

#Custom packages
from model import ResnetBlock, Resnet50
from testTransform import testTransform

#Speeds up training when input image sizes are constant, model architecture doesn't change shape dynamically, and batch size is consistent
torch.backends.cudnn.benchmark = True
#=====================================================================================================
#Report memory and number of parameters
def memoryAndParam(model):
    #Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    else:
        process = psutil.Process()
        print(f"CPU memory usage: {process.memory_info().rss / 1024**2:.2f} MB")

    # Parameter counts
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {params:,} total params | {trainable:,} trainable")
    
    return device
#=====================================================================================================
#DEFINE HYPERPARAMETERS
batchSize = 16
learnRate = 0.0001
epochs = 100
imageSize = 256 #Resize images to stated size
beta1 = 0.9 #Default for classificaltion
beta2 = 0.999 #Default for classificaltion
randomSeed = 42

torch.manual_seed(randomSeed)
np.random.seed(randomSeed)
random.seed(randomSeed)

pthName = "best_resnet50_3.pth"
#=====================================================================================================
#DEFINE TRANSFORM FOR TRAINING DATASET IMAGE RESIZING
#Transformation for traning dataset
trainTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), #Produce single-channel grayscale
    transforms.RandomResizedCrop(imageSize, scale=(0.8, 1.0)), #Randomly crop to between 80%-100%, then resize
    transforms.RandomHorizontalFlip(), #Randomly flip an image (or not)
    transforms.RandomAffine(degrees=10, translate=(0.03, 0.03), scale=(0.9, 1.1), shear=3,fill=128),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), #Alter brightness & contrast by between -10% to 10%
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
#=====================================================================================================
#DATASET READING AND LOADING
#Read images in a folder-per-class structure, assigns each class name an integer index.
#Returns a tuple for each image: (image_filepath, label(subfolder name)).
trainDataset = datasets.ImageFolder(root=r'train', transform=trainTransform)
valDataset = datasets.ImageFolder(root=r'val', transform=testTransform)
testDataset = datasets.ImageFolder(root=r'test', transform=testTransform)

#Provide iterator for fetching data in batches; handle batching, shuffling, & parallel loading.
#Returns batches in the form of tuples (images, labels)
trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
valLoader = DataLoader(valDataset, batch_size=batchSize)
testLoader = DataLoader(testDataset, batch_size=batchSize)
#=====================================================================================================
#CREATE MODEL, CLASS WEIGHTS TO HANDLE CLASS IMBALANCES, OPTIMIZER, LEARNING RATE SCHEDULER
model = Resnet50(ResnetBlock, [3, 4, 6, 3], numClass=2)
device = memoryAndParam(model)

#Calculate count class frequencies
classCounts = [0, 0] #For two classes
for _, label in trainDataset:
    classCounts[label] += 1 #NOTE: label has value of 0 or 1

total = sum(classCounts)

weights = [total/classCounts[i] for i in range(2)] #Weight for each class
weights = torch.tensor(weights, dtype=torch.float).to(device) #Convert to tensor, move to device

#Loss function via cross-entropy loss (BCEWithLogitsLoss can be considered for binary)
criterion = nn.CrossEntropyLoss(weight=weights) #When computing loss, multiply loss of class by its weight

optimizer = torch.optim.Adam(model.parameters(), lr=learnRate, betas=(beta1, beta2))

#Step lr scheduler, drop learning rate by a fixed factor every N epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, #Optimizer being adjusted
    mode='min', #Minimize (metric being monitored)
    factor=0.1, #Learning rate is multiplied by this factor when triggered
    patience=5, #Number of epochs with no improvement before reducing LR
    verbose=True
)
#=====================================================================================================
startTime = time.time()

best_val_loss = float('inf')
patience = 20
counter = 0
best_acc = 0

for epoch in range(epochs):
    #Training
    model.train() #Tell layers like Dropout and BatchNorm to behave in ways that help model learn.
    running_loss = 0.0
    
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() #Zero out old gradients
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    #Validation
    #.eval() tells layers like Dropout and BatchNorm to switch to a stable, deterministic behavior for testing
    #Does NOT affect gradient tracking
    model.eval()
    correct, total, val_loss = 0, 0, 0
    #The with statement makes gradient tracking enabling and disabling safe and automatic
    #Better than creating 2 lines of code to switch gradient tracking on and off
    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    val_acc = 100*correct/total
    
    #Print progress
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainLoader):.4f}, Val Loss: {val_loss/len(valLoader):.4f}, Val Acc: {val_acc:.2f}%")

    #Save best model based on highest accuracy; use val. loss as tiebreaker
    if val_acc > best_acc or (val_acc == best_acc and val_loss <= best_val_loss):
        best_acc = val_acc
        torch.save(model.state_dict(), pthName)
    
    #Stoppage when lowest validation loss remains unchanged for a pre-determined no. of epochs
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
    
    scheduler.step(val_loss) #Step lr scheduler, drop learning rate by a fixed factor every N epochs
    
    #NOTE: Best model accuracy & loss might not belong to the model from the same epoch.
    #NOTE: Both parameters are only there to serve their own respective purposes in the 'for' loop.
#-----------------------------------------------------------------------------------------------------
#Evaluate on test set
#.pth file contains keys (layer names) and values (weights and biases)
model.load_state_dict(torch.load(pthName)) #Load optimum weights and biases into model

correct, total = 0, 0
all_preds, all_labels, probabilities = [], [], []
model.eval()
with torch.no_grad(): #No gradient tracking, no computation graph, no intermediate activation storage
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        #For ROC curve
        probs = torch.softmax(outputs, dim=1)[:, 1] #Probability of class 1
        probabilities.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
probabilities = np.array(probabilities)

test_acc = 100*correct/total
print(f"Test Accuracy: {test_acc:.2f}%")

endTime = time.time()
#-----------------------------------------------------------------------------------------------------
#Results and reports
print(f"Runtime: {endTime-startTime:.2f} seconds")

classNames = ["Normal", "Pneumonia"] #Manually set class names

#Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classNames, yticklabels=classNames)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

#Classification report
print(classification_report(all_labels, all_preds, target_names=classNames))

#ROC Curve + AUC
fpr, tpr, thresholds = roc_curve(all_labels, probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

correct_idx = [i for i in range(len(all_labels)) if all_labels[i] == all_preds[i]] #Correctly classified indices
incorrect_idx = [i for i in range(len(all_labels)) if all_labels[i] != all_preds[i]] #Incorrectly classified indices

#Pick random samples
sampleCorrect = random.sample(correct_idx, min(5, len(correct_idx))) #Correctly classified samples
sampleIncorrect = random.sample(incorrect_idx, min(5, len(incorrect_idx))) #Incorrectly classified samples

def showSample(idxList, title): #Function for dislaying images
    plt.figure(figsize=(12,4))
    for i, index in enumerate(idxList):
        img, label = testDataset[index]
        plt.subplot(1, len(idxList), i+1)
        plt.imshow(img.squeeze(), cmap="gray") #Change to 3D array to 2D array (preferred by matplotlib for grayscale)
        plt.title(f"True: {classNames[label]}\nPred: {classNames[all_preds[index]]}")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()

showSample(sampleCorrect, "Correct Predictions")
showSample(sampleIncorrect, "Incorrect Predictions")
