#Chosen model with highest test accuracy
#ResNet 50 classes extracted and placed in a separate script for deployment and reusability
import torch
import torch.nn as nn
#=====================================================================================================
class ResnetBlock(nn.Module):
    expansion = 4  #Bottleneck expansion factor
    #in_ch = out_ch*expansion unless changing stages, in which case downsampling will be needed.
    #Default stride = 1 and downsample = None
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.layers = nn.Sequential(
            #Reduce dimensionality (reduce number of channels)
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            #Process spatial features
            nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            #Expand back up
            nn.Conv2d(out_ch, out_ch*self.expansion, 1, bias=False),
            nn.BatchNorm2d(out_ch*self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample #Downsampling blocks inside here (if any)

    def forward(self, inputTensor):
        out = self.layers(inputTensor)
        
        if self.downsample is not None:
            identity = self.downsample(inputTensor)
        else:
            identity = inputTensor
            
        out += identity
        return self.relu(out)
#-----------------------------------------------------------------------------------------------------
class Resnet50(nn.Module):
    def __init__(self, block, layers, numClass=2):
        super().__init__()
        self.in_ch = 64
        #Stem
        #1 input channel for grayscale
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) #Spatial downsampling here.
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #Spatial downsampling here.

        #Use make_layer defined below
        #No spatial downsampling in 1st layer.
        #Network still needs fine‑grained features (edges, textures).
        #If halved again immediately, too much spatial detail will be lost too early.
        self.layer1 = self.makeLayer(block, 64, layers[0]) #output shape=(B, 256, H1, W1)
        self.layer2 = self.makeLayer(block, 128, layers[1], stride=2) #Spatial downsampling here, output shape=(B, 512, H1/2, W1/2)
        self.layer3 = self.makeLayer(block, 256, layers[2], stride=2) #Spatial downsampling here, output shape=(B, 1024, H1/4, W1/4) 
        self.layer4 = self.makeLayer(block, 512, layers[3], stride=2) #Spatial downsampling here.

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #Outputs 1x1 spatial map regardless of input map size
        self.fc = nn.Linear(512*block.expansion, numClass) #Fully connected layer
    #-------------------------------------------------------------------------------------------------
    #make_layer(Block type, no. of input channels, no. of output channels, no. of blocks per stage, stride=1 as default)
    def makeLayer(self, block, out_ch, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_ch != out_ch*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch * block.expansion),
            )

        layers = [block(self.in_ch, out_ch, stride, downsample)]
        
        self.in_ch = out_ch*block.expansion #Update value of self.in_ch for next layer
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_ch, out_ch))
        return nn.Sequential(*layers)
    #-------------------------------------------------------------------------------------------------
    def forward(self, tensor):
        tensor = self.conv1(tensor)
        tensor = self.bn1(tensor)
        tensor = self.relu1(tensor)
        tensor = self.maxpool(tensor)

        tensor = self.layer1(tensor)
        tensor = self.layer2(tensor)
        tensor = self.layer3(tensor)
        tensor = self.layer4(tensor)
        
        tensor = self.avgpool(tensor)
        tensor = torch.flatten(tensor, 1)
        tensor = self.fc(tensor)
        return tensor
#=====================================================================================================