#Transformation for testing dataset
#Transforms extracted for deployment and reusability

from torchvision import transforms 

imageSize = 256 #Resize images to stated size

testTransform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), #Change to grayscale
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])