import torch
import streamlit as st
from PIL import Image
from model import ResnetBlock, Resnet50
from testTransform import testImageTransform

pthName = "best_resnet50_3.pth" #State dictionary giving model highest accuracy

#Model setup
model = Resnet50(ResnetBlock, [3, 4, 6, 3], numClass=2)
model.load_state_dict(torch.load(pthName, map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

classNames = ["Normal", "Pneumonia"] #Manually set class names

def predict(image, model, imageSize=256):
    transform = testImageTransform(imageSize)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    return classNames[torch.argmax(output, dim=1).item()]

#Streamlit UI
st.title("ResNet50 Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict(image, model, imageSize=256)
    st.write("Prediction:", prediction)