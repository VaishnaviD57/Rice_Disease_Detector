import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import gdown
import os

st.set_page_config(page_title="Rice Disease Detection", layout="wide")

st.title("🌾 Real-Time Rice Disease Detection")
st.write("Upload a rice leaf image")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ DOWNLOAD MODEL FROM GOOGLE DRIVE ------------------

MODEL_PATH = "rice_model.pth"
FILE_ID = "1ns2pDwMhfhIkym_kmxHQX8sLqQlCdHtX"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... Please wait ⏳"):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# ------------------ MODEL ARCHITECTURE ------------------

class RiceCNN(nn.Module):
    def __init__(self, num_classes=12):
        super(RiceCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 12)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------ LOAD MODEL ------------------

model = RiceCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ------------------ CLASS NAMES ------------------

classes = [
    "green_leafhopper",
    "planthopper",
    "rice_bacterialblight_Disease",
    "rice_blast_disease",
    "rice_brownspot_disease",
    "rice_bug",
    "rice_falsesmut_Disease",
    "rice_healthy_leaf",
    "rice_leaf_roller",
    "rice_leafscald",
    "rice_sheathblight",
    "rice_stemborer"
]

# ------------------ IMAGE TRANSFORM ------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Rice Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    disease = classes[pred.item()]
    confidence = confidence.item() * 100

    st.subheader("Prediction Result")

    if disease == "rice_healthy_leaf":
        st.success(f"Healthy Rice Leaf ({confidence:.2f}%)")
    else:
        st.error(f"Diseased: {disease}")
        st.warning(f"Confidence: {confidence:.2f}%")

    st.write("All Class Probabilities")
    for i, cls in enumerate(classes):
        st.write(f"{cls}: {probs[0][i].item() * 100:.2f}%")
