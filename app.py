import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define model architecture (must match training architecture)
class FruitCNN(nn.Module):
    def __init__(self, num_classes=33):
        super(FruitCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.flatten_size = 64 * 29 * 29
        self.fc1 = nn.Linear(self.flatten_size, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, self.flatten_size)
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# List of fruit categories
CATEGORIES = [
    "Apple Braeburn", "Apple Granny Smith", "Apricot", "Avocado", "Banana", "Blueberry",
    "Cactus fruit", "Cantaloupe", "Cherry", "Clementine", "Corn", "Cucumber Ripe",
    "Grape Blue", "Kiwi", "Lemon", "Limes", "Mango", "Onion White", "Orange", "Papaya",
    "Passion Fruit", "Peach", "Pear", "Pepper Green", "Pepper Red", "Pineapple",
    "Plum", "Pomegranate", "Potato Red", "Raspberry", "Strawberry", "Tomato", "Watermelon"
]

# Define transform (same as during training)
transform = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load the trained model
@st.cache_resource
def load_model():
    model = FruitCNN(num_classes=len(CATEGORIES))
    model.load_state_dict(torch.load("fruit_cnn_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit interface
st.title("🍓 ClassiFruit 🍓")
st.write("Upload an image of a fruit to receive the top-3 predicted fruit types with confidence scores.")

# Sidebar with category list
with st.sidebar:
    st.markdown("### 🍇 Supported Fruit Categories:")
    for cat in CATEGORIES:
        st.markdown(f"- {cat}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        _, pred_idx = torch.max(output, 1)
        predicted_class = CATEGORIES[pred_idx.item()]

    # Get top 3 predictions
    probs = torch.softmax(output, dim=1).squeeze()
    top3_probs, top3_idxs = torch.topk(probs, 3)

    st.markdown("---")
    st.markdown("### 🧠 Top 3 Predicted Fruits:")
    for i in range(3):
        st.markdown(f"{i+1}. {CATEGORIES[top3_idxs[i].item()]} — **{top3_probs[i].item()*100:.2f}%** confidence")

