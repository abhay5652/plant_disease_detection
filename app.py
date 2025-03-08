import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import Image
import streamlit as st
from transformers import pipeline

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load PlantVillage dataset
data_dir = "./PlantVillage"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load Pre-trained Model (ResNet50)
model = models.resnet50(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, len(dataset.classes))  # Adjust output layer for PlantVillage classes
model = model.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Fine-tune the model
train_model(epochs=5)

# Save the fine-tuned model
MODEL_PATH = "plant_disease_finetuned.pth"
torch.save(model.state_dict(), MODEL_PATH)

# Load LLM for explanations
llm = pipeline("text-generation", model="openai/gpt-3.5-turbo")

def detect_disease(image_path):
    """Predicts plant disease from an image."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return dataset.classes[predicted.item()]

def get_treatment_advice(disease_name):
    """Fetches treatment advice using an LLM."""
    prompt = f"I have detected {disease_name} on my plant. How should I treat it?"
    response = llm(prompt, max_length=100)
    return response[0]['generated_text']

# Streamlit UI for mobile app
st.title("🌿 Plant Disease Detector (Fine-tuned)")
st.write("Upload a plant leaf image to detect diseases and get treatment advice.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")
    
    image_path = "temp_leaf.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    disease = detect_disease(image_path)
    st.write(f"### 🌱 Detected Disease: {disease}")
    
    treatment_advice = get_treatment_advice(disease)
    st.write(f"### 💊 Treatment Advice:")
    st.write(treatment_advice)

st.write("Powered by AI 🤖 | Fine-tuned on PlantVillage Dataset")
