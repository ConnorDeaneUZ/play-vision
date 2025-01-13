import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


## import detection model
from detection import CNN_LSTM

## load trained model
model = CNN_LSTM(feature_dim=522, hidden_dim=128, num_classes=2)
model.load_state_dict(torch.load("goal_detection_model.pth"))
model.eval()


# Pre-trained ResNet model
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.fc = torch.nn.Identity()  # Remove the classification layer

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features from frames
def extract_frame_features(frame_dir):
    # List all frame files in the directory
    frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])

    features = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = resnet(input_tensor).squeeze(0)  # Extract features
        features.append(feature.numpy())
    
    return torch.from_numpy(np.array(features))

# Example usage
frame_features = extract_frame_features("frames/")
print(frame_features.shape)  # Output the shape of the tensor



# Function to predict if the video contains a goal
def predict_goal(frame_features):
    frame_features = frame_features.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(frame_features)
        _, predicted = torch.max(output, 1)
    return "Goal" if predicted.item() == 1 else "No Goal"



# Step 2: Extract frame features
features = extract_frame_features("frames/")
print(f"Extracted features shape: {features.shape}")

# Step 3: Predict if a goal occurred
prediction = predict_goal(features)
print(f"Prediction: {prediction}")
