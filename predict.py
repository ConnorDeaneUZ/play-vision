import torch

# Function to predict if the video contains a goal

def predict_goal(frame_features):
    frame_features = frame_features.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(frame_features)
        _, predicted = torch.max(output, 1)
    return "Goal" if predicted.item() == 1 else "No Goal"
