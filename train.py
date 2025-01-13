import torch
import torch.nn as nn
import torch.optim as optim
from detection import CNN_LSTM
from torch.utils.data import DataLoader, Dataset

# Sample Dataset class (replace this with your actual dataset)
class VideoDataset(Dataset):
    def __init__(self, feature_files, labels):
        self.feature_files = feature_files
        self.labels = labels

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features = torch.load(self.feature_files[idx])  # Load pre-extracted features
        label = self.labels[idx]
        return features, label

# Load your dataset
feature_files = ["features/goal_1.pt", "features/no_goal_1.pt"]  # Example files
labels = [1, 0]  # 1 = Goal, 0 = No Goal
dataset = VideoDataset(feature_files, labels)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Create the model
model = CNN_LSTM(feature_dim=512, hidden_dim=128, num_classes=2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Train for 10 epochs
    total_loss = 0
    for features, label in dataloader:
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# ✅ Save the trained model
torch.save(model.state_dict(), "goal_detection_model.pth")
print("Model saved as goal_detection_model.pth")
