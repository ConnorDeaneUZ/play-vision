import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from detection import CNN_LSTM

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

class ModelTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer_class=optim.Adam,
        learning_rate=0.001,
        batch_size=2,
        num_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def prepare_data(self, feature_files, labels):
        dataset = VideoDataset(feature_files, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_step(self, features, labels):
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, dataloader):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for features, labels in dataloader:
                loss = self.train_step(features, labels)
                total_loss += loss

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as {path}")

def main():
    # Configuration
    feature_files = ["features/goal_1.pt", "features/no_goal_1.pt"]
    labels = [1, 0]
    model_config = {
        'feature_dim': 512,
        'hidden_dim': 128,
        'num_classes': 2
    }

    # Initialize model and trainer
    model = CNN_LSTM(**model_config)
    trainer = ModelTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        num_epochs=10
    )

    # Prepare data and train
    dataloader = trainer.prepare_data(feature_files, labels)
    trainer.train(dataloader)
    trainer.save_model("goal_detection_model.pth")

if __name__ == "__main__":
    main()
