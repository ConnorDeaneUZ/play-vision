import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.cnn_lstm import CNN_LSTM
import numpy as np

# Suppress the FutureWarning for torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

class VideoDataset(Dataset):
    def __init__(self, feature_files, labels, max_seq_length=400):
        self.feature_files = feature_files
        self.labels = labels
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features = torch.load(self.feature_files[idx])
        features = features.clone().detach().to(torch.float32)
        
        # Pad or truncate sequence to max_seq_length
        seq_length = features.size(0)
        if seq_length > self.max_seq_length:
            # Truncate
            features = features[:self.max_seq_length]
        elif seq_length < self.max_seq_length:
            # Pad with zeros
            padding = torch.zeros(self.max_seq_length - seq_length, features.size(1), dtype=torch.float32)
            features = torch.cat([features, padding], dim=0)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
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
        patience=3,  # Early stopping patience
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=2, factor=0.5
        )
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

    def prepare_data(self, feature_files, labels, val_split=0.2):
        # Convert to numpy arrays for easier manipulation
        feature_files = np.array(feature_files)
        labels = np.array(labels)
        
        # Split data by class
        goal_indices = np.where(labels == 1)[0]
        no_goal_indices = np.where(labels == 0)[0]
        
        # Shuffle indices within each class
        np.random.shuffle(goal_indices)
        np.random.shuffle(no_goal_indices)
        
        # Calculate split sizes for each class
        n_val_goal = int(np.ceil(len(goal_indices) * val_split))
        n_val_no_goal = int(np.ceil(len(no_goal_indices) * val_split))
        
        # Split each class into train/val
        val_indices = np.concatenate([
            goal_indices[:n_val_goal],
            no_goal_indices[:n_val_no_goal]
        ])
        train_indices = np.concatenate([
            goal_indices[n_val_goal:],
            no_goal_indices[n_val_no_goal:]
        ])
        
        # Create train/val splits
        train_features = feature_files[train_indices].tolist()
        train_labels = labels[train_indices].tolist()
        val_features = feature_files[val_indices].tolist()
        val_labels = labels[val_indices].tolist()
        
        # Calculate class weights for balanced training
        n_samples = len(labels)
        n_goals = np.sum(labels == 1)
        class_weights = torch.FloatTensor([n_samples / (2 * (n_samples - n_goals)), 
                                         n_samples / (2 * n_goals)]).to(self.device)
        
        # Print split information
        print(f"\nDataset split info:")
        print(f"Total samples: {len(labels)}")
        print(f"Training samples: {len(train_features)}")
        print(f"Validation samples: {len(val_features)}")
        print(f"Training goal/no-goal ratio: {sum(train_labels)}/{len(train_labels)-sum(train_labels)}")
        print(f"Validation goal/no-goal ratio: {sum(val_labels)}/{len(val_labels)-sum(val_labels)}")
        print(f"Class weights: {class_weights.cpu().numpy()}\n")
        
        train_dataset = VideoDataset(train_features, train_labels)
        val_dataset = VideoDataset(val_features, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader, class_weights

    def train_step(self, features, labels):
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
        
        return total_loss / len(dataloader), (correct_predictions / total_predictions) * 100

    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            for features, labels in train_loader:
                loss = self.train_step(features, labels)
                train_loss += loss
                
                with torch.no_grad():
                    outputs = self.model(features.to(self.device))
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == labels.to(self.device)).sum().item()
                    total_predictions += labels.size(0)
            
            train_loss = train_loss / len(train_loader)
            train_accuracy = (correct_predictions / total_predictions) * 100
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Accuracy: {train_accuracy:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered")
                    break

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved as {path}")

def main():
    # Get all feature files and labels
    feature_files, labels = get_all_feature_files()
    
    # Initialize trainer with default loss first
    model = CNN_LSTM(ModelConfig(
        feature_dim=512,
        hidden_dim=256,
        num_classes=2,
        num_layers=2,
        dropout=0.2
    ))
    
    trainer = ModelTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),  # Start with unweighted loss
        learning_rate=0.001,
        batch_size=4,
        num_epochs=20,
        patience=5
    )
    
    # Prepare data and get class weights
    train_loader, val_loader, class_weights = trainer.prepare_data(feature_files, labels)
    
    # Update the criterion with class weights
    trainer.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Train the model
    trainer.train(train_loader, val_loader)
    trainer.save_model("goal_detection_model.pth")

if __name__ == "__main__":
    main()
