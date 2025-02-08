import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.cnn_lstm import CNN_LSTM
import numpy as np
import math

# Suppress the FutureWarning for torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

class VideoDataset(Dataset):
    def __init__(self, feature_files, labels, max_seq_length=400):
        self.feature_files = feature_files
        self.labels = labels
        self.max_seq_length = max_seq_length
        
        # Calculate mean and std for feature normalization
        self.mean = None
        self.std = None
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute dataset statistics for normalization"""
        features_list = []
        for file in self.feature_files[:100]:  # Use subset for efficiency
            features = torch.load(file)
            features_list.append(features)
        
        features_tensor = torch.cat(features_list, dim=0)
        self.mean = features_tensor.mean(dim=0)
        self.std = features_tensor.std(dim=0)

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        features = torch.load(self.feature_files[idx])
        features = features.clone().detach().to(torch.float32)
        
        # Normalize features
        features = (features - self.mean) / (self.std + 1e-7)
        
        # Create attention mask for padding
        seq_length = features.size(0)
        attention_mask = torch.ones(self.max_seq_length, dtype=torch.bool)
        
        if seq_length > self.max_seq_length:
            features = features[:self.max_seq_length]
            attention_mask = attention_mask
        else:
            # Pad with zeros
            padding = torch.zeros(
                self.max_seq_length - seq_length,
                features.size(1),
                dtype=torch.float32
            )
            features = torch.cat([features, padding], dim=0)
            attention_mask[seq_length:] = 0
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {
            'features': features,
            'attention_mask': attention_mask,
            'label': label
        }

class ModelTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer_class=optim.AdamW,
        learning_rate=0.001,
        batch_size=32,
        num_epochs=30,      # Reduced epochs
        patience=8,         # Reduced patience
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Modified optimizer settings
        self.optimizer = optimizer_class(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.003,    # Slightly reduced weight decay
            betas=(0.9, 0.999)    # Keep default betas
        )
        
        # Use CosineAnnealingLR with longer cycle
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs // 2,  # Half epoch cycle
            eta_min=1e-6
        )
        
        self.grad_clip = 0.5      # Keep moderate gradient clipping
    
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

    def frame_shuffle_augmentation(self, features):
        """
        Randomly shuffle some frames while keeping start/end frames fixed
        
        Args:
            features: Tensor of shape (batch_size, sequence_length, feature_dim)
        """
        batch_size, seq_len, feat_dim = features.shape
        shuffled_features = features.clone()
        
        # Keep first and last 2 frames fixed, shuffle middle frames
        for i in range(batch_size):
            middle_idx = torch.randperm(seq_len-4) + 2
            shuffled_features[i, 2:-2] = features[i, middle_idx]
        
        return shuffled_features

    def train_step(self, batch):
        features = batch['features'].to(self.device)
        labels = batch['label'].to(self.device)
        
        # Always apply some form of augmentation
        features = self.apply_augmentations(features)
        
        # Add dropout to input features
        feature_dropout = nn.Dropout(p=0.1)
        features = feature_dropout(features)
        
        self.optimizer.zero_grad()
        outputs = self.model(features)
        
        # Use a lower label smoothing factor
        smooth_factor = 0.05  # Reduced smoothing factor (was 0.1)
        n_classes = outputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(outputs)
            true_dist.fill_(smooth_factor / (n_classes - 1))
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - smooth_factor)
        
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(outputs, dim=1),
            true_dist,
            reduction='batchmean'
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def apply_augmentations(self, features):
        """Enhanced augmentation pipeline"""
        # Always apply frame shuffling
        features = self.frame_shuffle_augmentation(features)
        
        # Increased probabilities for other augmentations
        if torch.rand(1).item() < 0.5:  # Increased from 0.3
            features = self.random_frame_drop(features)
        
        if torch.rand(1).item() < 0.4:  # Increased from 0.2
            features = self.temporal_crop(features)
        
        # Add Gaussian noise
        noise = torch.randn_like(features) * 0.05
        features = features + noise
        
        return features
    
    def random_frame_drop(self, features):
        """Randomly drop frames"""
        batch_size, seq_len, feat_dim = features.shape
        mask = torch.rand(batch_size, seq_len, 1, device=features.device) > 0.1
        return features * mask
    
    def temporal_crop(self, features):
        """Random temporal cropping"""
        batch_size, seq_len, feat_dim = features.shape
        crop_len = int(seq_len * 0.8)  # Crop to 80% of original length
        start = torch.randint(0, seq_len - crop_len, (1,)).item()
        cropped = features[:, start:start+crop_len, :]
        # Pad back to original length
        padding = torch.zeros(batch_size, seq_len - crop_len, feat_dim, device=features.device)
        return torch.cat([cropped, padding], dim=1)

    def validate(self, val_loader):
        """Validation step"""
        self.model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Normalize features
                features = (features - features.mean()) / (features.std() + 1e-8)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        return val_loss, val_acc

    def train(self, train_loader, val_loader):
        """Train the model"""
        print("\nStarting training...")
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training loop
            for batch in train_loader:
                loss = self.train_step(batch)
                total_train_loss += loss
                
                # Calculate accuracy
                outputs = self.model(batch['features'].to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                labels = batch['label'].to(self.device)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation loop
            val_loss, val_acc = self.validate(val_loader)
            
            # Calculate metrics
            train_loss = total_train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Update learning rate based on validation loss
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print epoch results
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Acc:  {train_acc:>6.2f}% | Val Acc:  {val_acc:>6.2f}%")
            print(f"  Learning Rate: {current_lr:6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pth")
                print("  → New best validation loss!")
            else:
                patience_counter += 1
                print(f"  → No improvement for {patience_counter} epochs")
                if patience_counter >= self.patience:
                    print("\nEarly stopping triggered!")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    print(f"Training stopped after {epoch + 1} epochs")
                    # Load best model
                    self.model.load_state_dict(torch.load("best_model.pth"))
                    break
            
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'lr': current_lr
            })
        
        return training_history

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
    training_history = trainer.train(train_loader, val_loader)
    trainer.save_model("goal_detection_model.pth")

if __name__ == "__main__":
    main()
