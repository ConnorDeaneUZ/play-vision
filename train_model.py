import os
import torch
import torch.nn as nn
from models.cnn_lstm import CNN_LSTM, ModelConfig
from training.trainer import ModelTrainer

def get_all_feature_files():
    """Get all feature files and their corresponding labels"""
    feature_files = []
    labels = []
    
    # Process goal features
    goal_dir = "features/goal"
    for feature_file in os.listdir(goal_dir):
        if feature_file.endswith(".pt"):
            feature_files.append(os.path.join(goal_dir, feature_file))
            labels.append(1)  # 1 for goal
    
    # Process no-goal features
    no_goal_dir = "features/no_goal"
    for feature_file in os.listdir(no_goal_dir):
        if feature_file.endswith(".pt"):
            feature_files.append(os.path.join(no_goal_dir, feature_file))
            labels.append(0)  # 0 for no goal
    
    return feature_files, labels

def main():
    feature_files, labels = get_all_feature_files()
    
    model_config = ModelConfig(
        feature_dim=512,
        hidden_dim=160,     # Back to proven hidden dim
        num_classes=2,
        num_layers=1,
        dropout=0.2,        # Keep moderate dropout
        bidirectional=True  # Keep bidirectional
    )
    
    model = CNN_LSTM(model_config)
    
    trainer = ModelTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        learning_rate=0.001,
        batch_size=32,
        num_epochs=30,      # Back to 30 epochs
        patience=8          # Standard patience
    )
    
    # Prepare data and get class weights
    train_loader, val_loader, class_weights = trainer.prepare_data(feature_files, labels, val_split=0.3)  # Increased validation split
    
    # Update criterion with class weights
    trainer.criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train the model
    trainer.train(train_loader, val_loader)
    trainer.save_model("goal_detection_model.pth")

if __name__ == "__main__":
    main() 
