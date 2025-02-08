from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration class for CNN_LSTM model parameters."""
    feature_dim: int
    hidden_dim: int
    num_classes: int
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False


class CNN_LSTM(nn.Module):
    """Neural network combining CNN and LSTM for sequence classification.
    
    Attributes:
        config (ModelConfig): Configuration parameters for the model
        lstm (nn.LSTM): LSTM layer for sequential processing
        fc (nn.Linear): Fully connected layer for classification
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input normalization
        self.layer_norm = nn.LayerNorm(config.feature_dim)
        
        # Feature reduction with residual
        self.feature_reduction = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Classifier with skip connection
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    
    def forward(self, x):
        # Input normalization
        x = self.layer_norm(x)
        
        # Feature reduction
        x = self.feature_reduction(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling
        out = torch.mean(lstm_out, dim=1)
        
        # Classification
        out = self.classifier(out)
        return out
    
    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage:
if __name__ == "__main__":
    config = ModelConfig(
        feature_dim=64,
        hidden_dim=128,
        num_classes=10
    )
    model = CNN_LSTM(config)
