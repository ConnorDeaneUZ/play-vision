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
        
        self.lstm = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Adjust final layer size if bidirectional
        fc_input_dim = config.hidden_dim * 2 if config.bidirectional else config.hidden_dim
        self.fc = nn.Linear(fc_input_dim, config.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, feature_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        lstm_out, _ = self.lstm(x)
        final_hidden_state = lstm_out[:, -1, :]
        return self.fc(final_hidden_state)
    
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
