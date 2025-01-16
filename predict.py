import torch
from typing import Union, List

class GoalPredictor:
    def __init__(self, model: torch.nn.Module):
        """
        Initialize the GoalPredictor with a trained PyTorch model
        
        Args:
            model: A trained PyTorch model for goal detection
        """
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        
    def predict(self, frame_features: torch.Tensor) -> str:
        """
        Predict whether a frame contains a goal
        
        Args:
            frame_features: Tensor containing frame features
            
        Returns:
            str: "Goal" or "No Goal" prediction
        """
        if frame_features.dim() == 1:
            frame_features = frame_features.unsqueeze(0)  # Add batch dimension
            
        with torch.no_grad():
            output = self.model(frame_features)
            _, predicted = torch.max(output, 1)
            
        return "Goal" if predicted.item() == 1 else "No Goal"
    
    def predict_batch(self, batch_features: torch.Tensor) -> List[str]:
        """
        Predict goals for a batch of frames
        
        Args:
            batch_features: Tensor containing multiple frame features
            
        Returns:
            List[str]: List of predictions for each frame
        """
        with torch.no_grad():
            outputs = self.model(batch_features)
            _, predicted = torch.max(outputs, 1)
            
        return ["Goal" if pred == 1 else "No Goal" for pred in predicted]
