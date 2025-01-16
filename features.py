import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self, model_name='resnet18', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = self._setup_model(model_name)
        self.transform = self._setup_transforms()
    
    def _setup_model(self, model_name):
        # Initialize and configure the model
        if model_name == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            model.fc = torch.nn.Identity()  # Remove the classification layer
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        return model.to(self.device).eval()
    
    def _setup_transforms(self):
        # Define image transformations
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def extract_single_image(self, image_path):
        """Extract features from a single image."""
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(input_tensor).squeeze(0)
        
        return features.cpu()
    
    def extract_from_directory(self, frame_dir, file_extension=".jpg", save_path=None, save_dir=None):
        """Extract features from all images in a directory.
        
        Args:
            frame_dir (str): Directory containing the frames
            file_extension (str): File extension to filter images
            save_path (str, optional): Path to save the features. If None, features are only returned
        """
        frame_paths = sorted([
            os.path.join(frame_dir, f) 
            for f in os.listdir(frame_dir) 
            if f.endswith(file_extension)
        ])
        
        features = []
        for frame_path in frame_paths:
            feature = self.extract_single_image(frame_path)
            features.append(feature.numpy())
        
        features_tensor = torch.from_numpy(np.array(features))
        
        if save_path:
            # Create directory first
            dir_name = os.path.dirname(save_path)
            os.makedirs(dir_name, exist_ok=True)
            # Then save the features
            torch.save(features_tensor, save_path)
            print(f"Features saved to {save_path}")
        
        return features_tensor



