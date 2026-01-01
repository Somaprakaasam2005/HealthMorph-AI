"""
Neural Network Backbone for Phase 3 (v0.5)
Provides pre-trained feature extractors (ResNet50, EfficientNet-B0) for facial analysis.
Supports transfer learning and fine-tuning on medical datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Optional
import numpy as np


class FacialFeatureExtractor:
    """
    Wrapper for pre-trained neural networks for facial feature extraction.
    Supports: ResNet50, EfficientNet-B0, VGG16
    """
    
    def __init__(self, model_name: str = "resnet50", pretrained: bool = True, device: str = "cpu"):
        """
        Args:
            model_name: "resnet50", "efficientnet", or "vgg16"
            pretrained: whether to use ImageNet pre-trained weights
            device: "cpu" or "cuda"
        """
        
        self.model_name = model_name
        self.device = device
        self.pretrained = pretrained
        
        self.model = self._load_model(model_name, pretrained)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Feature dimension depends on model
        self.feature_dim = self._get_feature_dim(model_name)
    
    def _load_model(self, model_name: str, pretrained: bool):
        """Load pre-trained model from torchvision."""
        
        try:
            if model_name == "resnet50":
                model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
                # Remove classification head, keep up to avgpool
                model = nn.Sequential(*list(model.children())[:-1])
            
            elif model_name == "efficientnet":
                model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
                model = nn.Sequential(*list(model.children())[:-1])
            
            elif model_name == "vgg16":
                model = models.vgg16(weights="IMAGENET1K_V1" if pretrained else None)
                model.classifier = nn.Identity()  # Remove classifier
                model = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
            
            else:
                raise ValueError(f"Unsupported model: {model_name}")
            
            return model
        
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            # Fallback to simple CNN
            return self._simple_cnn()
    
    def _simple_cnn(self):
        """Fallback simple CNN if pre-trained model unavailable."""
        
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def _get_feature_dim(self, model_name: str) -> int:
        """Get feature vector dimension."""
        if model_name == "resnet50":
            return 2048
        elif model_name == "efficientnet":
            return 1280
        elif model_name == "vgg16":
            return 512
        else:
            return 512
    
    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vector from image tensor.
        
        Args:
            image_tensor: shape (B, 3, H, W), normalized to [0, 1]
        
        Returns:
            torch.Tensor: shape (B, feature_dim)
        """
        
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            features = self.model(image_tensor)
            features = features.flatten(1)  # Flatten to (B, feature_dim)
        
        return features
    
    def extract_multi_scale_features(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features at multiple scales for richer representation.
        Returns low-level (edges, textures) and high-level (semantic) features.
        """
        
        # Simplified multi-scale: downsample and extract
        scales = {
            "full": image_tensor,
            "half": torch.nn.functional.interpolate(image_tensor, scale_factor=0.5),
            "quarter": torch.nn.functional.interpolate(image_tensor, scale_factor=0.25),
        }
        
        features = {}
        for scale_name, scale_img in scales.items():
            with torch.no_grad():
                feat = self.extract_features(scale_img)
                features[f"{scale_name}_features"] = feat
        
        return features


def image_to_tensor(image_array: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Convert numpy BGR image to torch tensor (normalized to [0, 1]).
    
    Args:
        image_array: numpy BGR array
        target_size: (H, W) for resizing
    
    Returns:
        torch.Tensor: shape (1, 3, H, W)
    """
    
    try:
        import cv2
        
        # Resize
        image = cv2.resize(image_array, (target_size[1], target_size[0]))
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # (H, W, 3) -> (3, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        # (3, H, W) -> (1, 3, H, W)
        image = np.expand_dims(image, 0)
        
        # numpy -> torch
        tensor = torch.from_numpy(image)
        
        return tensor
    
    except Exception as e:
        print(f"Image conversion error: {e}")
        # Return zeros tensor on error
        return torch.zeros(1, 3, target_size[0], target_size[1])


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor back to numpy array."""
    
    if tensor.device.type == "cuda":
        tensor = tensor.cpu()
    
    return tensor.numpy()


def extract_facial_features(image_array: np.ndarray, model_name: str = "resnet50") -> Dict:
    """
    High-level function to extract facial features from image.
    
    Args:
        image_array: numpy BGR image
        model_name: neural backbone to use
    
    Returns:
        dict with:
        - features: 1D feature vector (numpy)
        - feature_dim: dimension of feature vector
        - model_used: which model was used
    """
    
    try:
        # Initialize feature extractor
        extractor = FacialFeatureExtractor(model_name=model_name, device="cpu")
        
        # Convert image to tensor
        image_tensor = image_to_tensor(image_array)
        
        # Extract features
        features_tensor = extractor.extract_features(image_tensor)
        
        # Convert back to numpy
        features_numpy = tensor_to_array(features_tensor[0])
        
        return {
            "features": features_numpy,
            "feature_dim": extractor.feature_dim,
            "model_used": model_name,
            "shape": features_numpy.shape,
        }
    
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return {
            "features": np.zeros(2048),
            "feature_dim": 2048,
            "model_used": "fallback",
            "error": str(e),
        }


def compute_feature_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.
    Useful for comparing facial features across images.
    
    Args:
        features1, features2: 1D numpy arrays
    
    Returns:
        float: similarity score 0-1 (1 = identical, 0 = orthogonal)
    """
    
    try:
        # Normalize
        f1 = features1 / (np.linalg.norm(features1) + 1e-8)
        f2 = features2 / (np.linalg.norm(features2) + 1e-8)
        
        # Cosine similarity
        similarity = float(np.dot(f1, f2))
        
        return float(np.clip(similarity, 0, 1))
    
    except Exception:
        return 0.0
