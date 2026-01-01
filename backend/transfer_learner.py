"""
Transfer Learning Module for Phase 3 (v0.5)
Fine-tunes pre-trained models on medical/health datasets.
Includes layer freezing strategies, augmentation, and optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class TrainingConfig:
    """Configuration for transfer learning."""
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.3
    freeze_backbone: bool = True  # Freeze pre-trained weights initially
    unfreeze_at_epoch: Optional[int] = None  # Epoch to unfreeze backbone
    use_focal_loss: bool = True  # For imbalanced medical data
    augmentation: bool = True
    save_best_only: bool = True


class MedicalImageDataset(Dataset):
    """
    Dataset for medical images (facial, X-ray, etc.)
    
    Expected structure:
    - images: list of numpy arrays (H, W, 3)
    - labels: list of int labels
    """
    
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        augmentation: bool = True
    ):
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32)
        label = self.labels[idx]
        
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # CHW format
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        # Data augmentation
        if self.augmentation:
            image = self._augment(image)
        
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)
    
    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Apply simple augmentations."""
        
        # Random flip horizontal
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2)
        
        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness_factor, 0, 1)
        
        # Random rotation (small)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15)
            # Simple rotation approximation (would use cv2.rotate in production)
            pass
        
        return image


class FocalLoss(nn.Module):
    """
    Focal Loss for handling imbalanced medical classification.
    Helpful when certain classes (e.g., rare syndromes) have fewer examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        
        # Softmax probabilities
        p = torch.softmax(logits, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal loss = -alpha * (1 - p_t)^gamma * CE(p, y)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()


class FineTuner:
    """
    Fine-tunes pre-trained neural networks on custom medical datasets.
    Supports layer freezing and progressive unfreezing strategies.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        config: TrainingConfig,
        device: str = "cpu"
    ):
        """
        Args:
            backbone: pre-trained model (e.g., ResNet50)
            num_classes: number of output classes
            config: TrainingConfig object
            device: "cpu" or "cuda"
        """
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.config = config
        self.device = device
        
        # Add classification head
        self.model = self._build_model(backbone, num_classes)
        self.model = self.model.to(device)
        
        # Loss function
        if config.use_focal_loss:
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer (configured after freezing)
        self.optimizer = None
        self.scheduler = None
        
        self.training_history = []
    
    def _build_model(self, backbone: nn.Module, num_classes: int) -> nn.Module:
        """Build model with classification head on top of backbone."""
        
        # Assume backbone outputs feature vector
        # Wrap with classification head
        
        class ClassificationHead(nn.Module):
            def __init__(self, input_size: int, num_classes: int, dropout: float):
                super().__init__()
                self.backbone = backbone
                self.fc = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, num_classes),
                )
            
            def forward(self, x):
                x = self.backbone(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x
        
        # Infer input size from backbone
        input_size = 2048 if hasattr(self.backbone, '__name__') and 'resnet50' in str(self.backbone) else 1280
        
        return ClassificationHead(input_size, num_classes, self.config.dropout_rate)
    
    def freeze_backbone(self):
        """Freeze all backbone weights."""
        
        for param in self.model.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone weights for fine-tuning."""
        
        for param in self.model.backbone.parameters():
            param.requires_grad = True
    
    def prepare_optimizer(self):
        """Create optimizer after setting up freezing."""
        
        if self.config.freeze_backbone:
            # Only train head
            trainable_params = self.model.fc.parameters()
        else:
            # Train everything
            trainable_params = self.model.parameters()
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with progressive unfreezing.
        
        Returns:
            Training history
        """
        
        # Initial setup: freeze backbone, train head
        if self.config.freeze_backbone:
            self.freeze_backbone()
        
        self.prepare_optimizer()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(self.config.num_epochs):
            
            # Progressive unfreezing
            if (
                self.config.unfreeze_at_epoch
                and epoch >= self.config.unfreeze_at_epoch
            ):
                self.unfreeze_backbone()
                # Reduce learning rate for fine-tuning
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * 0.1
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader) if val_loader else train_metrics
            
            self.scheduler.step()
            
            # Save best model
            if self.config.save_best_only:
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            
            # Logging
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_metrics['loss'],
                "train_acc": train_metrics['accuracy'],
                "val_loss": val_metrics['loss'],
                "val_acc": val_metrics['accuracy'],
            })
            
            if verbose and (epoch % 10 == 0 or epoch == self.config.num_epochs - 1):
                print(
                    f"Epoch {epoch+1}/{self.config.num_epochs} | "
                    f"Train: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.2f}% | "
                    f"Val: loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.2f}%"
                )
        
        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return {
            "history": self.training_history,
            "best_val_loss": best_val_loss,
        }
    
    def save_model(self, filepath: str):
        """Save fine-tuned model."""
        
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config.__dict__,
            'num_classes': self.num_classes,
            'history': self.training_history,
            'timestamp': datetime.now().isoformat(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load fine-tuned model."""
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_history = checkpoint.get('history', [])


class LearningRateScheduler:
    """Custom learning rate scheduling for medical model training."""
    
    @staticmethod
    def cosine_annealing(epoch: int, initial_lr: float, max_epochs: int) -> float:
        """Cosine annealing schedule."""
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
    
    @staticmethod
    def exponential_decay(epoch: int, initial_lr: float, decay_rate: float) -> float:
        """Exponential decay schedule."""
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def linear_warmup_then_cosine(
        epoch: int,
        initial_lr: float,
        max_epochs: int,
        warmup_epochs: int
    ) -> float:
        """Linear warmup followed by cosine annealing."""
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch / warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
