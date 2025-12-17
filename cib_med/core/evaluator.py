"""
Radiology Evaluator Interface.

This module implements the frozen radiology evaluator F described in Section 2.2,
which maps images to clinically interpretable multi-label scores.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models


# Standard CheXpert/MIMIC-CXR label ordering
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

EXTENDED_LABELS = CHEXPERT_LABELS + [
    "Pleural Thickening",
    "Fibrosis",
    "Emphysema",
]


class RadiologyEvaluator(ABC):
    """
    Abstract base class for radiology evaluators.
    
    The evaluator maps images to multi-label logits for clinical findings.
    It is used ONLY for evaluation and guidance, never trained on generated samples.
    """
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W], normalized to [0, 1] or [-1, 1]
            
        Returns:
            Logits tensor [B, num_labels]
        """
        pass
    
    @abstractmethod
    def get_label_names(self) -> List[str]:
        """Return list of label names in order."""
        pass
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward(images)
    
    def get_label_index(self, label_name: str) -> int:
        """Get index for a specific label."""
        labels = self.get_label_names()
        return labels.index(label_name)
    
    def get_gradients(
        self,
        images: torch.Tensor,
        label_name: str,
    ) -> torch.Tensor:
        """
        Compute gradients of a specific label's logit w.r.t. images.
        
        Used for classifier guidance in diffusion models.
        
        Args:
            images: Input images [B, C, H, W] with requires_grad=True
            label_name: Name of target label
            
        Returns:
            Gradient tensor [B, C, H, W]
        """
        images = images.requires_grad_(True)
        logits = self.forward(images)
        label_idx = self.get_label_index(label_name)
        target_logit = logits[:, label_idx].sum()
        target_logit.backward()
        return images.grad


class DenseNetEvaluator(RadiologyEvaluator):
    """
    DenseNet-121 based CXR evaluator.
    
    This is a common architecture for CXR classification, trained on
    CheXpert or MIMIC-CXR datasets.
    
    Args:
        weights_path: Path to pretrained weights
        num_labels: Number of output labels
        label_names: Optional custom label names
        device: Compute device
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_labels: int = 14,
        label_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.num_labels = num_labels
        self._label_names = label_names or CHEXPERT_LABELS[:num_labels]
        
        # Build model
        self.model = models.densenet121(pretrained=weights_path is None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_labels)
        
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Standard CXR preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute logits for images."""
        images = images.to(self.device)
        
        # Handle grayscale by repeating channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        # Apply preprocessing
        images = self.transform(images)
        
        with torch.set_grad_enabled(images.requires_grad):
            logits = self.model(images)
        
        return logits
    
    def get_label_names(self) -> List[str]:
        return self._label_names


class ResNetEvaluator(RadiologyEvaluator):
    """
    ResNet-50 based CXR evaluator.
    
    Alternative architecture for CXR classification.
    """
    
    def __init__(
        self,
        weights_path: Optional[str] = None,
        num_labels: int = 14,
        label_names: Optional[List[str]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.num_labels = num_labels
        self._label_names = label_names or CHEXPERT_LABELS[:num_labels]
        
        self.model = models.resnet50(pretrained=weights_path is None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_labels)
        
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        images = self.transform(images)
        with torch.set_grad_enabled(images.requires_grad):
            logits = self.model(images)
        return logits
    
    def get_label_names(self) -> List[str]:
        return self._label_names


class EnsembleEvaluator(RadiologyEvaluator):
    """
    Ensemble of multiple evaluators for more robust predictions.
    
    Combines predictions from multiple models via averaging.
    """
    
    def __init__(
        self,
        evaluators: List[RadiologyEvaluator],
        weights: Optional[List[float]] = None,
    ):
        self.evaluators = evaluators
        
        if weights is None:
            weights = [1.0 / len(evaluators)] * len(evaluators)
        self.weights = weights
        
        # Verify all evaluators have same labels
        label_names = evaluators[0].get_label_names()
        for e in evaluators[1:]:
            if e.get_label_names() != label_names:
                raise ValueError("All evaluators must have same label names")
        
        self._label_names = label_names
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits_list = [e(images) for e in self.evaluators]
        weighted_logits = sum(w * l for w, l in zip(self.weights, logits_list))
        return weighted_logits
    
    def get_label_names(self) -> List[str]:
        return self._label_names


class MockEvaluator(RadiologyEvaluator):
    """
    Mock evaluator for testing and development.
    
    Returns random but consistent logits based on image statistics.
    Useful for unit tests and debugging without GPU.
    """
    
    def __init__(
        self,
        num_labels: int = 14,
        label_names: Optional[List[str]] = None,
        seed: int = 42,
    ):
        self.num_labels = num_labels
        self._label_names = label_names or CHEXPERT_LABELS[:num_labels]
        self.rng = np.random.RandomState(seed)
        
        # Fixed projection matrix for consistent outputs
        self._proj = torch.tensor(
            self.rng.randn(num_labels, 224 * 224),
            dtype=torch.float32
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        
        # Resize and flatten
        images_resized = F.interpolate(images, size=(224, 224), mode="bilinear")
        if images_resized.shape[1] > 1:
            images_resized = images_resized.mean(dim=1, keepdim=True)
        images_flat = images_resized.view(B, -1)
        
        # Project to logits
        self._proj = self._proj.to(images.device)
        logits = torch.matmul(images_flat, self._proj.T) / 1000.0
        
        return logits
    
    def get_label_names(self) -> List[str]:
        return self._label_names


def load_evaluator(
    model_type: str = "densenet",
    weights_path: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> RadiologyEvaluator:
    """
    Factory function to load an evaluator.
    
    Args:
        model_type: One of "densenet", "resnet", "mock"
        weights_path: Path to pretrained weights
        device: Compute device
        **kwargs: Additional arguments passed to evaluator
        
    Returns:
        RadiologyEvaluator instance
    """
    if model_type == "densenet":
        return DenseNetEvaluator(weights_path=weights_path, device=device, **kwargs)
    elif model_type == "resnet":
        return ResNetEvaluator(weights_path=weights_path, device=device, **kwargs)
    elif model_type == "mock":
        return MockEvaluator(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
