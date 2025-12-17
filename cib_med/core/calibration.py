"""
Isotonic Calibration for Target Coordinates.

This module implements the isotonic regression calibration described in
Section 2.2, Eq. (2), which maps raw classifier logits to calibrated
probability coordinates.
"""

from typing import Optional, Tuple, List
from pathlib import Path
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import torch


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for target coordinate.
    
    Maps raw logits to calibrated probabilities via order-preserving
    isotonic regression, as described in Eq. (2):
    
        p_eff(x) := g(ℓ_eff(x)) ∈ [0,1]
    
    where g is the isotonic regression map fit on held-out labeled data.
    
    Args:
        clip_min: Minimum output value
        clip_max: Maximum output value
    """
    
    def __init__(
        self,
        clip_min: float = 0.001,
        clip_max: float = 0.999,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.model = IsotonicRegression(
            y_min=clip_min,
            y_max=clip_max,
            out_of_bounds="clip",
        )
        self._is_fitted = False
        self._logit_range: Optional[Tuple[float, float]] = None
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Fit calibrator on held-out validation data.
        
        Args:
            logits: Raw classifier logits [N,]
            labels: Binary ground truth labels [N,]
            
        Returns:
            self (fitted calibrator)
        """
        logits = np.asarray(logits).flatten()
        labels = np.asarray(labels).flatten()
        
        if len(logits) != len(labels):
            raise ValueError("logits and labels must have same length")
        
        # Store logit range for extrapolation handling
        self._logit_range = (float(logits.min()), float(logits.max()))
        
        # Fit isotonic regression
        self.model.fit(logits, labels)
        self._is_fitted = True
        
        return self
    
    def transform(self, logit: float) -> float:
        """
        Transform a single logit to calibrated probability.
        
        Args:
            logit: Raw classifier logit
            
        Returns:
            Calibrated probability in [clip_min, clip_max]
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        logit_arr = np.array([[logit]])
        prob = self.model.transform(logit_arr)[0]
        return float(np.clip(prob, self.clip_min, self.clip_max))
    
    def transform_batch(self, logits: np.ndarray) -> np.ndarray:
        """
        Transform a batch of logits to calibrated probabilities.
        
        Args:
            logits: Array of raw logits
            
        Returns:
            Array of calibrated probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        logits = np.asarray(logits).flatten()
        probs = self.model.transform(logits)
        return np.clip(probs, self.clip_min, self.clip_max)
    
    def evaluate_calibration(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """
        Evaluate calibration quality on test data.
        
        Args:
            logits: Raw classifier logits
            labels: Binary ground truth labels
            n_bins: Number of bins for reliability diagram
            
        Returns:
            Dictionary with calibration metrics
        """
        probs = self.transform_batch(logits)
        
        # Compute reliability diagram data
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probs, n_bins=n_bins, strategy="uniform"
        )
        
        # Expected calibration error (ECE)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bin_edges[1:-1])
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_acc = labels[mask].mean()
                bin_conf = probs[mask].mean()
                bin_size = mask.sum() / len(probs)
                ece += bin_size * abs(bin_acc - bin_conf)
        
        # Maximum calibration error (MCE)
        mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Brier score
        brier = np.mean((probs - labels) ** 2)
        
        return {
            "ece": float(ece),
            "mce": float(mce),
            "brier_score": float(brier),
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        }
    
    def save(self, path: Path):
        """Save calibrator to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model parameters
        data = {
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "is_fitted": self._is_fitted,
            "logit_range": self._logit_range,
        }
        
        if self._is_fitted:
            data["X_thresholds_"] = self.model.X_thresholds_.tolist()
            data["y_thresholds_"] = self.model.y_thresholds_.tolist()
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        """Load calibrator from disk."""
        with open(path) as f:
            data = json.load(f)
        
        calibrator = cls(
            clip_min=data["clip_min"],
            clip_max=data["clip_max"],
        )
        calibrator._is_fitted = data["is_fitted"]
        calibrator._logit_range = tuple(data["logit_range"]) if data["logit_range"] else None
        
        if calibrator._is_fitted:
            calibrator.model.X_thresholds_ = np.array(data["X_thresholds_"])
            calibrator.model.y_thresholds_ = np.array(data["y_thresholds_"])
        
        return calibrator


class PlattScalingCalibrator:
    """
    Platt scaling calibrator (alternative to isotonic regression).
    
    Uses logistic regression to map logits to calibrated probabilities:
    
        p(x) = σ(a * logit + b)
    
    where a and b are learned parameters.
    """
    
    def __init__(self):
        self.a: float = 1.0
        self.b: float = 0.0
        self._is_fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> "PlattScalingCalibrator":
        """
        Fit Platt scaling parameters via gradient descent.
        
        Args:
            logits: Raw classifier logits [N,]
            labels: Binary ground truth labels [N,]
            max_iter: Maximum optimization iterations
            lr: Learning rate
            
        Returns:
            self (fitted calibrator)
        """
        logits = np.asarray(logits).flatten()
        labels = np.asarray(labels).flatten()
        
        # Initialize parameters
        a = 1.0
        b = 0.0
        
        for _ in range(max_iter):
            # Forward pass
            z = a * logits + b
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            
            # Gradients (NLL loss)
            grad_z = p - labels
            grad_a = np.mean(grad_z * logits)
            grad_b = np.mean(grad_z)
            
            # Update
            a -= lr * grad_a
            b -= lr * grad_b
        
        self.a = float(a)
        self.b = float(b)
        self._is_fitted = True
        
        return self
    
    def transform(self, logit: float) -> float:
        """Transform a single logit to calibrated probability."""
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        z = self.a * logit + self.b
        return float(1.0 / (1.0 + np.exp(-z)))
    
    def transform_batch(self, logits: np.ndarray) -> np.ndarray:
        """Transform a batch of logits to calibrated probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
        
        logits = np.asarray(logits).flatten()
        z = self.a * logits + self.b
        return 1.0 / (1.0 + np.exp(-z))


def fit_calibrator_from_dataset(
    evaluator,
    dataset,
    target_label: str = "Pleural Effusion",
    calibrator_type: str = "isotonic",
) -> IsotonicCalibrator:
    """
    Fit calibrator from a labeled dataset.
    
    Args:
        evaluator: RadiologyEvaluator instance
        dataset: Dataset with images and labels
        target_label: Name of target finding
        calibrator_type: Type of calibrator ("isotonic" or "platt")
        
    Returns:
        Fitted calibrator
    """
    logits_list = []
    labels_list = []
    
    label_idx = evaluator.get_label_index(target_label)
    
    for batch in dataset:
        images, labels = batch["image"], batch["labels"]
        
        with torch.no_grad():
            logits = evaluator(images)
        
        logits_list.append(logits[:, label_idx].cpu().numpy())
        labels_list.append(labels[:, label_idx].cpu().numpy())
    
    all_logits = np.concatenate(logits_list)
    all_labels = np.concatenate(labels_list)
    
    if calibrator_type == "isotonic":
        calibrator = IsotonicCalibrator()
    else:
        calibrator = PlattScalingCalibrator()
    
    calibrator.fit(all_logits, all_labels)
    return calibrator
