"""
Anchor Selection and Management.

This module implements the anchor selection protocol described in Section 5.1,
ensuring that anchors are suitable for meaningful trajectory evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from cib_med.core.semantic_coordinates import SemanticCoordinates, SemanticCoordinateSystem


@dataclass
class Anchor:
    """
    Represents an anchor image with its metadata and semantic profile.
    
    Attributes:
        anchor_id: Unique identifier
        image: Image tensor [C, H, W]
        coordinates: Semantic coordinates at anchor
        metadata: Additional metadata (patient info, study info, etc.)
        clinical_profile: Summary of clinical findings
    """
    anchor_id: str
    image: torch.Tensor
    coordinates: SemanticCoordinates
    metadata: Dict[str, Any] = field(default_factory=dict)
    clinical_profile: Optional[Dict[str, str]] = None
    
    @property
    def target_score(self) -> float:
        """Get calibrated target score."""
        return self.coordinates.calibrated_target
    
    def has_finding(self, finding: str, threshold: float = 0.5) -> bool:
        """Check if anchor has a specific finding above threshold."""
        return self.coordinates.probabilities.get(finding, 0.0) > threshold
    
    def get_findings_summary(self, threshold: float = 0.5) -> List[str]:
        """Get list of findings above threshold."""
        return [
            name for name, prob in self.coordinates.probabilities.items()
            if prob > threshold
        ]
    
    def to_dict(self) -> Dict:
        """Serialize anchor (excluding image tensor)."""
        return {
            "anchor_id": self.anchor_id,
            "coordinates": self.coordinates.to_dict(),
            "metadata": self.metadata,
            "clinical_profile": self.clinical_profile,
        }
    
    def save(self, path: Path, save_image: bool = True):
        """Save anchor to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / "anchor.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        if save_image:
            torch.save(self.image, path / "image.pt")
    
    @classmethod
    def load(cls, path: Path, load_image: bool = True) -> "Anchor":
        """Load anchor from disk."""
        path = Path(path)
        
        with open(path / "anchor.json") as f:
            data = json.load(f)
        
        image = None
        if load_image and (path / "image.pt").exists():
            image = torch.load(path / "image.pt")
        
        return cls(
            anchor_id=data["anchor_id"],
            image=image,
            coordinates=SemanticCoordinates.from_dict(data["coordinates"]),
            metadata=data.get("metadata", {}),
            clinical_profile=data.get("clinical_profile"),
        )


class AnchorSelector:
    """
    Selects suitable anchors from a dataset based on CIB-Med-1 criteria.
    
    Selection criteria (from Section 5.1):
    1. Non-saturated target score p_eff ∈ (0.1, 0.9)
    2. Well-defined evaluator outputs across off-target set K
    3. Diversity across baseline clinical profiles
    
    Args:
        coordinate_system: SemanticCoordinateSystem for computing coordinates
        target_range: (min, max) range for target score
        min_off_target_variance: Minimum variance in off-target scores
        stratify_by: Optional finding to stratify selection by
    """
    
    def __init__(
        self,
        coordinate_system: SemanticCoordinateSystem,
        target_range: Tuple[float, float] = (0.1, 0.9),
        min_off_target_variance: float = 0.01,
        stratify_by: Optional[str] = None,
    ):
        self.coordinate_system = coordinate_system
        self.target_range = target_range
        self.min_off_target_variance = min_off_target_variance
        self.stratify_by = stratify_by
    
    def is_valid_anchor(self, coordinates: SemanticCoordinates) -> bool:
        """Check if coordinates meet anchor selection criteria."""
        # Criterion 1: Non-saturated target score
        target = coordinates.calibrated_target
        if not (self.target_range[0] <= target <= self.target_range[1]):
            return False
        
        # Criterion 2: Well-defined off-target outputs
        off_target_probs = [
            coordinates.probabilities.get(f, -1)
            for f in self.coordinate_system.off_target_findings
        ]
        
        # Check for missing/invalid values
        if any(p < 0 for p in off_target_probs):
            return False
        
        # Check for minimum variance (not all zeros or ones)
        if np.var(off_target_probs) < self.min_off_target_variance:
            return False
        
        return True
    
    def select_from_dataset(
        self,
        dataset: Dataset,
        max_anchors: int = 100,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "AnchorSet":
        """
        Select anchors from a dataset.
        
        Args:
            dataset: PyTorch dataset yielding images
            max_anchors: Maximum number of anchors to select
            batch_size: Batch size for processing
            device: Compute device
            
        Returns:
            AnchorSet containing selected anchors
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        candidates = []
        
        for batch_idx, batch in enumerate(loader):
            if len(candidates) >= max_anchors * 2:  # Oversample for filtering
                break
            
            if isinstance(batch, dict):
                images = batch["image"]
                ids = batch.get("id", [f"img_{batch_idx}_{i}" for i in range(len(images))])
            else:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                ids = [f"img_{batch_idx}_{i}" for i in range(len(images))]
            
            images = images.to(device)
            
            # Compute coordinates
            coords_list = self.coordinate_system.compute_coordinates_batch(
                images, image_ids=ids
            )
            
            # Filter valid anchors
            for i, (img, coords) in enumerate(zip(images, coords_list)):
                if self.is_valid_anchor(coords):
                    candidates.append(Anchor(
                        anchor_id=ids[i] if isinstance(ids, list) else f"{ids}_{i}",
                        image=img.cpu(),
                        coordinates=coords,
                    ))
        
        # Apply stratification if requested
        if self.stratify_by is not None:
            candidates = self._stratify_candidates(candidates)
        
        # Select final anchors
        selected = candidates[:max_anchors]
        
        return AnchorSet(selected)
    
    def _stratify_candidates(self, candidates: List[Anchor]) -> List[Anchor]:
        """Stratify candidates to ensure diversity."""
        if not candidates:
            return candidates
        
        # Group by stratification finding
        high_group = [c for c in candidates if c.has_finding(self.stratify_by)]
        low_group = [c for c in candidates if not c.has_finding(self.stratify_by)]
        
        # Interleave groups
        result = []
        max_len = max(len(high_group), len(low_group))
        
        for i in range(max_len):
            if i < len(high_group):
                result.append(high_group[i])
            if i < len(low_group):
                result.append(low_group[i])
        
        return result
    
    def select_from_images(
        self,
        images: List[torch.Tensor],
        image_ids: Optional[List[str]] = None,
    ) -> "AnchorSet":
        """
        Select anchors from a list of images.
        
        Args:
            images: List of image tensors
            image_ids: Optional list of identifiers
            
        Returns:
            AnchorSet containing valid anchors
        """
        if image_ids is None:
            image_ids = [f"img_{i}" for i in range(len(images))]
        
        anchors = []
        
        for img, img_id in zip(images, image_ids):
            coords = self.coordinate_system.compute_coordinates(img, image_id=img_id)
            
            if self.is_valid_anchor(coords):
                anchors.append(Anchor(
                    anchor_id=img_id,
                    image=img,
                    coordinates=coords,
                ))
        
        return AnchorSet(anchors)


class AnchorSet:
    """
    Collection of anchors for benchmark evaluation.
    
    Provides utilities for analysis, iteration, and persistence.
    """
    
    def __init__(self, anchors: List[Anchor]):
        self.anchors = anchors
        self._id_map = {a.anchor_id: a for a in anchors}
    
    def __len__(self) -> int:
        return len(self.anchors)
    
    def __getitem__(self, idx: int) -> Anchor:
        return self.anchors[idx]
    
    def __iter__(self):
        return iter(self.anchors)
    
    def get_by_id(self, anchor_id: str) -> Optional[Anchor]:
        """Get anchor by ID."""
        return self._id_map.get(anchor_id)
    
    def get_target_distribution(self) -> np.ndarray:
        """Get distribution of target scores across anchors."""
        return np.array([a.target_score for a in self.anchors])
    
    def get_off_target_distribution(self, finding: str) -> np.ndarray:
        """Get distribution of a specific off-target finding."""
        return np.array([
            a.coordinates.probabilities.get(finding, 0.0) for a in self.anchors
        ])
    
    def compute_associations(
        self,
        coordinate_system: SemanticCoordinateSystem,
    ) -> Dict[str, float]:
        """
        Compute target-off_target associations across anchors.
        
        Implements Assoc(k) from Section 6.1.
        """
        coords_list = [a.coordinates for a in self.anchors]
        return coordinate_system.compute_association(coords_list)
    
    def filter_by_target_range(
        self,
        min_score: float,
        max_score: float,
    ) -> "AnchorSet":
        """Filter anchors by target score range."""
        filtered = [
            a for a in self.anchors
            if min_score <= a.target_score <= max_score
        ]
        return AnchorSet(filtered)
    
    def filter_by_finding(
        self,
        finding: str,
        present: bool = True,
        threshold: float = 0.5,
    ) -> "AnchorSet":
        """Filter anchors by presence/absence of a finding."""
        filtered = [
            a for a in self.anchors
            if a.has_finding(finding, threshold) == present
        ]
        return AnchorSet(filtered)
    
    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> Tuple["AnchorSet", "AnchorSet"]:
        """Split into train/test sets."""
        np.random.seed(seed)
        indices = np.random.permutation(len(self.anchors))
        split_idx = int(len(indices) * train_ratio)
        
        train_anchors = [self.anchors[i] for i in indices[:split_idx]]
        test_anchors = [self.anchors[i] for i in indices[split_idx:]]
        
        return AnchorSet(train_anchors), AnchorSet(test_anchors)
    
    def save(self, path: Path, save_images: bool = True):
        """Save anchor set to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index = {
            "num_anchors": len(self.anchors),
            "anchor_ids": [a.anchor_id for a in self.anchors],
        }
        with open(path / "index.json", "w") as f:
            json.dump(index, f, indent=2)
        
        # Save individual anchors
        for anchor in self.anchors:
            anchor_dir = path / anchor.anchor_id
            anchor.save(anchor_dir, save_image=save_images)
    
    @classmethod
    def load(cls, path: Path, load_images: bool = True) -> "AnchorSet":
        """Load anchor set from disk."""
        path = Path(path)
        
        with open(path / "index.json") as f:
            index = json.load(f)
        
        anchors = []
        for anchor_id in index["anchor_ids"]:
            anchor_dir = path / anchor_id
            if anchor_dir.exists():
                anchors.append(Anchor.load(anchor_dir, load_image=load_images))
        
        return cls(anchors)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the anchor set."""
        target_scores = self.get_target_distribution()
        
        return {
            "num_anchors": len(self.anchors),
            "target_score_mean": float(np.mean(target_scores)),
            "target_score_std": float(np.std(target_scores)),
            "target_score_min": float(np.min(target_scores)),
            "target_score_max": float(np.max(target_scores)),
            "target_score_median": float(np.median(target_scores)),
        }
