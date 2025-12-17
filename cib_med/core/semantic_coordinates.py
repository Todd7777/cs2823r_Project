"""
Semantic Coordinate System for Clinical Findings.

This module implements the semantic coordinate system described in Section 2 of the paper,
mapping images to clinically interpretable multi-label scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, TYPE_CHECKING
from enum import Enum
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from cib_med.core.evaluator import RadiologyEvaluator
    from cib_med.core.calibration import IsotonicCalibrator


class FindingCategory(Enum):
    """Clinical finding categories for off-target grouping."""
    PARENCHYMAL = "parenchymal"
    CARDIOMEDIASTINAL = "cardiomediastinal"
    PLEURAL = "pleural"
    CHRONIC = "chronic"
    ARTIFACT = "artifact"


@dataclass
class ClinicalFinding:
    """Represents a clinical finding with its properties."""
    name: str
    category: FindingCategory
    description: str
    is_target: bool = False
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, ClinicalFinding):
            return self.name == other.name
        return False


# Standard CXR findings as defined in Eq. (3) of the paper
STANDARD_FINDINGS: Dict[str, ClinicalFinding] = {
    # Target finding
    "Pleural Effusion": ClinicalFinding(
        name="Pleural Effusion",
        category=FindingCategory.PLEURAL,
        description="Accumulation of fluid in the pleural space",
        is_target=True
    ),
    # Parenchymal findings
    "Atelectasis": ClinicalFinding(
        name="Atelectasis",
        category=FindingCategory.PARENCHYMAL,
        description="Partial or complete lung collapse"
    ),
    "Consolidation": ClinicalFinding(
        name="Consolidation",
        category=FindingCategory.PARENCHYMAL,
        description="Lung tissue filled with fluid/inflammatory material"
    ),
    "Pneumonia": ClinicalFinding(
        name="Pneumonia",
        category=FindingCategory.PARENCHYMAL,
        description="Infection causing inflammation in the lungs"
    ),
    "Edema": ClinicalFinding(
        name="Edema",
        category=FindingCategory.PARENCHYMAL,
        description="Pulmonary edema - fluid accumulation in lungs"
    ),
    "Lung Opacity": ClinicalFinding(
        name="Lung Opacity",
        category=FindingCategory.PARENCHYMAL,
        description="Non-specific increased opacity in lung fields"
    ),
    # Cardiomediastinal findings
    "Cardiomegaly": ClinicalFinding(
        name="Cardiomegaly",
        category=FindingCategory.CARDIOMEDIASTINAL,
        description="Enlarged heart"
    ),
    "Enlarged Cardiomediastinum": ClinicalFinding(
        name="Enlarged Cardiomediastinum",
        category=FindingCategory.CARDIOMEDIASTINAL,
        description="Widened mediastinal silhouette"
    ),
    # Other pleural findings (non-effusion)
    "Pneumothorax": ClinicalFinding(
        name="Pneumothorax",
        category=FindingCategory.PLEURAL,
        description="Air in the pleural space"
    ),
    "Pleural Thickening": ClinicalFinding(
        name="Pleural Thickening",
        category=FindingCategory.PLEURAL,
        description="Thickening of pleural membrane"
    ),
    "Pleural Other": ClinicalFinding(
        name="Pleural Other",
        category=FindingCategory.PLEURAL,
        description="Other pleural abnormalities"
    ),
    # Chronic findings
    "Fibrosis": ClinicalFinding(
        name="Fibrosis",
        category=FindingCategory.CHRONIC,
        description="Scarring of lung tissue"
    ),
    "Emphysema": ClinicalFinding(
        name="Emphysema",
        category=FindingCategory.CHRONIC,
        description="Destruction of alveolar walls"
    ),
    # Artifact/intervention canaries
    "Support Devices": ClinicalFinding(
        name="Support Devices",
        category=FindingCategory.ARTIFACT,
        description="Medical devices (tubes, lines, pacemakers)"
    ),
    "Fracture": ClinicalFinding(
        name="Fracture",
        category=FindingCategory.ARTIFACT,
        description="Bone fractures visible on CXR"
    ),
}


@dataclass
class SemanticCoordinates:
    """
    Container for semantic coordinates of an image.
    
    Attributes:
        logits: Raw classifier logits for each finding
        probabilities: Sigmoid probabilities v_k(x) as in Eq. (1)
        calibrated_target: Calibrated target coordinate p_eff(x) as in Eq. (2)
        image_id: Optional identifier for the source image
    """
    logits: Dict[str, float]
    probabilities: Dict[str, float]
    calibrated_target: float
    image_id: Optional[str] = None
    
    def get_target_logit(self) -> float:
        """Get the raw effusion logit for gradient guidance."""
        return self.logits.get("Pleural Effusion", 0.0)
    
    def get_off_target_vector(self, findings: List[str]) -> np.ndarray:
        """Get vector of off-target probabilities."""
        return np.array([self.probabilities.get(f, 0.0) for f in findings])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "logits": self.logits,
            "probabilities": self.probabilities,
            "calibrated_target": self.calibrated_target,
            "image_id": self.image_id,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "SemanticCoordinates":
        """Create from dictionary."""
        return cls(**d)


class SemanticCoordinateSystem:
    """
    Maps images to clinical semantic coordinates.
    
    This class implements the semantic coordinate system from Section 2.2,
    providing:
    - Target coordinate p_eff(x) via isotonic calibration
    - Off-target coordinates v_k(x) via sigmoid transformation
    - Finding groupings for ablation studies
    
    Args:
        evaluator: RadiologyEvaluator instance for computing logits
        calibrator: Optional IsotonicCalibrator for target calibration
        target_finding: Name of target finding (default: "Pleural Effusion")
        off_target_findings: List of off-target finding names (default: standard K set)
    """
    
    def __init__(
        self,
        evaluator: "RadiologyEvaluator",
        calibrator: Optional["IsotonicCalibrator"] = None,
        target_finding: str = "Pleural Effusion",
        off_target_findings: Optional[List[str]] = None,
    ):
        self.evaluator = evaluator
        self.calibrator = calibrator
        self.target_finding = target_finding
        
        # Default off-target set K as defined in Eq. (3)
        if off_target_findings is None:
            self.off_target_findings = [
                "Atelectasis", "Consolidation", "Pneumonia", "Edema", "Lung Opacity",
                "Cardiomegaly", "Enlarged Cardiomediastinum",
                "Pneumothorax", "Pleural Thickening", "Pleural Other",
                "Fibrosis", "Emphysema", "Support Devices", "Fracture"
            ]
        else:
            self.off_target_findings = off_target_findings
        
        # Validate findings
        self._validate_findings()
        
        # Build finding groups for ablations
        self.finding_groups = self._build_finding_groups()
    
    def _validate_findings(self):
        """Validate that all specified findings are known."""
        all_findings = set(STANDARD_FINDINGS.keys())
        
        if self.target_finding not in all_findings:
            raise ValueError(f"Unknown target finding: {self.target_finding}")
        
        for f in self.off_target_findings:
            if f not in all_findings:
                raise ValueError(f"Unknown off-target finding: {f}")
    
    def _build_finding_groups(self) -> Dict[FindingCategory, List[str]]:
        """Build groups of findings by category for ablation studies."""
        groups = {cat: [] for cat in FindingCategory}
        
        for name in self.off_target_findings:
            finding = STANDARD_FINDINGS[name]
            groups[finding.category].append(name)
        
        return groups
    
    def compute_coordinates(
        self,
        image: torch.Tensor,
        image_id: Optional[str] = None,
    ) -> SemanticCoordinates:
        """
        Compute semantic coordinates for an image.
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            image_id: Optional identifier for the image
            
        Returns:
            SemanticCoordinates containing logits, probabilities, and calibrated target
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Get logits from evaluator
        logits_tensor = self.evaluator(image)  # [B, num_labels]
        
        # Convert to probabilities via sigmoid (Eq. 1)
        probs_tensor = torch.sigmoid(logits_tensor)
        
        # Extract values for first image in batch
        logits_np = logits_tensor[0].detach().cpu().numpy()
        probs_np = probs_tensor[0].detach().cpu().numpy()
        
        # Build dictionaries
        label_names = self.evaluator.get_label_names()
        logits = {name: float(logits_np[i]) for i, name in enumerate(label_names)}
        probabilities = {name: float(probs_np[i]) for i, name in enumerate(label_names)}
        
        # Compute calibrated target coordinate (Eq. 2)
        target_logit = logits[self.target_finding]
        if self.calibrator is not None:
            calibrated_target = self.calibrator.transform(target_logit)
        else:
            # Fall back to sigmoid if no calibrator
            calibrated_target = float(torch.sigmoid(torch.tensor(target_logit)))
        
        return SemanticCoordinates(
            logits=logits,
            probabilities=probabilities,
            calibrated_target=calibrated_target,
            image_id=image_id,
        )
    
    def compute_coordinates_batch(
        self,
        images: torch.Tensor,
        image_ids: Optional[List[str]] = None,
    ) -> List[SemanticCoordinates]:
        """
        Compute semantic coordinates for a batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
            image_ids: Optional list of identifiers
            
        Returns:
            List of SemanticCoordinates for each image
        """
        if image_ids is None:
            image_ids = [None] * images.shape[0]
        
        logits_tensor = self.evaluator(images)
        probs_tensor = torch.sigmoid(logits_tensor)
        
        label_names = self.evaluator.get_label_names()
        results = []
        
        for i in range(images.shape[0]):
            logits_np = logits_tensor[i].detach().cpu().numpy()
            probs_np = probs_tensor[i].detach().cpu().numpy()
            
            logits = {name: float(logits_np[j]) for j, name in enumerate(label_names)}
            probabilities = {name: float(probs_np[j]) for j, name in enumerate(label_names)}
            
            target_logit = logits[self.target_finding]
            if self.calibrator is not None:
                calibrated_target = self.calibrator.transform(target_logit)
            else:
                calibrated_target = float(torch.sigmoid(torch.tensor(target_logit)))
            
            results.append(SemanticCoordinates(
                logits=logits,
                probabilities=probabilities,
                calibrated_target=calibrated_target,
                image_id=image_ids[i],
            ))
        
        return results
    
    def get_off_target_set(self, exclude: Optional[Set[str]] = None) -> List[str]:
        """
        Get off-target findings, optionally excluding some.
        
        Used for leave-one-out ablations (Section 3.5).
        
        Args:
            exclude: Set of finding names to exclude
            
        Returns:
            List of off-target finding names
        """
        if exclude is None:
            return self.off_target_findings.copy()
        return [f for f in self.off_target_findings if f not in exclude]
    
    def get_findings_by_category(
        self,
        categories: List[FindingCategory],
    ) -> List[str]:
        """
        Get findings belonging to specified categories.
        
        Used for grouped ablations (Section 3.5).
        
        Args:
            categories: List of FindingCategory to include
            
        Returns:
            List of finding names in those categories
        """
        result = []
        for cat in categories:
            result.extend(self.finding_groups.get(cat, []))
        return result
    
    def compute_association(
        self,
        coordinates_list: List[SemanticCoordinates],
        method: str = "spearman",
    ) -> Dict[str, float]:
        """
        Compute empirical associations between target and off-target coordinates.
        
        Implements Assoc(k) from Section 6.1.
        
        Args:
            coordinates_list: List of SemanticCoordinates from anchor distribution
            method: Correlation method ("spearman" or "pearson")
            
        Returns:
            Dictionary mapping finding names to association values
        """
        from scipy.stats import spearmanr, pearsonr
        
        n = len(coordinates_list)
        if n < 3:
            return {f: 0.0 for f in self.off_target_findings}
        
        # Extract target values
        target_values = np.array([c.calibrated_target for c in coordinates_list])
        
        associations = {}
        for finding in self.off_target_findings:
            off_target_values = np.array([
                c.probabilities.get(finding, 0.0) for c in coordinates_list
            ])
            
            if method == "spearman":
                corr, _ = spearmanr(target_values, off_target_values)
            else:
                corr, _ = pearsonr(target_values, off_target_values)
            
            associations[finding] = float(corr) if not np.isnan(corr) else 0.0
        
        return associations
