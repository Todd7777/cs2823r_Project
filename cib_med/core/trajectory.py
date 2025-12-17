"""
Edit Trajectory Management.

This module implements the trajectory representation and generation
described in Section 2.1 of the paper.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Iterator
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image

from cib_med.core.semantic_coordinates import SemanticCoordinates, SemanticCoordinateSystem


@dataclass
class TrajectoryStep:
    """
    Single step in an edit trajectory.
    
    Attributes:
        step_index: Index t in the trajectory
        image: Image tensor at this step
        coordinates: Semantic coordinates at this step
        guidance_params: Parameters used for this edit step
        metadata: Additional step-level metadata
    """
    step_index: int
    image: torch.Tensor
    coordinates: SemanticCoordinates
    guidance_params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Serialize step (excluding image tensor)."""
        return {
            "step_index": self.step_index,
            "coordinates": self.coordinates.to_dict(),
            "guidance_params": self.guidance_params,
            "metadata": self.metadata,
        }


@dataclass
class EditTrajectory:
    """
    Complete edit trajectory from anchor to final image.
    
    Represents {x^t}_{t=0}^T as described in Section 2.1.
    
    Attributes:
        anchor_id: Identifier for the anchor image
        steps: List of trajectory steps
        editor_name: Name of the editing method used
        config: Configuration used for generation
        seed: Random seed used
    """
    anchor_id: str
    steps: List[TrajectoryStep]
    editor_name: str
    config: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    
    @property
    def anchor(self) -> TrajectoryStep:
        """Get anchor step (t=0)."""
        return self.steps[0]
    
    @property
    def final(self) -> TrajectoryStep:
        """Get final step (t=T)."""
        return self.steps[-1]
    
    @property
    def num_steps(self) -> int:
        """Number of edit steps T."""
        return len(self.steps) - 1
    
    def get_target_progression(self) -> np.ndarray:
        """Get array of calibrated target values over trajectory."""
        return np.array([s.coordinates.calibrated_target for s in self.steps])
    
    def get_off_target_progression(self, finding: str) -> np.ndarray:
        """Get array of off-target probability values for a specific finding."""
        return np.array([
            s.coordinates.probabilities.get(finding, 0.0) for s in self.steps
        ])
    
    def get_images(self) -> torch.Tensor:
        """Stack all images into a single tensor [T+1, C, H, W]."""
        return torch.stack([s.image for s in self.steps])
    
    def iter_pairs(self) -> Iterator[tuple]:
        """Iterate over consecutive (step_t, step_t+1) pairs."""
        for i in range(len(self.steps) - 1):
            yield self.steps[i], self.steps[i + 1]
    
    def to_dict(self) -> Dict:
        """Serialize trajectory (excluding image tensors)."""
        return {
            "anchor_id": self.anchor_id,
            "steps": [s.to_dict() for s in self.steps],
            "editor_name": self.editor_name,
            "config": self.config,
            "seed": self.seed,
        }
    
    def save(self, path: Path, save_images: bool = True):
        """
        Save trajectory to disk.
        
        Args:
            path: Directory to save trajectory
            save_images: Whether to save image tensors
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        with open(path / "trajectory.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save images
        if save_images:
            for step in self.steps:
                img_path = path / f"step_{step.step_index:03d}.pt"
                torch.save(step.image, img_path)
    
    @classmethod
    def load(cls, path: Path, load_images: bool = True) -> "EditTrajectory":
        """Load trajectory from disk."""
        path = Path(path)
        
        with open(path / "trajectory.json") as f:
            data = json.load(f)
        
        steps = []
        for step_data in data["steps"]:
            image = None
            if load_images:
                img_path = path / f"step_{step_data['step_index']:03d}.pt"
                if img_path.exists():
                    image = torch.load(img_path)
            
            steps.append(TrajectoryStep(
                step_index=step_data["step_index"],
                image=image,
                coordinates=SemanticCoordinates.from_dict(step_data["coordinates"]),
                guidance_params=step_data.get("guidance_params"),
                metadata=step_data.get("metadata"),
            ))
        
        return cls(
            anchor_id=data["anchor_id"],
            steps=steps,
            editor_name=data["editor_name"],
            config=data.get("config", {}),
            seed=data.get("seed"),
        )


class TrajectoryGenerator:
    """
    Generates edit trajectories using a specified editor.
    
    This class handles the trajectory generation protocol described in Section 5.2.
    
    Args:
        editor: Editing method (callable that takes image and returns edited image)
        coordinate_system: SemanticCoordinateSystem for computing coordinates
        num_steps: Number of edit steps T
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        editor: Callable,
        coordinate_system: SemanticCoordinateSystem,
        num_steps: int = 20,
        seed: Optional[int] = None,
    ):
        self.editor = editor
        self.coordinate_system = coordinate_system
        self.num_steps = num_steps
        self.seed = seed
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def generate(
        self,
        anchor_image: torch.Tensor,
        anchor_id: str,
        guidance_params: Optional[Dict[str, Any]] = None,
    ) -> EditTrajectory:
        """
        Generate a complete edit trajectory from an anchor image.
        
        Args:
            anchor_image: Starting image tensor [C, H, W]
            anchor_id: Identifier for the anchor
            guidance_params: Parameters for the editor
            
        Returns:
            EditTrajectory containing all steps
        """
        guidance_params = guidance_params or {}
        
        # Initialize with anchor
        current_image = anchor_image.clone()
        anchor_coords = self.coordinate_system.compute_coordinates(
            current_image, image_id=anchor_id
        )
        
        steps = [TrajectoryStep(
            step_index=0,
            image=current_image.clone(),
            coordinates=anchor_coords,
            guidance_params=None,
            metadata={"is_anchor": True},
        )]
        
        # Generate trajectory
        for t in range(self.num_steps):
            # Apply editor
            step_params = {**guidance_params, "step": t, "total_steps": self.num_steps}
            edited_image = self.editor(current_image, **step_params)
            
            # Compute coordinates
            coords = self.coordinate_system.compute_coordinates(
                edited_image, image_id=f"{anchor_id}_step_{t+1}"
            )
            
            steps.append(TrajectoryStep(
                step_index=t + 1,
                image=edited_image.clone(),
                coordinates=coords,
                guidance_params=step_params,
            ))
            
            current_image = edited_image
        
        return EditTrajectory(
            anchor_id=anchor_id,
            steps=steps,
            editor_name=getattr(self.editor, "name", "unknown"),
            config=guidance_params,
            seed=self.seed,
        )
    
    def generate_batch(
        self,
        anchor_images: List[torch.Tensor],
        anchor_ids: List[str],
        guidance_params: Optional[Dict[str, Any]] = None,
    ) -> List[EditTrajectory]:
        """Generate trajectories for multiple anchors."""
        return [
            self.generate(img, aid, guidance_params)
            for img, aid in zip(anchor_images, anchor_ids)
        ]


class TrajectoryDataset:
    """
    Dataset of edit trajectories for analysis.
    
    Provides utilities for loading, filtering, and iterating over trajectories.
    """
    
    def __init__(self, trajectories: List[EditTrajectory]):
        self.trajectories = trajectories
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> EditTrajectory:
        return self.trajectories[idx]
    
    def __iter__(self) -> Iterator[EditTrajectory]:
        return iter(self.trajectories)
    
    def filter_by_editor(self, editor_name: str) -> "TrajectoryDataset":
        """Filter trajectories by editor name."""
        filtered = [t for t in self.trajectories if t.editor_name == editor_name]
        return TrajectoryDataset(filtered)
    
    def filter_by_anchor_score(
        self,
        min_score: float = 0.1,
        max_score: float = 0.9,
    ) -> "TrajectoryDataset":
        """Filter trajectories by anchor target score range."""
        filtered = [
            t for t in self.trajectories
            if min_score <= t.anchor.coordinates.calibrated_target <= max_score
        ]
        return TrajectoryDataset(filtered)
    
    def get_all_target_progressions(self) -> np.ndarray:
        """Get target progressions for all trajectories [N, T+1]."""
        return np.stack([t.get_target_progression() for t in self.trajectories])
    
    @classmethod
    def load_from_directory(cls, path: Path) -> "TrajectoryDataset":
        """Load all trajectories from a directory."""
        path = Path(path)
        trajectories = []
        
        for traj_dir in sorted(path.iterdir()):
            if traj_dir.is_dir() and (traj_dir / "trajectory.json").exists():
                trajectories.append(EditTrajectory.load(traj_dir))
        
        return cls(trajectories)
    
    def save_to_directory(self, path: Path, save_images: bool = True):
        """Save all trajectories to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        for i, traj in enumerate(self.trajectories):
            traj_dir = path / f"trajectory_{i:04d}"
            traj.save(traj_dir, save_images=save_images)
