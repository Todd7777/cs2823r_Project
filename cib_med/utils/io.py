"""I/O utilities for saving and loading results."""

from typing import Dict, Any, Optional
from pathlib import Path
import json
import pickle
import torch


def save_results(
    results: Dict[str, Any],
    path: Path,
    format: str = "json",
):
    """Save results to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            return obj
        
        serializable = json.loads(
            json.dumps(results, default=convert)
        )
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)
    elif format == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_results(path: Path, format: str = "json") -> Dict[str, Any]:
    """Load results from file."""
    path = Path(path)
    
    if format == "json":
        with open(path) as f:
            return json.load(f)
    elif format == "pickle":
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, path)
    
    if is_best:
        best_path = path.parent / 'best_checkpoint.pt'
        torch.save(state, best_path)


def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load model checkpoint."""
    return torch.load(path, map_location=map_location)
