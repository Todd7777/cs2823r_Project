"""Baseline editing methods for comparison."""

from cib_med.baselines.image_to_image import Pix2PixEditor, CycleGANEditor
from cib_med.baselines.prompt_based import InstructPix2PixEditor, TextGuidedEditor

__all__ = [
    "Pix2PixEditor",
    "CycleGANEditor",
    "InstructPix2PixEditor",
    "TextGuidedEditor",
]
