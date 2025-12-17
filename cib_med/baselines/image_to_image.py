"""
Image-to-Image Translation Baselines.

This module implements paired and pseudo-paired image translation methods
as baselines for directional medical image editing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEditor(ABC):
    """Abstract base class for image editors."""
    
    name: str = "base"
    
    @abstractmethod
    def edit(
        self,
        image: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply editing to an image.
        
        Args:
            image: Input image tensor [C, H, W] or [B, C, H, W]
            **kwargs: Editor-specific parameters
            
        Returns:
            Edited image tensor
        """
        pass
    
    def __call__(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.edit(image, **kwargs)


class Pix2PixEditor(BaseEditor):
    """
    Pix2Pix-style paired image translation.
    
    Trained to map lower-effusion images to higher-effusion images
    using paired or pseudo-paired data.
    
    Args:
        generator: Trained generator network
        device: Compute device
        normalize_input: Whether to normalize input to [-1, 1]
    """
    
    name = "pix2pix"
    
    def __init__(
        self,
        generator: Optional[nn.Module] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalize_input: bool = True,
    ):
        self.device = device
        self.normalize_input = normalize_input
        
        if generator is not None:
            self.generator = generator.to(device)
        elif weights_path is not None:
            self.generator = self._load_generator(weights_path)
        else:
            # Create default UNet generator
            self.generator = UNetGenerator().to(device)
        
        self.generator.eval()
    
    def _load_generator(self, weights_path: str) -> nn.Module:
        """Load generator from weights file."""
        generator = UNetGenerator()
        state_dict = torch.load(weights_path, map_location=self.device)
        generator.load_state_dict(state_dict)
        return generator.to(self.device)
    
    def edit(
        self,
        image: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Apply Pix2Pix translation."""
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Normalize if needed
        if self.normalize_input:
            image = image * 2.0 - 1.0  # [0,1] -> [-1,1]
        
        with torch.no_grad():
            output = self.generator(image)
        
        # Denormalize
        if self.normalize_input:
            output = (output + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        output = output.clamp(0, 1)
        
        return output.squeeze(0) if output.shape[0] == 1 else output


class CycleGANEditor(BaseEditor):
    """
    CycleGAN-style unpaired image translation.
    
    Learns to translate between low-effusion and high-effusion
    domains without paired training data.
    
    Args:
        generator_A2B: Generator from domain A to B (low to high effusion)
        generator_B2A: Generator from domain B to A (optional, for cycle)
        device: Compute device
    """
    
    name = "cyclegan"
    
    def __init__(
        self,
        generator_A2B: Optional[nn.Module] = None,
        generator_B2A: Optional[nn.Module] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        
        if generator_A2B is not None:
            self.generator = generator_A2B.to(device)
        elif weights_path is not None:
            self.generator = self._load_generator(weights_path)
        else:
            self.generator = ResNetGenerator().to(device)
        
        self.generator_B2A = generator_B2A
        if self.generator_B2A is not None:
            self.generator_B2A = self.generator_B2A.to(device)
        
        self.generator.eval()
    
    def _load_generator(self, weights_path: str) -> nn.Module:
        """Load generator from weights file."""
        generator = ResNetGenerator()
        state_dict = torch.load(weights_path, map_location=self.device)
        generator.load_state_dict(state_dict)
        return generator.to(self.device)
    
    def edit(
        self,
        image: torch.Tensor,
        direction: str = "forward",  # "forward" or "backward"
        **kwargs
    ) -> torch.Tensor:
        """Apply CycleGAN translation."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        
        with torch.no_grad():
            if direction == "forward":
                output = self.generator(image)
            elif direction == "backward" and self.generator_B2A is not None:
                output = self.generator_B2A(image)
            else:
                output = self.generator(image)
        
        # Denormalize
        output = (output + 1.0) / 2.0
        output = output.clamp(0, 1)
        
        return output.squeeze(0) if output.shape[0] == 1 else output


class UNetGenerator(nn.Module):
    """
    UNet-style generator for Pix2Pix.
    
    Encoder-decoder architecture with skip connections.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        num_downs: int = 7,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Build UNet
        # Innermost layer
        unet_block = UNetSkipBlock(
            base_filters * 8, base_filters * 8,
            innermost=True
        )
        
        # Intermediate layers
        for i in range(num_downs - 5):
            unet_block = UNetSkipBlock(
                base_filters * 8, base_filters * 8,
                submodule=unet_block, use_dropout=True
            )
        
        # Outer layers with decreasing filters
        unet_block = UNetSkipBlock(base_filters * 4, base_filters * 8, submodule=unet_block)
        unet_block = UNetSkipBlock(base_filters * 2, base_filters * 4, submodule=unet_block)
        unet_block = UNetSkipBlock(base_filters, base_filters * 2, submodule=unet_block)
        
        # Outermost layer
        self.model = UNetSkipBlock(
            out_channels, base_filters,
            input_nc=in_channels, submodule=unet_block, outermost=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UNetSkipBlock(nn.Module):
    """UNet skip connection block."""
    
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int = None,
        submodule: nn.Module = None,
        outermost: bool = False,
        innermost: bool = False,
        use_dropout: bool = False,
    ):
        super().__init__()
        
        self.outermost = outermost
        
        if input_nc is None:
            input_nc = outer_nc
        
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)
        
        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class ResNetGenerator(nn.Module):
    """
    ResNet-style generator for CycleGAN.
    
    Uses residual blocks for better gradient flow.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        n_residual: int = 9,
    ):
        super().__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(True),
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(base_filters * mult, base_filters * mult * 2,
                         kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(base_filters * mult * 2),
                nn.ReLU(True),
            ]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_residual):
            model += [ResidualBlock(base_filters * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(base_filters * mult, base_filters * mult // 2,
                                  kernel_size=3, stride=2, padding=1,
                                  output_padding=1, bias=False),
                nn.InstanceNorm2d(base_filters * mult // 2),
                nn.ReLU(True),
            ]
        
        # Output
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    """Residual block with instance normalization."""
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
