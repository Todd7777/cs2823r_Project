"""
UNet Architecture for Diffusion Models.

Implements the denoising network used in diffusion-based editing.
"""

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal timestep embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        h, _ = self.attention(h, h, h)
        
        h = h.permute(0, 2, 1).view(B, C, H, W)
        
        return x + h


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(ResBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout,
            ))
            if use_attention:
                self.layers.append(AttentionBlock(out_channels))
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = None
        
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_layers: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, 4, stride=2, padding=1
        )
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            # First layer takes concatenated skip connection
            layer_in = in_channels * 2 if i == 0 else out_channels
            self.layers.append(ResBlock(
                layer_in,
                out_channels,
                time_emb_dim,
                dropout,
            ))
            if use_attention:
                self.layers.append(AttentionBlock(out_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        
        for layer in self.layers:
            if isinstance(layer, ResBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        
        return x


class UNet2DModel(nn.Module):
    """
    UNet architecture for image denoising.
    
    Standard UNet with:
    - Sinusoidal time embeddings
    - Residual blocks with time conditioning
    - Optional self-attention
    - Skip connections
    
    Args:
        in_channels: Input image channels
        out_channels: Output image channels
        base_channels: Base channel count
        channel_mults: Channel multipliers for each level
        num_res_blocks: Residual blocks per level
        attention_levels: Which levels use attention
        dropout: Dropout rate
        time_emb_dim: Time embedding dimension
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (2, 3),
        dropout: float = 0.1,
        time_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        channels = [base_channels]
        current_channels = base_channels
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = i in attention_levels
            
            self.down_blocks.append(DownBlock(
                current_channels,
                out_ch,
                time_emb_dim,
                num_res_blocks,
                use_attn,
                dropout,
            ))
            
            current_channels = out_ch
            channels.append(current_channels)
        
        # Middle
        self.mid_block1 = ResBlock(current_channels, current_channels, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(current_channels)
        self.mid_block2 = ResBlock(current_channels, current_channels, time_emb_dim, dropout)
        
        # Upsampling path
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            use_attn = (len(channel_mults) - 1 - i) in attention_levels
            
            self.up_blocks.append(UpBlock(
                current_channels,
                out_ch,
                time_emb_dim,
                num_res_blocks,
                use_attn,
                dropout,
            ))
            
            current_channels = out_ch
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Noisy input [B, C, H, W]
            timestep: Diffusion timestep [B]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_mlp(timestep.float())
        
        # Initial conv
        h = self.conv_in(x)
        
        # Downsampling
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, time_emb)
            skips.append(skip)
        
        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # Upsampling
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, time_emb)
        
        # Output
        h = self.conv_out(h)
        
        return h


def create_unet(
    config: str = "small",
    in_channels: int = 1,
    out_channels: int = 1,
) -> UNet2DModel:
    """
    Factory function to create UNet models.
    
    Args:
        config: "small", "medium", or "large"
        in_channels: Input channels
        out_channels: Output channels
        
    Returns:
        UNet2DModel instance
    """
    configs = {
        "small": {
            "base_channels": 32,
            "channel_mults": (1, 2, 4),
            "num_res_blocks": 1,
            "attention_levels": (2,),
        },
        "medium": {
            "base_channels": 64,
            "channel_mults": (1, 2, 4, 8),
            "num_res_blocks": 2,
            "attention_levels": (2, 3),
        },
        "large": {
            "base_channels": 128,
            "channel_mults": (1, 2, 4, 8),
            "num_res_blocks": 3,
            "attention_levels": (1, 2, 3),
        },
    }
    
    cfg = configs.get(config, configs["medium"])
    
    return UNet2DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        **cfg,
    )
