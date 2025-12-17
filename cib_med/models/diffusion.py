"""
Diffusion Model Implementation for Medical Image Editing.

Provides the core diffusion sampling with classifier guidance support.
"""

from typing import Dict, Any, Optional, Callable, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class DDPMScheduler:
    """
    DDPM noise scheduler for diffusion models.
    
    Implements the forward and reverse diffusion processes.
    
    Args:
        num_train_timesteps: Number of diffusion timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Schedule type ("linear", "cosine", "quadratic")
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
    ):
        self.num_train_timesteps = num_train_timesteps
        
        # Compute betas
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            steps = torch.arange(num_train_timesteps + 1) / num_train_timesteps
            alphas_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            self.betas = torch.clamp(betas, 0.0001, 0.9999)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Compute derived quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to the diffusion schedule.
        
        q(x_t | x_0) = N(sqrt(α̅_t) x_0, (1 - α̅_t) I)
        """
        device = original.device
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(device)
        
        # Reshape for broadcasting
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Perform one reverse diffusion step.
        
        p(x_{t-1} | x_t) ∝ N(μ_θ(x_t, t), σ_t² I)
        """
        device = sample.device
        t = timestep
        
        # Get coefficients
        beta_t = self.betas[t].to(device)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].to(device)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].to(device)
        
        # Compute predicted x_0
        pred_original = sqrt_recip_alpha_t * (
            sample - beta_t / sqrt_one_minus_alpha_cumprod_t * model_output
        )
        
        # Clip to valid range
        pred_original = pred_original.clamp(-1, 1)
        
        # Compute posterior mean
        alpha_t = self.alphas[t].to(device)
        alpha_cumprod_t = self.alphas_cumprod[t].to(device)
        alpha_cumprod_prev_t = self.alphas_cumprod_prev[t].to(device)
        
        posterior_mean = (
            torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1 - alpha_cumprod_t) * pred_original +
            torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * sample
        )
        
        # Add noise (except for t=0)
        if t > 0:
            variance = self.posterior_variance[t].to(device)
            noise = torch.randn_like(sample, generator=generator)
            sample = posterior_mean + torch.sqrt(variance) * noise
        else:
            sample = posterior_mean
        
        return sample


class DiffusionEditor:
    """
    Diffusion-based image editor with classifier guidance.
    
    Implements the editing pipeline for CIB-Med-1:
    1. Encode image to latent (if using latent diffusion)
    2. Add noise up to specified level
    3. Denoise with classifier guidance
    4. Decode back to image space
    
    Args:
        model: Denoising model (UNet)
        scheduler: Noise scheduler
        guidance_method: Guidance method for steering
        device: Compute device
    """
    
    name = "diffusion_editor"
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        guidance_method: Optional[Callable] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        noise_level: float = 0.5,  # Fraction of total timesteps to noise
    ):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.guidance_method = guidance_method
        self.device = device
        self.noise_level = noise_level
        
        self.model.eval()
    
    def edit(
        self,
        image: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        return_intermediate: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply diffusion editing to an image.
        
        Args:
            image: Input image [B, C, H, W] or [C, H, W]
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier guidance scale
            return_intermediate: Whether to return intermediate steps
            **kwargs: Additional arguments for guidance
            
        Returns:
            Edited image tensor
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        B = image.shape[0]
        
        # Normalize to [-1, 1] if needed
        if image.min() >= 0:
            image = image * 2.0 - 1.0
        
        # Determine starting timestep based on noise level
        start_timestep = int(self.scheduler.num_train_timesteps * self.noise_level)
        timesteps = list(range(start_timestep, 0, -max(1, start_timestep // num_inference_steps)))
        
        # Add noise to starting point
        noise = torch.randn_like(image)
        noisy_image = self.scheduler.add_noise(
            image, noise, torch.tensor([start_timestep])
        )
        
        sample = noisy_image
        intermediates = [sample.clone()] if return_intermediate else None
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            t_tensor = torch.tensor([t] * B, device=self.device)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(sample, t_tensor)
            
            # Apply classifier guidance
            if self.guidance_method is not None:
                guidance = self.guidance_method(
                    sample, t,
                    total_steps=len(timesteps),
                    step=i,
                    **kwargs
                )
                # Scale guidance
                guidance = guidance_scale * guidance
                # Modify noise prediction
                noise_pred = noise_pred - guidance
            
            # Reverse step
            sample = self.scheduler.step(noise_pred, t, sample)
            
            if return_intermediate:
                intermediates.append(sample.clone())
        
        # Denormalize
        sample = (sample + 1.0) / 2.0
        sample = sample.clamp(0, 1)
        
        if return_intermediate:
            return sample, intermediates
        return sample
    
    def __call__(
        self,
        image: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Convenience call method."""
        return self.edit(image, **kwargs)
    
    def generate_trajectory(
        self,
        anchor_image: torch.Tensor,
        num_steps: int = 20,
        step_noise_level: float = 0.1,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate an edit trajectory by iterative editing.
        
        This is the main method for CIB-Med-1 trajectory generation.
        
        Args:
            anchor_image: Starting image
            num_steps: Number of trajectory steps
            step_noise_level: Noise level per step
            **kwargs: Additional guidance arguments
            
        Returns:
            List of images forming the trajectory
        """
        current = anchor_image.clone()
        trajectory = [current.clone()]
        
        original_noise_level = self.noise_level
        self.noise_level = step_noise_level
        
        for step in range(num_steps):
            # Apply editing step
            edited = self.edit(
                current,
                step=step,
                total_steps=num_steps,
                **kwargs
            )
            trajectory.append(edited.clone())
            current = edited
        
        self.noise_level = original_noise_level
        return trajectory


class SDEditEditor(DiffusionEditor):
    """
    SDEdit-style editor using stochastic differential equations.
    
    Adds noise and denoises in a single pass for image editing.
    """
    
    name = "sdedit"
    
    def edit(
        self,
        image: torch.Tensor,
        strength: float = 0.5,
        num_inference_steps: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply SDEdit-style editing.
        
        Args:
            image: Input image
            strength: Edit strength (higher = more change)
            num_inference_steps: Denoising steps
            
        Returns:
            Edited image
        """
        self.noise_level = strength
        return super().edit(image, num_inference_steps=num_inference_steps, **kwargs)
