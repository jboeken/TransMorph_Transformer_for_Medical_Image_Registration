"""
TransMorph API for programmatic registration
Simple interface to use TransMorph models in other Python scripts
"""

import os
import torch
import numpy as np
from skimage.transform import resize
import models.transformation as transformation
from models.TransMorph_bspl import CONFIGS as CONFIGS_TM
import models.TransMorph_bspl as TransMorph_bspl


class TransMorph:
    """
    TransMorph registration interface for programmatic use.
    
    Usage:
        # Initialize with model
        transmorph = TransMorph(model_path='path/to/model.pth.tar')
        
        # Register images
        registered_image = transmorph.register(moving_image, fixed_image)
    """
    
    def __init__(self, model_path, model_size=(160, 192, 224), device='cuda'):
        """
        Initialize TransMorph with a trained model.
        
        Args:
            model_path (str): Path to trained model checkpoint (.pth.tar)
            model_size (tuple): Model input size (H, W, D). Default: (160, 192, 224)
            device (str): Device to run on ('cuda' or 'cpu'). Default: 'cuda'
        """
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the TransMorph model from checkpoint."""
        # Load model configuration
        config = CONFIGS_TM['TransMorphBSpline']
        config.img_size = self.model_size
        self.model = TransMorph_bspl.TranMorphBSplineNet(config)
        
        # Load trained weights
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f'Loaded TransMorph model from: {self.model_path}')
                if 'best_dsc' in checkpoint:
                    print(f'Model training DSC: {checkpoint["best_dsc"]:.4f}')
            else:
                raise ValueError("Invalid checkpoint format! Expected 'state_dict' key.")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Move to device and set to eval mode
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model.cuda()
        else:
            self.device = 'cpu'
            self.model.cpu()
            
        self.model.eval()
        print(f'Model loaded on device: {self.device}')
        
    def _preprocess_image(self, image):
        """
        Preprocess image for model input.
        
        Args:
            image (np.ndarray): Input image of shape (H, W, D) or (1, H, W, D)
            
        Returns:
            torch.Tensor: Preprocessed image tensor [1, 1, H, W, D]
        """
        # Handle different input shapes
        if len(image.shape) == 4 and image.shape[0] == 1:
            image = image[0]  # Remove batch dimension if present
        elif len(image.shape) != 3:
            raise ValueError(f"Expected image shape (H, W, D) or (1, H, W, D), got {image.shape}")
        
        # Store original size for later use
        original_size = image.shape
        
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        
        # Resize to model input size if needed
        if original_size != tuple(self.model_size):
            image = resize(image, self.model_size, anti_aliasing=False, order=1, preserve_range=True)
        
        # Add batch and channel dimensions: [1, 1, H, W, D]
        image_tensor = torch.from_numpy(image[None, None, ...]).float()
        
        # Move to device
        if self.device == 'cuda':
            image_tensor = image_tensor.cuda()
            
        return image_tensor, original_size
    
    def _resize_flow_to_original(self, flow, original_size):
        """
        Resize flow field back to original image size.
        
        Args:
            flow (torch.Tensor): Flow field [1, 3, H, W, D]
            original_size (tuple): Target size (H, W, D)
            
        Returns:
            torch.Tensor: Resized flow field
        """
        if flow.shape[2:] == original_size:
            return flow
            
        flow_np = flow[0].cpu().numpy()  # [3, H, W, D]
        flow_resized = np.zeros((3,) + original_size, dtype=np.float32)
        scale_factors = [original_size[i] / flow_np.shape[i+1] for i in range(3)]
        
        # Resize each flow component and scale appropriately
        for i in range(3):
            flow_resized[i] = resize(flow_np[i], original_size, anti_aliasing=False, order=1, preserve_range=True)
            flow_resized[i] *= scale_factors[i]
        
        # Convert back to tensor
        flow_tensor = torch.from_numpy(flow_resized[None, ...]).float()
        if self.device == 'cuda':
            flow_tensor = flow_tensor.cuda()
            
        return flow_tensor
    
    def register(self, moving_image, fixed_image, return_flow=False):
        """
        Register moving image to fixed image.
        
        Args:
            moving_image (np.ndarray): Moving image of shape (H, W, D)
            fixed_image (np.ndarray): Fixed image of shape (H, W, D)  
            return_flow (bool): Whether to return displacement field. Default: False
            
        Returns:
            np.ndarray: Registered moving image of shape (H, W, D)
            If return_flow=True: (registered_image, displacement_field)
        """
        with torch.no_grad():
            # Preprocess images
            moving_tensor, moving_original_size = self._preprocess_image(moving_image)
            fixed_tensor, fixed_original_size = self._preprocess_image(fixed_image)
            
            # Check that both images have same original size
            if moving_original_size != fixed_original_size:
                print(f"Warning: Moving image size {moving_original_size} != Fixed image size {fixed_original_size}")
                print(f"Using moving image size as target: {moving_original_size}")
            
            target_size = moving_original_size
            
            # Forward pass through model
            registered_model, _, displacement_model = self.model((moving_tensor, fixed_tensor))
            
            # Resize displacement field back to original resolution
            displacement_original = self._resize_flow_to_original(displacement_model, target_size)
            
            # Apply displacement to original resolution moving image
            moving_original_tensor = torch.from_numpy(moving_image[None, None, ...]).float()
            if self.device == 'cuda':
                moving_original_tensor = moving_original_tensor.cuda()
                
            registered_original = transformation.warp(
                moving_original_tensor, 
                displacement_original, 
                interp_mode='bilinear'
            )
            
            # Convert back to numpy
            registered_image = registered_original[0, 0].cpu().numpy()
            
            if return_flow:
                displacement_field = displacement_original[0].cpu().numpy()  # [3, H, W, D]
                return registered_image, displacement_field
            else:
                return registered_image
    
    def register_batch(self, moving_images, fixed_images, return_flows=False):
        """
        Register a batch of image pairs.
        
        Args:
            moving_images (list): List of moving images, each of shape (H, W, D)
            fixed_images (list): List of fixed images, each of shape (H, W, D)
            return_flows (bool): Whether to return displacement fields. Default: False
            
        Returns:
            list: List of registered images
            If return_flows=True: (list_of_registered_images, list_of_displacement_fields)
        """
        if len(moving_images) != len(fixed_images):
            raise ValueError("Number of moving and fixed images must match")
            
        registered_images = []
        displacement_fields = [] if return_flows else None
        
        for i, (moving, fixed) in enumerate(zip(moving_images, fixed_images)):
            print(f"Processing pair {i+1}/{len(moving_images)}")
            
            if return_flows:
                registered, flow = self.register(moving, fixed, return_flow=True)
                registered_images.append(registered)
                displacement_fields.append(flow)
            else:
                registered = self.register(moving, fixed, return_flow=False)
                registered_images.append(registered)
        
        if return_flows:
            return registered_images, displacement_fields
        else:
            return registered_images
    
    def get_model_info(self):
        """Get information about the loaded model."""
        return {
            'model_path': self.model_path,
            'model_size': self.model_size,
            'device': self.device,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }