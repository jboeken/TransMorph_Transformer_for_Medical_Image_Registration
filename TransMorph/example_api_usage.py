"""
Example usage of TransMorph API for programmatic registration
"""

import numpy as np
import nibabel as nib
from transmorph_api import TransMorph


def load_nifti_image(filepath):
    """Load a NIfTI image and return as numpy array."""
    nii = nib.load(filepath)
    return nii.get_fdata().astype(np.float32)


def save_nifti_image(image, filepath, reference_nii=None):
    """Save numpy array as NIfTI image."""
    if reference_nii is not None:
        # Use reference image's affine and header
        nii = nib.Nifti1Image(image, reference_nii.affine, reference_nii.header)
    else:
        # Create with identity affine
        nii = nib.Nifti1Image(image, np.eye(4))
    nib.save(nii, filepath)


def main():
    # Example 1: Basic usage with file paths
    print("=== Example 1: Basic Registration ===")
    
    # Initialize TransMorph with trained model
    model_path = "experiments/TransMorphBSpline_Liver_mse_1_diffusion_0.01/liver_dsc0.750.pth.tar"
    transmorph = TransMorph(model_path=model_path)
    
    # Print model information
    info = transmorph.get_model_info()
    print(f"Model parameters: {info['parameters']:,}")
    print(f"Device: {info['device']}")
    
    # Load images (replace with your image paths)
    moving_path = "data/moving_image.nii.gz"
    fixed_path = "data/fixed_image.nii.gz"
    
    # Note: Uncomment and modify these lines with your actual image paths
    # moving_image = load_nifti_image(moving_path)
    # fixed_image = load_nifti_image(fixed_path)
    
    # For demonstration, create synthetic images
    print("Creating synthetic test images...")
    moving_image = np.random.rand(128, 128, 64).astype(np.float32)
    fixed_image = np.random.rand(128, 128, 64).astype(np.float32)
    
    print(f"Moving image shape: {moving_image.shape}")
    print(f"Fixed image shape: {fixed_image.shape}")
    
    # Perform registration
    print("Performing registration...")
    registered_image = transmorph.register(moving_image, fixed_image)
    
    print(f"Registered image shape: {registered_image.shape}")
    print("Registration completed!")
    
    # Save result (uncomment to save)
    # save_nifti_image(registered_image, "results/registered_image.nii.gz")
    
    
    # Example 2: Registration with displacement field
    print("\n=== Example 2: Registration with Displacement Field ===")
    
    registered_image, displacement_field = transmorph.register(
        moving_image, fixed_image, return_flow=True
    )
    
    print(f"Displacement field shape: {displacement_field.shape}")  # Should be [3, H, W, D]
    print("Registration with displacement field completed!")
    
    
    # Example 3: Batch processing
    print("\n=== Example 3: Batch Processing ===")
    
    # Create multiple image pairs
    moving_images = [np.random.rand(64, 64, 32).astype(np.float32) for _ in range(3)]
    fixed_images = [np.random.rand(64, 64, 32).astype(np.float32) for _ in range(3)]
    
    # Register all pairs
    registered_images = transmorph.register_batch(moving_images, fixed_images)
    
    print(f"Processed {len(registered_images)} image pairs")
    for i, reg_img in enumerate(registered_images):
        print(f"  Pair {i+1}: {reg_img.shape}")
    
    
    # Example 4: Working with different image sizes
    print("\n=== Example 4: Different Image Sizes ===")
    
    # The API automatically handles different input sizes
    small_moving = np.random.rand(64, 80, 48).astype(np.float32)
    small_fixed = np.random.rand(64, 80, 48).astype(np.float32)
    
    large_moving = np.random.rand(256, 256, 128).astype(np.float32)
    large_fixed = np.random.rand(256, 256, 128).astype(np.float32)
    
    print("Registering small images...")
    small_registered = transmorph.register(small_moving, small_fixed)
    print(f"Small: {small_moving.shape} -> {small_registered.shape}")
    
    print("Registering large images...")
    large_registered = transmorph.register(large_moving, large_fixed)
    print(f"Large: {large_moving.shape} -> {large_registered.shape}")
    
    print("All examples completed!")


def example_with_real_data():
    """Example using real medical images (uncomment and modify paths)."""
    
    # Initialize model
    model_path = "path/to/your/model.pth.tar"
    transmorph = TransMorph(model_path=model_path)
    
    # Load real images
    moving_path = "path/to/moving_image.nii.gz"
    fixed_path = "path/to/fixed_image.nii.gz"
    
    moving_nii = nib.load(moving_path)
    fixed_nii = nib.load(fixed_path)
    
    moving_image = moving_nii.get_fdata().astype(np.float32)
    fixed_image = fixed_nii.get_fdata().astype(np.float32)
    
    # Perform registration
    registered_image = transmorph.register(moving_image, fixed_image)
    
    # Save with original NIfTI metadata
    save_nifti_image(registered_image, "registered_output.nii.gz", reference_nii=moving_nii)
    
    print("Real data registration completed!")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        print("Please update the model_path in the script to point to your trained model.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have all required dependencies and a valid model file.")