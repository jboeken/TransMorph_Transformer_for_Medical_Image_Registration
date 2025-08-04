"""
Simple example showing how to use TransMorph API for medical image registration
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from transmorph_api import TransMorph


def main():
    """Simple example of using TransMorph for registration."""
    
    print("=== TransMorph Registration Example ===\n")
    
    # Step 1: Initialize TransMorph with your trained model
    model_path = "experiments/TransMorphBSpline_Liver_mse_1_diffusion_0.01/liver_dsc0.750.pth.tar"
    
    print("Loading TransMorph model...")
    transmorph = TransMorph(
        model_path=model_path,
        model_size=(160, 192, 224),  # Model training size
        device='cuda'  # Use 'cpu' if no GPU available
    )
    
    # Step 2: Load your images
    # Option A: Load from NIfTI files
    # moving_image = nib.load('path/to/moving.nii.gz').get_fdata().astype(np.float32)
    # fixed_image = nib.load('path/to/fixed.nii.gz').get_fdata().astype(np.float32)
    
    # Option B: Create example synthetic images for demonstration
    print("Creating synthetic test images...")
    np.random.seed(42)  # For reproducible results
    
    # Create realistic-looking medical images with some structure
    moving_image = create_synthetic_brain_image(shape=(128, 128, 64))
    fixed_image = create_synthetic_brain_image(shape=(128, 128, 64), shift=(10, 5, 3))
    
    print(f"Moving image shape: {moving_image.shape}")
    print(f"Fixed image shape: {fixed_image.shape}")
    print(f"Moving image range: [{moving_image.min():.3f}, {moving_image.max():.3f}]")
    print(f"Fixed image range: [{fixed_image.min():.3f}, {fixed_image.max():.3f}]")
    
    # Step 3: Perform registration
    print("\nPerforming registration...")
    registered_image = transmorph.register(moving_image, fixed_image)
    
    print(f"Registration completed!")
    print(f"Registered image shape: {registered_image.shape}")
    print(f"Registered image range: [{registered_image.min():.3f}, {registered_image.max():.3f}]")
    
    # Step 4: Visualize results (middle slice)
    visualize_registration(moving_image, fixed_image, registered_image)
    
    # Step 5: Save results (optional)
    save_results = False  # Set to True to save
    if save_results:
        print("\nSaving results...")
        # Save as NIfTI files
        nib.save(nib.Nifti1Image(moving_image, np.eye(4)), 'moving_image.nii.gz')
        nib.save(nib.Nifti1Image(fixed_image, np.eye(4)), 'fixed_image.nii.gz')
        nib.save(nib.Nifti1Image(registered_image, np.eye(4)), 'registered_image.nii.gz')
        print("Results saved!")
    
    # Step 6: Get displacement field (optional)
    print("\nGetting displacement field...")
    registered_with_flow, displacement_field = transmorph.register(
        moving_image, fixed_image, return_flow=True
    )
    
    print(f"Displacement field shape: {displacement_field.shape}")  # [3, H, W, D]
    print(f"Max displacement: {np.abs(displacement_field).max():.2f} voxels")
    
    print("\n=== Example completed successfully! ===")



def create_synthetic_brain_image(shape=(128, 128, 64), shift=(0, 0, 0)):
    """Create a synthetic brain-like image with some realistic structures."""
    h, w, d = shape
    
    # Create coordinate grids
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    z = np.linspace(-1, 1, d)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Apply shift
    X += shift[0] / w * 2
    Y += shift[1] / h * 2
    Z += shift[2] / d * 2
    
    # Create brain-like structures
    # Outer brain boundary (ellipsoid)
    brain_mask = (X**2/0.8 + Y**2/0.6 + Z**2/0.7) < 1
    
    # Inner structures
    # Ventricles (darker regions)
    ventricle1 = ((X-0.15)**2/0.1 + (Y-0.1)**2/0.05 + Z**2/0.15) < 1
    ventricle2 = ((X+0.15)**2/0.1 + (Y-0.1)**2/0.05 + Z**2/0.15) < 1
    
    # Gray matter (medium intensity)
    gray_matter = brain_mask & ~(ventricle1 | ventricle2)
    
    # White matter (higher intensity, inner region)
    white_matter = (X**2/0.4 + Y**2/0.3 + Z**2/0.35) < 1
    white_matter = white_matter & ~(ventricle1 | ventricle2)
    
    # Create the image
    image = np.zeros(shape, dtype=np.float32)
    image[gray_matter] = 0.3 + 0.1 * np.random.rand(np.sum(gray_matter))
    image[white_matter] = 0.6 + 0.1 * np.random.rand(np.sum(white_matter))
    image[ventricle1 | ventricle2] = 0.05 + 0.05 * np.random.rand(np.sum(ventricle1 | ventricle2))
    
    # Add some noise
    image += 0.02 * np.random.randn(*shape)
    image = np.clip(image, 0, 1)
    
    return image


def visualize_registration(moving, fixed, registered):
    """Visualize registration results."""
    # Select middle slices
    mid_slice = moving.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: Axial view (XY plane)
    axes[0, 0].imshow(moving[:, :, mid_slice], cmap='gray')
    axes[0, 0].set_title('Moving Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fixed[:, :, mid_slice], cmap='gray')
    axes[0, 1].set_title('Fixed Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(registered[:, :, mid_slice], cmap='gray')
    axes[0, 2].set_title('Registered Image')
    axes[0, 2].axis('off')
    
    # Bottom row: Difference images
    diff_before = np.abs(moving[:, :, mid_slice] - fixed[:, :, mid_slice])
    diff_after = np.abs(registered[:, :, mid_slice] - fixed[:, :, mid_slice])
    
    axes[1, 0].imshow(diff_before, cmap='hot')
    axes[1, 0].set_title(f'Difference Before\n(Mean: {diff_before.mean():.4f})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(diff_after, cmap='hot')
    axes[1, 1].set_title(f'Difference After\n(Mean: {diff_after.mean():.4f})')
    axes[1, 1].axis('off')
    
    # Overlay comparison
    axes[1, 2].imshow(fixed[:, :, mid_slice], cmap='gray', alpha=0.7)
    axes[1, 2].imshow(registered[:, :, mid_slice], cmap='Reds', alpha=0.3)
    axes[1, 2].set_title('Overlay (Fixed=Gray, Registered=Red)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('registration_example.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Mean absolute difference before registration: {diff_before.mean():.4f}")
    print(f"Mean absolute difference after registration: {diff_after.mean():.4f}")
    print(f"Improvement: {((diff_before.mean() - diff_after.mean()) / diff_before.mean() * 100):.1f}%")


def real_data_example():
    """Example with real medical data (modify paths as needed)."""
    
    # Initialize TransMorph
    transmorph = TransMorph(model_path="path/to/your/model.pth.tar")
    
    # Load real NIfTI images
    moving_nii = nib.load("path/to/moving_scan.nii.gz")
    fixed_nii = nib.load("path/to/fixed_scan.nii.gz")
    
    moving_image = moving_nii.get_fdata().astype(np.float32)
    fixed_image = fixed_nii.get_fdata().astype(np.float32)
    
    print(f"Loaded moving image: {moving_image.shape}")
    print(f"Loaded fixed image: {fixed_image.shape}")
    
    # Perform registration
    registered_image = transmorph.register(moving_image, fixed_image)
    
    # Save result with original NIfTI metadata
    registered_nii = nib.Nifti1Image(
        registered_image, 
        moving_nii.affine, 
        moving_nii.header
    )
    nib.save(registered_nii, "registered_output.nii.gz")
    
    print("Registration completed and saved!")


def batch_processing_example():
    """Example of processing multiple image pairs."""
    
    # Initialize TransMorph
    transmorph = TransMorph(model_path="path/to/your/model.pth.tar")
    
    # Create multiple image pairs (in practice, load from files)
    moving_images = [
        create_synthetic_brain_image(shift=(5, 0, 0)),
        create_synthetic_brain_image(shift=(0, 8, 0)), 
        create_synthetic_brain_image(shift=(3, 3, 2))
    ]
    
    fixed_images = [
        create_synthetic_brain_image(),
        create_synthetic_brain_image(),
        create_synthetic_brain_image()
    ]
    
    print(f"Processing {len(moving_images)} image pairs...")
    
    # Process all pairs
    registered_images = transmorph.register_batch(moving_images, fixed_images)
    
    print("Batch processing completed!")
    
    # Save results
    for i, reg_img in enumerate(registered_images):
        nib.save(nib.Nifti1Image(reg_img, np.eye(4)), f'registered_pair_{i:02d}.nii.gz')


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease make sure to:")
        print("1. Update the model_path to point to your trained TransMorph model")
        print("2. Install required packages: torch, numpy, nibabel, matplotlib, scikit-image")
        print("3. Make sure you have CUDA available if using GPU")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please check your installation and model file.")