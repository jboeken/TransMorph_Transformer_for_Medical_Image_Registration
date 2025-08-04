#!/usr/bin/env python3
"""
Test script to validate the liver MRI dataset structure and loading.
"""
import os
import sys
import argparse
from torchvision import transforms
from data.liver_datasets import LiverMRIDataset, LiverMRIValidationDataset
from data import trans
import numpy as np

def test_data_loading(data_root):
    """Test if the dataset can be loaded properly"""
    print(f"Testing data loading from: {data_root}")
    print("=" * 50)
    
    # Check directory structure
    mri_dir = os.path.join(data_root, 'MRI')
    labels_dir = os.path.join(data_root, 'labels')
    
    if not os.path.exists(mri_dir):
        print(f"ERROR: MRI directory not found: {mri_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return False
    
    print(f"✓ Found MRI directory: {mri_dir}")
    print(f"✓ Found Labels directory: {labels_dir}")
    
    # Test transforms
    train_transforms = transforms.Compose([
        trans.NumpyType((np.float32, np.float32)),
    ])
    
    val_transforms = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])
    
    try:
        # Test training dataset
        print("\nTesting Training Dataset:")
        train_dataset = LiverMRIDataset(
            data_root=data_root, 
            transforms=train_transforms, 
            split='train'
        )
        
        print(f"Training dataset size: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            # Test loading first sample
            sample = train_dataset[0]
            moving, fixed = sample
            print(f"Sample shapes - Moving: {moving.shape}, Fixed: {fixed.shape}")
            print(f"Data types - Moving: {moving.dtype}, Fixed: {fixed.dtype}")
            print(f"Value ranges - Moving: [{moving.min():.3f}, {moving.max():.3f}], Fixed: [{fixed.min():.3f}, {fixed.max():.3f}]")
        
        # Test validation dataset
        print("\nTesting Validation Dataset:")
        val_dataset = LiverMRIValidationDataset(
            data_root=data_root, 
            transforms=val_transforms
        )
        
        print(f"Validation dataset size: {len(val_dataset)}")
        
        if len(val_dataset) > 0:
            # Test loading first validation sample
            val_sample = val_dataset[0]
            moving, fixed, moving_seg, fixed_seg = val_sample
            print(f"Val sample shapes - Moving: {moving.shape}, Fixed: {fixed.shape}")
            print(f"Val segmentation shapes - Moving seg: {moving_seg.shape}, Fixed seg: {fixed_seg.shape}")
            print(f"Segmentation value ranges - Moving seg: [{moving_seg.min()}, {moving_seg.max()}], Fixed seg: [{fixed_seg.min()}, {fixed_seg.max()}]")
            
            # Analyze unique labels
            unique_labels = sorted(np.unique(moving_seg.numpy()))
            print(f"Unique segmentation labels: {unique_labels}")
            
            # Count voxels per label
            from collections import Counter
            label_counts = Counter(moving_seg.numpy().flatten())
            total_voxels = moving_seg.numel()
            
            print("Label distribution:")
            for label in unique_labels:
                count = label_counts[label]
                percentage = (count / total_voxels) * 100
                print(f"  Label {int(label):2d}: {count:8d} voxels ({percentage:5.1f}%)")
            
            print(f"\nRecommended training parameters:")
            print(f"  --num_classes {len(unique_labels)}")
            
            # Suggest class names based on your liver anatomy labels
            class_names = []
            for label in unique_labels:
                label_int = int(label)
                if label_int == 0:
                    class_names.append("background")
                elif label_int == 1:
                    class_names.append("liver_tissue")
                elif label_int == 2:
                    class_names.append("portal_vein")
                elif label_int == 3:
                    class_names.append("arterial_vessels")
                elif label_int == 4:
                    class_names.append("lesion")
                elif label_int == 6:
                    class_names.append("abdominal_vena_cava")
                elif label_int == 7:
                    class_names.append("thoracal_vena_cava")
                else:
                    class_names.append(f"class_{label_int}")
            
            print(f"  --class_names {' '.join(class_names)}")
        
        print("\n✓ Data loading test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Test liver MRI dataset loading')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing MRI/ and labels/ folders')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root directory does not exist: {args.data_root}")
        sys.exit(1)
    
    success = test_data_loading(args.data_root)
    
    if success:
        print("\n🎉 All tests passed! Your data structure is compatible.")
        print("\nYou can now run the training script with:")
        print(f"python train_TransMorph_bspl_liver.py --data_root {args.data_root} --pretrained_weights /path/to/pretrained/weights.pth")
    else:
        print("\n❌ Tests failed. Please check your data structure.")
        sys.exit(1)

if __name__ == '__main__':
    main()