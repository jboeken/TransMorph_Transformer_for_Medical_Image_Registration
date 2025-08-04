#!/usr/bin/env python3
"""
Analyze liver MRI segmentation labels to understand the class distribution
and determine the optimal number of classes for training.
"""
import os
import glob
import argparse
import nibabel as nib
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_labels(data_root):
    """Analyze all segmentation labels in the dataset"""
    labels_dir = os.path.join(data_root, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return
    
    # Find all label files
    label_files = glob.glob(os.path.join(labels_dir, '*.nii.gz'))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("No label files found!")
        return
    
    all_labels = []
    label_stats = {}
    
    print("\nAnalyzing label files...")
    for i, label_file in enumerate(label_files[:10]):  # Analyze first 10 files
        try:
            # Load label file
            nii = nib.load(label_file)
            seg = nii.get_fdata()
            
            # Get unique labels
            unique_labels = np.unique(seg)
            unique_labels = unique_labels[unique_labels >= 0]  # Remove negative values if any
            
            # Count voxels per label
            label_counts = Counter(seg.flatten())
            
            basename = os.path.basename(label_file)
            print(f"{i+1:2d}. {basename}")
            print(f"    Shape: {seg.shape}")
            print(f"    Labels: {sorted(unique_labels.astype(int))}")
            
            # Calculate percentages
            total_voxels = seg.size
            for label in sorted(unique_labels.astype(int)):
                count = label_counts[label]
                percentage = (count / total_voxels) * 100
                print(f"    Label {int(label):2d}: {count:8d} voxels ({percentage:5.1f}%)")
            
            all_labels.extend(unique_labels)
            label_stats[basename] = {
                'unique_labels': unique_labels,
                'shape': seg.shape,
                'label_counts': dict(label_counts)
            }
            
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    
    all_unique = sorted(set(all_labels))
    print(f"All unique labels across dataset: {[int(x) for x in all_unique]}")
    print(f"Number of classes (including background): {len(all_unique)}")
    
    # Suggest class mapping
    print(f"\nSuggested class configuration:")
    print(f"  --num_classes {len(all_unique)}")
    
    class_names = []
    for label in all_unique:
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
    
    # Create visualization
    try:
        plt.figure(figsize=(10, 6))
        
        # Count frequency of each label across all files
        label_frequency = Counter(all_labels)
        labels = sorted(label_frequency.keys())
        frequencies = [label_frequency[label] for label in labels]
        
        plt.bar([int(x) for x in labels], frequencies)
        plt.xlabel('Label Value')
        plt.ylabel('Frequency (across all analyzed files)')
        plt.title('Distribution of Segmentation Labels')
        plt.xticks([int(x) for x in labels])
        
        # Add text annotations
        for i, (label, freq) in enumerate(zip(labels, frequencies)):
            plt.text(int(label), freq + 0.1, str(freq), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_root, 'label_distribution.png'), dpi=150, bbox_inches='tight')
        print(f"\nSaved label distribution plot: {os.path.join(data_root, 'label_distribution.png')}")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    return all_unique, label_stats

def main():
    parser = argparse.ArgumentParser(description='Analyze liver MRI segmentation labels')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing labels/ folder')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"ERROR: Data root directory does not exist: {args.data_root}")
        return
    
    analyze_labels(args.data_root)

if __name__ == '__main__':
    main()