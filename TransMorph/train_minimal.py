"""
Minimal training script to debug the hanging issue
"""

import os
import torch
from torch.utils.data import DataLoader
from data.liver_datasets import LiverMRIDataset, LiverMRIValidationDataset
from data import trans
import numpy as np
from torchvision import transforms
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    args = parser.parse_args()
    
    print("=== Minimal Training Test ===")
    
    # Simple transforms
    train_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])
    
    # Create dataset with minimal configuration
    print("Creating dataset...")
    train_set = LiverMRIDataset(args.data_root, transforms=train_composed, split='train', target_size=(160, 192, 224))
    
    print(f"Dataset size: {len(train_set)}")
    
    # Create simple data loader - NO MULTIPROCESSING
    print("Creating data loader...")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    print("Testing data loading...")
    for i, data in enumerate(train_loader):
        print(f"Loaded batch {i+1}: {[d.shape for d in data]}")
        
        if i >= 2:  # Only test first 3 batches
            break
    
    print("Data loading test completed successfully!")

if __name__ == "__main__":
    main()