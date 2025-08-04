import os, glob, random
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from .data_utils import pkload
from skimage.transform import resize


class LiverMRIDataset(Dataset):
    """
    Dataset for liver MRI registration training.
    Handles variable number of sequences per patient and missing sequences.
    Data structure: MRI/patientID_B_C_R_applyXFM_XXXX.nii.gz
    """
    def __init__(self, data_root, transforms, split='train', val_ratio=0.2, target_size=(160, 192, 224)):
        self.data_root = data_root
        self.transforms = transforms
        self.target_size = target_size
        self.mri_dir = os.path.join(data_root, 'MRI')
        self.labels_dir = os.path.join(data_root, 'labels')
        
        # Get all MRI files and extract patient sessions
        mri_files = glob.glob(os.path.join(self.mri_dir, '*_B_C_R_applyXFM_*.nii.gz'))
        patient_sessions = {}
        
        for f in mri_files:
            basename = os.path.basename(f)
            # Extract patient session ID (everything before _B_C_R_applyXFM_XXXX.nii.gz)
            if '_B_C_R_applyXFM_' in basename:
                patient_id = basename.split('_B_C_R_applyXFM_')[0]
                sequence_num = basename.split('_B_C_R_applyXFM_')[1].replace('.nii.gz', '')
                
                if patient_id not in patient_sessions:
                    patient_sessions[patient_id] = []
                patient_sessions[patient_id].append(sequence_num)
        
        # Filter patients with at least 2 sequences available
        patient_data = []
        for patient_id, sequences in patient_sessions.items():
            if len(sequences) >= 2:  # Need at least 2 sequences for registration
                # Check if any corresponding label exists
                label_pattern = os.path.join(self.labels_dir, f'{patient_id}_B_C_R_applyXFM*')
                label_files = glob.glob(label_pattern)
                
                if label_files:  # At least one label file exists
                    patient_data.append({
                        'patient_id': patient_id,
                        'sequences': sorted(sequences),
                        'label_file': label_files[0]  # Use first available label
                    })
        
        # Split into train/val
        random.seed(42)
        random.shuffle(patient_data)
        split_idx = int(len(patient_data) * (1 - val_ratio))
        
        if split == 'train':
            self.patient_data = patient_data[:split_idx]
        else:
            self.patient_data = patient_data[split_idx:]
        
        # Pre-compute all valid pairs to avoid complex indexing during training
        self.pairs = []
        for patient in self.patient_data:
            sequences = patient['sequences']
            patient_id = patient['patient_id']
            label_file = patient['label_file']
            
            # Generate all possible pairs for this patient (limit to reduce memory)
            for i, moving_seq in enumerate(sequences):
                for j, fixed_seq in enumerate(sequences):
                    if i != j:  # Don't pair sequence with itself
                        self.pairs.append({
                            'patient_id': patient_id,
                            'moving_seq': moving_seq,
                            'fixed_seq': fixed_seq,
                            'label_file': label_file
                        })
        
        print(f"Loaded {len(self.patient_data)} patients for {split}")
        print(f"Total sequences available: {sum(len(p['sequences']) for p in self.patient_data)}")
        print(f"Generated {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def load_nifti(self, filepath):
        """Load NIfTI file and return data array"""
        nii = nib.load(filepath)
        return nii.get_fdata().astype(np.float32)

    def normalize_image(self, img):
        """Normalize image to [0, 1] range"""
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img
    
    def auto_resize(self, img):
        """Automatically resize image to target size"""
        if img.shape != self.target_size:
            img = resize(img, self.target_size, anti_aliasing=False, order=3, preserve_range=True)
        return img.astype(np.float32)

    def __getitem__(self, index):
        # Get pre-computed pair
        pair = self.pairs[index]
        patient_id = pair['patient_id']
        moving_seq = pair['moving_seq']
        fixed_seq = pair['fixed_seq']
        
        # Construct file paths
        moving_path = os.path.join(self.mri_dir, f'{patient_id}_B_C_R_applyXFM_{moving_seq}.nii.gz')
        fixed_path = os.path.join(self.mri_dir, f'{patient_id}_B_C_R_applyXFM_{fixed_seq}.nii.gz')
        
        # Load images
        try:
            moving_img = self.load_nifti(moving_path)
            fixed_img = self.load_nifti(fixed_path)
        except Exception as e:
            print(f"Error loading {moving_path} or {fixed_path}: {e}")
            # Return a random valid pair as fallback
            return self.__getitem__(0)
        
        # Auto-resize and normalize images
        moving_img = self.auto_resize(moving_img)
        fixed_img = self.auto_resize(fixed_img)
        moving_img = self.normalize_image(moving_img)
        fixed_img = self.normalize_image(fixed_img)
        
        # Add channel dimension: [1, H, W, D]
        moving_img = moving_img[None, ...]
        fixed_img = fixed_img[None, ...]
        
        # Apply transforms
        if self.transforms:
            moving_img, fixed_img = self.transforms([moving_img, fixed_img])
        
        # Ensure contiguous arrays
        moving_img = np.ascontiguousarray(moving_img)
        fixed_img = np.ascontiguousarray(fixed_img)
        
        # Convert to tensors
        moving_img = torch.from_numpy(moving_img)
        fixed_img = torch.from_numpy(fixed_img)
        
        return moving_img, fixed_img


class LiverMRIValidationDataset(Dataset):
    """
    Dataset for liver MRI validation with segmentation masks.
    Adapted for the new data structure with flexible sequence availability.
    """
    def __init__(self, data_root, transforms, target_size=(160, 192, 224)):
        self.data_root = data_root
        self.transforms = transforms
        self.target_size = target_size
        self.mri_dir = os.path.join(data_root, 'MRI')
        self.labels_dir = os.path.join(data_root, 'labels')
        
        # Get all MRI files and extract patient sessions
        mri_files = glob.glob(os.path.join(self.mri_dir, '*_B_C_R_applyXFM_*.nii.gz'))
        patient_sessions = {}
        
        for f in mri_files:
            basename = os.path.basename(f)
            if '_B_C_R_applyXFM_' in basename:
                patient_id = basename.split('_B_C_R_applyXFM_')[0]
                sequence_num = basename.split('_B_C_R_applyXFM_')[1].replace('.nii.gz', '')
                
                if patient_id not in patient_sessions:
                    patient_sessions[patient_id] = []
                patient_sessions[patient_id].append(sequence_num)
        
        # Filter patients with validation data
        self.patient_data = []
        for patient_id, sequences in patient_sessions.items():
            if len(sequences) >= 2:  # Need at least 2 sequences
                # Check for label files
                label_pattern = os.path.join(self.labels_dir, f'{patient_id}_B_C_R_applyXFM*')
                label_files = glob.glob(label_pattern)
                
                if label_files:
                    self.patient_data.append({
                        'patient_id': patient_id,
                        'sequences': sorted(sequences),
                        'label_files': label_files
                    })
        
        # Use subset for validation (limit to reasonable number)
        self.patient_data = self.patient_data[:min(20, len(self.patient_data))]
        print(f"Loaded {len(self.patient_data)} patients for validation")

    def load_nifti(self, filepath):
        """Load NIfTI file and return data array"""
        nii = nib.load(filepath)
        return nii.get_fdata().astype(np.float32)

    def normalize_image(self, img):
        """Normalize image to [0, 1] range"""
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        return img

    def auto_resize(self, img):
        """Automatically resize image to target size"""
        if img.shape != self.target_size:
            img = resize(img, self.target_size, anti_aliasing=False, order=3, preserve_range=True)
        return img.astype(np.float32)

    def auto_resize_segmentation(self, seg):
        """Automatically resize segmentation to target size using nearest neighbor"""
        if seg.shape != self.target_size:
            seg = resize(seg, self.target_size, anti_aliasing=False, order=0, preserve_range=True)
        return seg.astype(np.int16)

    def process_segmentation(self, seg):
        """Process multi-class segmentation labels"""
        # Your labels are already discrete integers representing different anatomical structures:
        # 0 = background
        # 1 = lesion  
        # 2 = vessel
        # 3 = normal liver tissue
        # Just ensure proper data type and handle any potential floating point artifacts
        seg_discrete = np.round(seg).astype(np.int16)
        
        # Optional: Print unique labels for debugging (first time only)
        if not hasattr(self, '_labels_printed'):
            unique_labels = np.unique(seg_discrete)
            print(f"Segmentation labels found: {unique_labels}")
            self._labels_printed = True
            
        return seg_discrete

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, index):
        patient = self.patient_data[index]
        patient_id = patient['patient_id']
        sequences = patient['sequences']
        
        # Use first two available sequences for consistency
        moving_seq = sequences[0]
        fixed_seq = sequences[1] if len(sequences) > 1 else sequences[0]
        
        # Construct file paths
        moving_path = os.path.join(self.mri_dir, f'{patient_id}_B_C_R_applyXFM_{moving_seq}.nii.gz')
        fixed_path = os.path.join(self.mri_dir, f'{patient_id}_B_C_R_applyXFM_{fixed_seq}.nii.gz')
        
        # Use first available label file
        label_path = patient['label_files'][0]
        
        try:
            # Load images and segmentation
            moving_img = self.load_nifti(moving_path)
            fixed_img = self.load_nifti(fixed_path)
            seg_img = self.load_nifti(label_path)
        except Exception as e:
            print(f"Error loading validation data for patient {patient_id}: {e}")
            # Return a fallback (first patient)
            if index > 0:
                return self.__getitem__(0)
            else:
                raise e
        
        # Auto-resize and normalize images
        moving_img = self.auto_resize(moving_img)
        fixed_img = self.auto_resize(fixed_img)
        moving_img = self.normalize_image(moving_img)
        fixed_img = self.normalize_image(fixed_img)
        
        # Auto-resize and process segmentation (use same seg for both moving and fixed for simplicity)
        seg_img = self.auto_resize_segmentation(seg_img)
        moving_seg = self.process_segmentation(seg_img)
        fixed_seg = self.process_segmentation(seg_img)
        
        # Add channel dimension
        moving_img = moving_img[None, ...]
        fixed_img = fixed_img[None, ...]
        moving_seg = moving_seg[None, ...]
        fixed_seg = fixed_seg[None, ...]
        
        # Apply transforms
        if self.transforms:
            moving_img, moving_seg = self.transforms([moving_img, moving_seg])
            fixed_img, fixed_seg = self.transforms([fixed_img, fixed_seg])
        
        # Ensure contiguous arrays
        moving_img = np.ascontiguousarray(moving_img)
        fixed_img = np.ascontiguousarray(fixed_img)
        moving_seg = np.ascontiguousarray(moving_seg)
        fixed_seg = np.ascontiguousarray(fixed_seg)
        
        # Convert to tensors
        moving_img = torch.from_numpy(moving_img)
        fixed_img = torch.from_numpy(fixed_img)
        moving_seg = torch.from_numpy(moving_seg)
        fixed_seg = torch.from_numpy(fixed_seg)
        
        return moving_img, fixed_img, moving_seg, fixed_seg