import os, losses, utils, glob
from torch.utils.data import DataLoader
from data.liver_datasets import LiverMRIValidationDataset
from data import trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
import models.transformation as transformation
from models.TransMorph_bspl import CONFIGS as CONFIGS_TM
import models.TransMorph_bspl as TransMorph_bspl
import torch.nn.functional as F
import argparse
import nibabel as nib
from skimage.transform import resize

def resize_image_to_model(img, target_size):
    """Resize image to model input size"""
    if len(img.shape) == 5:  # [B, C, H, W, D]
        img_np = img[0, 0].cpu().numpy()
    elif len(img.shape) == 4:  # [B, H, W, D] 
        img_np = img[0].cpu().numpy()
    else:  # [H, W, D]
        img_np = img
    
    if img_np.shape != tuple(target_size):
        img_resized = resize(img_np, target_size, anti_aliasing=False, order=1, preserve_range=True)
    else:
        img_resized = img_np
    
    return torch.from_numpy(img_resized).float().cuda()[None, None, ...]

def resize_flow_to_original(flow, original_size):
    """Resize flow field back to original size and scale appropriately"""
    flow_np = flow[0].cpu().numpy()  # [3, H, W, D] -> [3, H, W, D]
    original_size = tuple(original_size)
    
    if flow_np.shape[1:] == original_size:
        return flow
    
    # Resize each flow component
    flow_resized = np.zeros((3,) + original_size, dtype=np.float32)
    scale_factors = [original_size[i] / flow_np.shape[i+1] for i in range(3)]
    
    for i in range(3):
        flow_resized[i] = resize(flow_np[i], original_size, anti_aliasing=False, order=1, preserve_range=True)
        flow_resized[i] *= scale_factors[i]  # Scale flow values appropriately
    
    return torch.from_numpy(flow_resized).cuda()[None, ...]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/Users/janik/Documents/GitHub/FactorCL-Liver/2D_Liver/FactorCL/Flos Daten', help='Path to data root')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory for results')
    parser.add_argument('--model_size', type=int, nargs=3, default=[160, 192, 224], help='Model input size [H W D]')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of segmentation classes')
    parser.add_argument('--use_labels', action='store_true', help='Use segmentation labels for evaluation (if available)')
    args = parser.parse_args()
    
    # Load model configuration
    config = CONFIGS_TM['TransMorphBSpline']
    config.img_size = args.model_size
    model = TransMorph_bspl.TranMorphBSplineNet(config)
    
    # Load trained weights from training script format
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f'Loaded model from: {args.model_path}')
            if 'best_dsc' in checkpoint:
                print(f'Model best DSC: {checkpoint["best_dsc"]:.4f}')
        else:
            print("Invalid checkpoint format!")
            return
    else:
        print(f"Model file does not exist: {args.model_path}")
        return
    
    model.cuda()
    model.eval()
    
    # Create custom dataset that can handle optional labels
    class InferenceDataset(LiverMRIValidationDataset):
        def __init__(self, data_root, transforms, use_labels=False):
            self.use_labels = use_labels
            if use_labels:
                super().__init__(data_root, transforms, target_size=None)  # Don't auto-resize
            else:
                # Initialize for image-only mode
                self.data_root = data_root
                self.transforms = transforms
                self.target_size = None
                self.mri_dir = os.path.join(data_root, 'MRI')
                
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
                
                # Use all patients with at least 2 sequences (no label requirement)
                self.patient_data = []
                for patient_id, sequences in patient_sessions.items():
                    if len(sequences) >= 2:
                        self.patient_data.append({
                            'patient_id': patient_id,
                            'sequences': sorted(sequences)
                        })
                
                print(f"Loaded {len(self.patient_data)} patients for image-only inference")
            
        def auto_resize(self, img):
            # Don't resize - keep original resolution
            return img.astype(np.float32)
            
        def auto_resize_segmentation(self, seg):
            # Don't resize - keep original resolution  
            return seg.astype(np.int16)
            
        def __getitem__(self, index):
            if self.use_labels:
                return super().__getitem__(index)
            else:
                # Image-only mode
                # Find which patient and sequence pair this index corresponds to
                cumulative_pairs = 0
                patient_idx = 0
                
                for i, patient in enumerate(self.patient_data):
                    num_pairs = len(patient['sequences']) * (len(patient['sequences']) - 1)
                    if index < cumulative_pairs + num_pairs:
                        patient_idx = i
                        pair_idx = index - cumulative_pairs
                        break
                    cumulative_pairs += num_pairs
                
                patient = self.patient_data[patient_idx]
                sequences = patient['sequences']
                
                # Calculate which sequence pair this corresponds to
                num_sequences = len(sequences)
                moving_seq_idx = pair_idx // (num_sequences - 1)
                fixed_seq_idx = pair_idx % (num_sequences - 1)
                
                # Adjust fixed index to skip the moving sequence
                if fixed_seq_idx >= moving_seq_idx:
                    fixed_seq_idx += 1
                
                moving_seq = sequences[moving_seq_idx]
                fixed_seq = sequences[fixed_seq_idx]
                
                # Construct file paths
                moving_path = os.path.join(self.mri_dir, f'{patient["patient_id"]}_B_C_R_applyXFM_{moving_seq}.nii.gz')
                fixed_path = os.path.join(self.mri_dir, f'{patient["patient_id"]}_B_C_R_applyXFM_{fixed_seq}.nii.gz')
                
                # Load images
                try:
                    moving_img = self.load_nifti(moving_path)
                    fixed_img = self.load_nifti(fixed_path)
                except Exception as e:
                    print(f"Error loading {moving_path} or {fixed_path}: {e}")
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
        
        def __len__(self):
            if self.use_labels:
                return super().__len__()
            else:
                return sum(len(p['sequences']) * (len(p['sequences']) - 1) for p in self.patient_data)
    
    # Prepare test dataset
    test_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])
    
    test_set = InferenceDataset(args.data_root, transforms=test_composed, use_labels=args.use_labels)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    
    # Evaluation metrics
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    
    # CSV output setup
    os.makedirs(args.output_dir, exist_ok=True)
    csv_name = os.path.join(args.output_dir, 'inference_results')
    if os.path.exists(csv_name + '.csv'):
        os.remove(csv_name + '.csv')
    
    # Write CSV header based on whether labels are used
    if args.use_labels:
        csv_writter('Patient,DSC_Raw,DSC_Registered,Jacobian_Negative', csv_name)
    else:
        csv_writter('Patient,Jacobian_Negative', csv_name)
    
    print("Starting inference...")
    print(f"Model input size: {args.model_size}")
    print(f"Using labels: {args.use_labels}")
    
    with torch.no_grad():
        patient_idx = 0
        for data in test_loader:
            data = [t.cuda() for t in data]
            x_orig = data[0]  # moving image - original resolution
            y_orig = data[1]  # fixed image - original resolution
            
            if args.use_labels:
                x_seg_orig = data[2]  # moving segmentation - original resolution
                y_seg_orig = data[3]  # fixed segmentation - original resolution
            
            # Get original image size
            original_size = x_orig.shape[2:]  # [H, W, D]
            print(f'Patient {patient_idx}: Original size {original_size}, scaling to {args.model_size}')
            
            # Resize images to model input size
            x_model = resize_image_to_model(x_orig, args.model_size)
            y_model = resize_image_to_model(y_orig, args.model_size)
            
            # Ensure float type for model input (consistency with training)
            x_model = x_model.float()
            y_model = y_model.float()
            
            # Forward pass on model-sized images
            x_def_model, flow_model, disp_model = model((x_model, y_model))
            
            # Resize flow back to original resolution
            disp_orig = resize_flow_to_original(disp_model, original_size)
            
            # Calculate Jacobian determinant on original resolution
            jac_det = utils.jacobian_determinant_vxm(disp_orig.detach().cpu().numpy()[0, :, :, :, :])
            negative_jac_ratio = np.sum(jac_det <= 0) / np.prod(original_size)
            
            # Update Jacobian metric
            eval_det.update(negative_jac_ratio, x_orig.size(0))
            
            if args.use_labels:
                # Apply flow to segmentation and calculate Dice coefficients
                def_out = transformation.warp(x_seg_orig.cuda().float(), disp_orig.cuda(), interp_mode='nearest')
                dsc_registered = utils.dice_val(def_out.long(), y_seg_orig.long(), args.num_classes)
                dsc_raw = utils.dice_val(x_seg_orig.long(), y_seg_orig.long(), args.num_classes)
                
                # Update metrics
                eval_dsc_def.update(dsc_registered.item(), x_orig.size(0))
                eval_dsc_raw.update(dsc_raw.item(), x_orig.size(0))
                
                # Write results to CSV
                line = f'Patient_{patient_idx:03d},{dsc_raw.item():.4f},{dsc_registered.item():.4f},{negative_jac_ratio:.6f}'
                csv_writter(line, csv_name)
                
                print(f'Patient {patient_idx}: Raw DSC: {dsc_raw.item():.4f}, Registered DSC: {dsc_registered.item():.4f}, Neg Jac: {negative_jac_ratio:.6f}')
                
                # Optional: Save some example results
                if patient_idx < 5:
                    save_example_results(x_orig, y_orig, transformation.warp(x_orig.cuda().float(), disp_orig.cuda(), interp_mode='bilinear'), 
                                       def_out, y_seg_orig, patient_idx, args.output_dir)
            else:
                # No labels - just save registration results
                line = f'Patient_{patient_idx:03d},{negative_jac_ratio:.6f}'
                csv_writter(line, csv_name)
                
                print(f'Patient {patient_idx}: Registration completed, Neg Jac: {negative_jac_ratio:.6f}')
                
                # Optional: Save some example results without segmentation
                if patient_idx < 5:
                    save_example_results_no_seg(x_orig, y_orig, transformation.warp(x_orig.cuda().float(), disp_orig.cuda(), interp_mode='bilinear'), 
                                               patient_idx, args.output_dir)
            
            patient_idx += 1

        # Print final results
        print('\n=== Final Results ===')
        if args.use_labels:
            print('Raw DSC: {:.4f} +- {:.4f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
            print('Registered DSC: {:.4f} +- {:.4f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
            print('DSC Improvement: {:.4f}'.format(eval_dsc_def.avg - eval_dsc_raw.avg))
        print('Negative Jacobian Ratio: {:.6f} +- {:.6f}'.format(eval_det.avg, eval_det.std))
        
        # Write summary to CSV
        csv_writter('', csv_name)
        csv_writter('Summary:', csv_name)
        if args.use_labels:
            csv_writter(f'Raw DSC Mean,{eval_dsc_raw.avg:.4f}', csv_name)
            csv_writter(f'Raw DSC Std,{eval_dsc_raw.std:.4f}', csv_name)
            csv_writter(f'Registered DSC Mean,{eval_dsc_def.avg:.4f}', csv_name)
            csv_writter(f'Registered DSC Std,{eval_dsc_def.std:.4f}', csv_name)
            csv_writter(f'DSC Improvement,{eval_dsc_def.avg - eval_dsc_raw.avg:.4f}', csv_name)
        csv_writter(f'Negative Jacobian Mean,{eval_det.avg:.6f}', csv_name)

def save_example_results(moving, fixed, registered, seg_registered, seg_fixed, patient_idx, output_dir):
    """Save example registration results as images"""
    save_dir = os.path.join(output_dir, 'examples')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and select middle slice
    moving_np = moving.detach().cpu().numpy()[0, 0, :, :, moving.shape[-1]//2]
    fixed_np = fixed.detach().cpu().numpy()[0, 0, :, :, fixed.shape[-1]//2]
    registered_np = registered.detach().cpu().numpy()[0, 0, :, :, registered.shape[-1]//2]
    seg_reg_np = seg_registered.detach().cpu().numpy()[0, 0, :, :, seg_registered.shape[-1]//2]
    seg_fixed_np = seg_fixed.detach().cpu().numpy()[0, 0, :, :, seg_fixed.shape[-1]//2]
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Top row: images
    axes[0,0].imshow(moving_np, cmap='gray')
    axes[0,0].set_title('Moving Image')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(fixed_np, cmap='gray')
    axes[0,1].set_title('Fixed Image')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(registered_np, cmap='gray')
    axes[0,2].set_title('Registered Image')
    axes[0,2].axis('off')
    
    # Bottom row: segmentations
    axes[1,0].imshow(seg_reg_np, cmap='jet', alpha=0.7)
    axes[1,0].imshow(moving_np, cmap='gray', alpha=0.3)
    axes[1,0].set_title('Moving Seg (on moving)')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(seg_fixed_np, cmap='jet', alpha=0.7)
    axes[1,1].imshow(fixed_np, cmap='gray', alpha=0.3)
    axes[1,1].set_title('Fixed Seg (target)')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(seg_reg_np, cmap='jet', alpha=0.7)
    axes[1,2].imshow(fixed_np, cmap='gray', alpha=0.3)
    axes[1,2].set_title('Registered Seg (on fixed)')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_idx:03d}_results.png', dpi=150, bbox_inches='tight')
    plt.close()

def save_example_results_no_seg(moving, fixed, registered, patient_idx, output_dir):
    """Save example registration results without segmentation"""
    save_dir = os.path.join(output_dir, 'examples')
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy and select middle slice
    moving_np = moving.detach().cpu().numpy()[0, 0, :, :, moving.shape[-1]//2]
    fixed_np = fixed.detach().cpu().numpy()[0, 0, :, :, fixed.shape[-1]//2]
    registered_np = registered.detach().cpu().numpy()[0, 0, :, :, registered.shape[-1]//2]
    
    # Create comparison figure (images only)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(moving_np, cmap='gray')  
    axes[0].set_title('Moving Image')
    axes[0].axis('off')
    
    axes[1].imshow(fixed_np, cmap='gray')
    axes[1].set_title('Fixed Image')
    axes[1].axis('off')
    
    axes[2].imshow(registered_np, cmap='gray')
    axes[2].set_title('Registered Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_idx:03d}_registration.png', dpi=150, bbox_inches='tight')
    plt.close()

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()