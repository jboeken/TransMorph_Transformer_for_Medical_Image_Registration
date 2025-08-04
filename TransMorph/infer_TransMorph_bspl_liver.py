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

def main():
    # Data and model paths
    data_root = '/Users/janik/Documents/GitHub/FactorCL-Liver/2D_Liver/FactorCL/Flos Daten'
    model_folder = 'TransMorphBSpline_Liver_mse_1_diffusion_0.01/'
    model_dir = 'experiments/' + model_folder
    
    # Select which model checkpoint to use (-1 for latest)
    model_idx = -1
    
    # Load model configuration
    config = CONFIGS_TM['TransMorphBSpline']
    model = TransMorph_bspl.TranMorphBSplineNet(config)
    
    # Load trained weights
    if os.path.exists(model_dir):
        model_files = natsorted(glob.glob(model_dir + '*.pth.tar'))
        if model_files:
            best_model = torch.load(model_files[model_idx])['state_dict']
            print('Best model: {}'.format(os.path.basename(model_files[model_idx])))
            model.load_state_dict(best_model)
        else:
            print("No trained model found! Please train the model first.")
            return
    else:
        print("Model directory does not exist! Please train the model first.")
        return
    
    model.cuda()
    
    # Prepare test dataset
    test_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])
    
    test_set = LiverMRIValidationDataset(data_root, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    
    # Evaluation metrics
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    
    # CSV output setup
    csv_name = 'experiments/' + model_folder[:-1] + '_results'
    if os.path.exists(csv_name + '.csv'):
        os.remove(csv_name + '.csv')
    
    # Write CSV header
    csv_writter('Patient,DSC_Raw,DSC_Registered,Jacobian_Negative', csv_name)
    
    print("Starting inference...")
    with torch.no_grad():
        patient_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]  # moving image
            y = data[1]  # fixed image
            x_seg = data[2]  # moving segmentation
            y_seg = data[3]  # fixed segmentation

            # Forward pass
            x_def, flow, disp = model((x, y))
            flow = disp
            
            # Warp moving segmentation to fixed space
            def_out = transformation.warp(x_seg.cuda().float(), disp.cuda(), interp_mode='nearest')
            
            # Calculate Jacobian determinant
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            negative_jac_ratio = np.sum(jac_det <= 0) / np.prod(tar.shape)
            
            # Calculate Dice coefficients
            dsc_registered = utils.dice_val(def_out.long(), y_seg.long(), 3)  # 3 classes
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 3)
            
            # Update metrics
            eval_dsc_def.update(dsc_registered.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            eval_det.update(negative_jac_ratio, x.size(0))
            
            # Write results to CSV
            line = f'Patient_{patient_idx:03d},{dsc_raw.item():.4f},{dsc_registered.item():.4f},{negative_jac_ratio:.6f}'
            csv_writter(line, csv_name)
            
            print(f'Patient {patient_idx}: Raw DSC: {dsc_raw.item():.4f}, Registered DSC: {dsc_registered.item():.4f}, Neg Jac: {negative_jac_ratio:.6f}')
            patient_idx += 1
            
            # Optional: Save some example results
            if patient_idx <= 5:
                save_example_results(x, y, x_def, def_out, y_seg, patient_idx, model_folder)

        # Print final results
        print('\n=== Final Results ===')
        print('Raw DSC: {:.4f} +- {:.4f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
        print('Registered DSC: {:.4f} +- {:.4f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('Negative Jacobian Ratio: {:.6f} +- {:.6f}'.format(eval_det.avg, eval_det.std))
        print('DSC Improvement: {:.4f}'.format(eval_dsc_def.avg - eval_dsc_raw.avg))
        
        # Write summary to CSV
        csv_writter('', csv_name)
        csv_writter('Summary:', csv_name)
        csv_writter(f'Raw DSC Mean,{eval_dsc_raw.avg:.4f}', csv_name)
        csv_writter(f'Raw DSC Std,{eval_dsc_raw.std:.4f}', csv_name)
        csv_writter(f'Registered DSC Mean,{eval_dsc_def.avg:.4f}', csv_name)
        csv_writter(f'Registered DSC Std,{eval_dsc_def.std:.4f}', csv_name)
        csv_writter(f'Negative Jacobian Mean,{eval_det.avg:.6f}', csv_name)
        csv_writter(f'DSC Improvement,{eval_dsc_def.avg - eval_dsc_raw.avg:.4f}', csv_name)

def save_example_results(moving, fixed, registered, seg_registered, seg_fixed, patient_idx, model_folder):
    """Save example registration results as images"""
    save_dir = f'results/{model_folder[:-1]}/'
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