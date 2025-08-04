from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
import argparse
from torch.utils.data import DataLoader
from data import trans
from data.liver_datasets import LiverMRIDataset, LiverMRIValidationDataset
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
import models.transformation as transformation
from models.TransMorph_bspl import CONFIGS as CONFIGS_TM
import models.TransMorph_bspl as TransMorph_bspl

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='TransMorph B-spline Transfer Learning for Liver MRI Registration')
    
    # Data paths
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing MRI/ and labels/ folders')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for model checkpoints and logs')
    parser.add_argument('--pretrained_weights', type=str, required=True,
                       help='Path to pre-trained TransMorph B-spline weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.00001,
                       help='Learning rate for fine-tuning (default: 0.00001)')
    parser.add_argument('--max_epoch', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--loss_weights', type=float, nargs=2, default=[1.0, 0.01],
                       help='Loss weights for [similarity, regularization] (default: 1.0 0.01)')
    
    # Model parameters
    parser.add_argument('--img_size', type=int, nargs=3, default=[160, 192, 224],
                       help='Input image size [H W D] (default: 160 192 224)')
    parser.add_argument('--window_size', type=int, nargs=3, default=[5, 6, 7],
                       help='Window size for attention [H W D] (default: 5 6 7)')
    parser.add_argument('--num_classes', type=int, default=8,
                       help='Number of segmentation classes including background (default: 8 for liver)')
    parser.add_argument('--class_names', type=str, nargs='*', 
                       default=['background', 'liver_tissue', 'portal_vein', 'arterial_vessels', 'lesion', 'class_5', 'abdominal_vena_cava', 'thoracal_vena_cava'],
                       help='Names of segmentation classes (default: background liver_tissue portal_vein arterial_vessels lesion class_5 abdominal_vena_cava thoracal_vena_cava)')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID to use (default: 0)')
    
    # Other options
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Train from scratch without pre-trained weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    return parser.parse_args()

def main(args):
    # Training parameters from arguments
    batch_size = args.batch_size
    data_root = args.data_root
    weights = args.loss_weights
    lr = args.lr
    max_epoch = args.max_epoch
    
    # Create output directories
    save_dir = 'TransMorphBSpline_Liver_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    exp_dir = os.path.join(args.output_dir, save_dir)
    log_dir = os.path.join('logs', save_dir)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_dir)
    
    # Training parameters
    epoch_start = 0
    
    # Pre-trained weights settings
    load_pretrained = not args.no_pretrained
    pretrained_path = args.pretrained_weights
    
    '''
    Initialize model with configuration from arguments
    '''
    config = CONFIGS_TM['TransMorphBSpline']
    # Update config with command line arguments
    config.img_size = tuple(args.img_size)
    config.window_size = tuple(args.window_size)
    
    model = TransMorph_bspl.TranMorphBSplineNet(config)
    model.cuda()

    '''
    Load pre-trained weights for transfer learning
    '''
    if load_pretrained and os.path.exists(pretrained_path):
        print(f"Loading pre-trained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path)
        if 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
        
        model_dict = model.state_dict()
        # Filter out unnecessary keys and load compatible weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} pretrained parameters")
    else:
        print("Training from scratch")

    '''
    Initialize training datasets
    '''
    train_composed = transforms.Compose([
        trans.RandomFlip(0),  # Random flip for data augmentation
        trans.NumpyType((np.float32, np.float32)),
    ])

    val_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])

    # Create datasets
    train_set = LiverMRIDataset(data_root, transforms=train_composed, split='train', target_size=config.img_size)
    val_set = LiverMRIValidationDataset(data_root, transforms=val_composed, target_size=config.img_size)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+save_dir)
    
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            output = model((x, y))
            loss = 0
            loss_vals = []
            _ = 0
            for n, loss_function in enumerate(criterions):
                if _ > 1: break
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
                _ += 1
            loss_all.update(loss.item(), y.numel())
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del output
            # flip fixed and moving images for bidirectional training
            loss = 0
            output = model((y, x))
            _ = 0
            for n, loss_function in enumerate(criterions):
                if _ > 1: break
                curr_loss = loss_function(output[n], x) * weights[n]
                loss_vals[n] += curr_loss
                loss += curr_loss
                _ += 1
            loss_all.update(loss.item(), y.numel())
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:  # Print every 10 iterations
                print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(
                    idx, len(train_loader), loss.item(), loss_vals[0].item()/2, loss_vals[1].item()/2))

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model((x, y))
                
                with torch.cuda.device(args.gpu):
                    def_out = transformation.warp(x_seg.cuda().float(), output[2].cuda(), interp_mode='nearest')
                    def_grid = transformation.warp(grid_img.float(), output[2].cuda(), interp_mode='bilinear')
                
                # Calculate Dice for liver structures
                dsc = utils.dice_val(def_out.long(), y_seg.long(), args.num_classes)
                
                # Optional: Print per-class dice scores for debugging
                if epoch == 0 and idx == 0:  # First validation of first epoch
                    print(f"Using {args.num_classes} classes: {args.class_names}")
                eval_dsc.update(dsc.item(), x.size(0))
                
        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir=exp_dir+'/', filename='liver_dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        
        # Visualization
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
        
        print('Epoch {} DSC: {:.4f}, Best DSC: {:.4f}'.format(epoch, eval_dsc.avg, best_dsc))
    
    writer.close()

def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=5):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*.pth.tar'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*.pth.tar'))

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    '''
    GPU configuration
    '''
    GPU_iden = args.gpu
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    
    if GPU_num > 0:
        torch.cuda.set_device(GPU_iden)
        GPU_avai = torch.cuda.is_available()
        print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
        print('If the GPU is available? ' + str(GPU_avai))
    else:
        print('No GPU available, using CPU')
    
    main(args)