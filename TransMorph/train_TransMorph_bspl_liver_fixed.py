"""
Fixed TransMorph training script with dataset size limiting and proper data loading
Copy this to your remote machine as train_TransMorph_bspl_liver_fixed.py
"""

import argparse
import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchvision import transforms
from natsort import natsorted

import utils
from data.liver_datasets import LiverMRIDataset, LiverMRIValidationDataset
from data import trans
import models.transformation as transformation
from models.TransMorph_bspl import CONFIGS as CONFIGS_TM
import models.TransMorph_bspl as TransMorph_bspl


class Logger(object):
    def __init__(self, log_dir):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_dir, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_sz[0], grid_step):
        grid_img[j+line_thickness-1, :, :] = 1
    for i in range(0, grid_sz[1], grid_step):
        grid_img[:, i+line_thickness-1, :] = 1
    for i in range(0, grid_sz[2], grid_step):
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


def adjust_learning_rate(optimizer, epoch, max_epoch, initial_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(initial_lr * np.power(1 - (epoch) / max_epoch, power), 8)


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)
    
    # Load configuration
    config = CONFIGS_TM['TransMorphBSpline']
    config.img_size = args.img_size
    config.window_size = args.window_size
    
    # Create model
    model = TransMorph_bspl.TranMorphBSplineNet(config)
    model.cuda()
    
    # Load pretrained weights if provided
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        pretrained_dict = torch.load(args.pretrained_weights, map_location='cpu')
        model_dict = model.state_dict()
        
        # Filter pretrained dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f'Loading pre-trained weights from: {args.pretrained_weights}')
        print(f'Loaded {len(pretrained_dict)} pretrained parameters')
    
    # Data transforms
    train_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])
    val_composed = transforms.Compose([
        trans.NumpyType((np.float32, np.int16)),
    ])
    
    # Create datasets with LIMITED SIZE
    print("Creating datasets...")
    train_set = LiverMRIDataset(args.data_root, transforms=train_composed, split='train', target_size=config.img_size)
    val_set = LiverMRIValidationDataset(args.data_root, transforms=val_composed, target_size=config.img_size)
    
    # CRITICAL: Limit training dataset size to prevent memory issues
    if hasattr(train_set, 'pairs') and len(train_set.pairs) > args.max_pairs:
        print(f"Limiting training pairs from {len(train_set.pairs)} to {args.max_pairs}")
        train_set.pairs = train_set.pairs[:args.max_pairs]
    
    print(f"Final training dataset size: {len(train_set)}")
    print(f"Validation dataset size: {len(val_set)}")
    
    # Create data loaders - NO MULTIPROCESSING
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, 
                           num_workers=0, pin_memory=False, drop_last=True)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0, amsgrad=True)
    criterion = nn.MSELoss()
    
    # Create output directories
    save_dir = f'TransMorphBSpline_Liver_mse_{args.loss_weights[0]}_diffusion_{args.loss_weights[1]}/'
    exp_dir = os.path.join(args.output_dir, save_dir)
    log_dir = os.path.join('logs', save_dir)
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    writer = SummaryWriter(log_dir=log_dir)
    sys.stdout = Logger(log_dir)
    
    print("=== Training Configuration ===")
    print(f"Model: TransMorph B-Spline")
    print(f"Dataset size: {len(train_set)} training pairs")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"Loss weights: {args.loss_weights}")
    print(f"Image size: {args.img_size}")
    print(f"Number of classes: {args.num_classes}")
    print("=" * 40)
    
    # Training loop
    best_dsc = 0
    transformation_model = transformation.SpatialTransformer(config.img_size, mode='bilinear')
    transformation_model.cuda()
    
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch}/{args.max_epoch}')
        
        # Training phase
        model.train()
        loss_all = utils.AverageMeter()
        
        for idx, data in enumerate(train_loader):
            adjust_learning_rate(optimizer, epoch, args.max_epoch, args.lr)
            
            data = [t.cuda() for t in data]
            x = data[0]  # moving image
            y = data[1]  # fixed image
            
            # Forward pass
            x_def, flow, disp = model((x, y))
            
            # Calculate losses
            loss_mse = criterion(x_def, y)
            loss_smooth = utils.smoothloss(disp)
            
            loss = loss_mse * args.loss_weights[0] + loss_smooth * args.loss_weights[1]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_all.update(loss.item(), y.numel())
            
            if idx % 10 == 0:
                print(f'Iter {idx+1}/{len(train_loader)} loss {loss.item():.4f}, '
                      f'MSE: {loss_mse.item():.6f}, Smooth: {loss_smooth.item():.6f}')
        
        print(f'Epoch {epoch} training loss: {loss_all.avg:.4f}')
        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        
        # Validation phase
        if epoch % 5 == 0:  # Validate every 5 epochs
            model.eval()
            eval_dsc = utils.AverageMeter()
            
            with torch.no_grad():
                for data in val_loader:
                    data = [t.cuda() for t in data]
                    x = data[0]
                    y = data[1]
                    x_seg = data[2]
                    y_seg = data[3]
                    
                    # Forward pass
                    output = model((x, y))
                    
                    # Warp segmentation
                    def_out = transformation_model(x_seg.cuda().float(), output[2].cuda())
                    
                    # Calculate Dice
                    dsc = utils.dice_val(def_out.long(), y_seg.long(), args.num_classes)
                    eval_dsc.update(dsc.item(), x.size(0))
            
            print(f'Validation DSC: {eval_dsc.avg:.4f}')
            writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
            
            # Save best model
            if eval_dsc.avg > best_dsc:
                best_dsc = eval_dsc.avg
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dsc': best_dsc,
                    'optimizer': optimizer.state_dict(),
                }, save_dir=exp_dir+'/', filename=f'liver_dsc{eval_dsc.avg:.3f}.pth.tar')
                print(f'New best model saved with DSC: {best_dsc:.4f}')
    
    print('Training completed!')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path to data root directory')
    parser.add_argument('--pretrained_weights', type=str, help='Path to pretrained weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int, default=200, help='Maximum epochs')
    parser.add_argument('--img_size', type=int, nargs=3, default=[160, 192, 224], help='Image size [H W D]')
    parser.add_argument('--window_size', type=int, nargs=3, default=[5, 6, 7], help='Window size [H W D]')
    parser.add_argument('--num_classes', type=int, default=8, help='Number of segmentation classes')
    parser.add_argument('--class_names', type=str, nargs='*', 
                       default=['background', 'liver_tissue', 'portal_vein', 'arterial_vessels', 'lesion', 'class_5', 'abdominal_vena_cava', 'thoracal_vena_cava'],
                       help='Class names for segmentation')
    parser.add_argument('--loss_weights', type=float, nargs=2, default=[1.0, 0.01], help='Loss weights [mse, smooth]')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    parser.add_argument('--max_pairs', type=int, default=1000, help='Maximum training pairs to prevent memory issues')
    
    args = parser.parse_args()
    
    main(args)