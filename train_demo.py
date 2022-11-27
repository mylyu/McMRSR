import os
import argparse
import json
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn

from torchvision.utils import save_image
import torchvision.transforms as T
from utils import prepare_sub_folder
from datasets import get_datasets
from models import create_model
import scipy.io as sio
import csv
import pdb

parser = argparse.ArgumentParser(description='McMRSR')

# model name
parser.add_argument('--experiment_name', type=str, default='test_fixefft', help='give a experiment name before training')
parser.add_argument('--model_type', type=str, default='McMRSR', help='model type')
parser.add_argument('--resume', type=str, default=None, help='Filename of the checkpoint to resume')

# dataset
parser.add_argument('--data_root', type=str, default='/data/SZTU200H5', help='data root folder')
parser.add_argument('--list_dir', type=str, default='data_list_M4Raw', help='data list_dir root folder')
parser.add_argument('--mask_path', type=str, default='data_demo/dc_mask/lr_4x', help='data list_dir root folder')
parser.add_argument('--dataset', type=str, default='M4Raw', help='dataset name')
parser.add_argument('--input_contrast', type=str, default='FLAIR', help='input contrast type, e.g. T1, T2, FLAIR')
parser.add_argument('--ref_contrast', type=str, default='T1', help='ref contrast type, e.g. T1, T2, FLAIR')
parser.add_argument('--online_reg', type=str, default= None, help='online reg using ANTS')
parser.add_argument('--multinex', default=False, action='store_true', help='whether to use multiplex NEXs')

# model architectures
parser.add_argument('--net_G', type=str, default='McMRSR', help='generator network')
parser.add_argument('--n_recurrent', type=int, default=1, help='No modification required')
parser.add_argument('--use_prior', default=False, action='store_true', help='No modification required')

#Swin pama
parser.add_argument('--upscale', type=int, default=4, help='upscale')
parser.add_argument('--window_size', type=int, default=8, help='window_size')
parser.add_argument('--height', type=int, default=64, help='lr_height')
parser.add_argument('--width', type=int, default=64, help='lr_width')

# loss options
parser.add_argument('--wr_L1', type=float, default=1, help='weight for reconstruction L1 loss')

# training options
parser.add_argument('--n_epochs', type=int, default=200, help='number of epoch')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')

# evaluation options
parser.add_argument('--eval_epochs', type=int, default=10, help='evaluation epochs')
parser.add_argument('--save_epochs', type=int, default=50, help='save evaluation for every number of epochs')

parser.add_argument('--center_fractions', type=float, default=1.0/8.0, help='Cartesian: cernter fraction')
parser.add_argument('--accelerations', type=float, default=5.0, help='Cartesian: acceleration rate')

parser.add_argument('--n_lines', type=float, default=np.round(256 * 0.16), help='Radial: number of radial lines')

parser.add_argument('--n_interleaves', type=float, default=11, help='Spiral: number of interleaves')

# optimizer
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# parser.add_argument('--d_lr', type=float, default=0.0004, help='learning rate')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
# parser.add_argument('--d_beta1', type=float, default=0.0, help='beta1 for ADAM')
# parser.add_argument('--d_beta2', type=float, default=0.9, help='beta2 for ADAM')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

# learning rate policy
parser.add_argument('--lr_policy', type=str, default='step', help='learning rate decay policy')
parser.add_argument('--step_size', type=int, default=1000, help='step size for step scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay ratio for step scheduler')

# logger options
parser.add_argument('--snapshot_epochs', type=int, default=10, help='save model for every number of epochs')
parser.add_argument('--log_freq', type=int, default=100, help='save model for every number of epochs')
parser.add_argument('--output_path', default='./', type=str, help='Output path.')

# other
parser.add_argument('--num_workers', type=int, default=8, help='number of threads to load data')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
# parser.add_argument('--gpu_ids', type=int, default=[0,1], help='list of gpu ids')
opts = parser.parse_args()

options_str = json.dumps(opts.__dict__, indent=4, sort_keys=False)
print("------------------- Options -------------------")
print(options_str[2:-2])
print("-----------------------------------------------")

cudnn.benchmark = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = create_model(opts)
model.setgpu(opts.gpu_ids)
# model.cuda()

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Number of parameters: {} \n'.format(num_param))

if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_iter = 0
else:
    ep0, total_iter = model.resume(opts.resume)

model.set_scheduler(opts, ep0)
ep0 += 1
print('Start training at epoch {} \n'.format(ep0))

# select dataset
train_set, val_set, test_set = get_datasets(opts)
train_loader = DataLoader(dataset=train_set, num_workers=opts.num_workers, batch_size=opts.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=test_set, num_workers=opts.num_workers, batch_size=1, shuffle=False)

# Setup directories
output_directory = os.path.join(opts.output_path, 'outputs', opts.experiment_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)

with open(os.path.join(output_directory, 'options.json'), 'w') as f:
    f.write(options_str)

with open(os.path.join(output_directory, 'train_loss.csv'), 'w') as f:
    writer = csv.writer(f)
    writer.writerow(model.loss_names)

if opts.resume is not None:
    print('Direct Evaluation after Resume......')
    model.eval()
    with torch.no_grad():
        model.evaluate(val_loader)
# training loop
for epoch in range(ep0, opts.n_epochs + 1):

    train_bar = tqdm(train_loader)
    model.train()
    model.set_epoch(epoch)
    for it, data in enumerate(train_bar):
        total_iter += 1
        model.set_input(data)
        model.optimize()
        train_bar.set_description(desc='[Epoch {}]'.format(epoch) + model.loss_summary)

        if it % opts.log_freq == 0:
            with open(os.path.join(output_directory, 'train_loss.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(model.get_current_losses().values())

    model.update_learning_rate()

    # save checkpoint
    if (epoch+1) % opts.snapshot_epochs == 0:
        checkpoint_name = os.path.join(checkpoint_directory, 'model_{}.pt'.format(epoch))
        model.save(checkpoint_name, epoch, total_iter)

    # evaluation
    if (epoch+1) % opts.eval_epochs == 0:
        print('Validation Evaluation ......')
        pred = os.path.join(image_directory, 'pred_{:03d}.png'.format(epoch))
        gt = os.path.join(image_directory, 'gt_{:03d}.png'.format(epoch))
        input_sub = os.path.join(image_directory, 'input_{:03d}.png'.format(epoch))

        if opts.wr_L1 > 0:
            print(model.recon.detach().shape)
            recon = model.recon.detach()*2-1
            vis_pred = (torch.abs(recon[:, 0:1, :, :]) ** 2 + torch.abs(recon[:, 1:2, :, :]) ** 2).sqrt()
            save_image(vis_pred, pred, normalize=True, scale_each=True, padding=5)
            tag_image_full = model.tag_image_full.detach()*2-1
            vis_gt = (torch.abs(tag_image_full[:, 0:1, :, :]) ** 2 + torch.abs(tag_image_full[:, 1:2, :, :]) ** 2).sqrt()
            save_image(vis_gt, gt, normalize=True, scale_each=True, padding=5)
            tag_image_sub = model.tag_image_sub.detach()*2-1
            vis_input = (torch.abs(tag_image_sub[:, 0:1, :, :]) ** 2 + torch.abs(tag_image_sub[:, 1:2, :, :]) ** 2).sqrt()
            transform = T.Resize(size = (256,256))
            vis_input = transform(vis_input)
            save_image(vis_input, input_sub, normalize=True, scale_each=True, padding=5)

        model.eval()
        with torch.no_grad():
            model.evaluate(val_loader)

        with open(os.path.join(output_directory, 'metrics.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, model.psnr_recon, model.ssim_recon])

    if (epoch+1) % opts.save_epochs == 0:
        sio.savemat(os.path.join(image_directory, 'eval.mat'), model.results)
