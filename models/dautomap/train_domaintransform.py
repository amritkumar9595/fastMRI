"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Fourier Initialisation of the domain_transform layer of the dAutomap

"""

import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch
import torchvision
try:
    from tensorboardX import SummaryWriter
except:
    pass
from torch.nn import functional as F
from torch.utils.data import DataLoader

import sys, os
sys.path.append(os.getcwd())

from common.args import Args
from common.subsample import MaskFunc2,no_masking_func, arc_masking_func

from data.mri_data import SliceData2
# from models.unet.unet_model import UnetModel
from models.dautomap.dautomap_model import dAUTOMAP
from models.wrappers import ResidualForm, ModelWithDC
from data import transforms
from tqdm import tqdm
# from data.transforms import stack_to_rss, stack_to_chans


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from data.my_transforms import BasicMaskingTransform # SquareDataTransformC3_multi 

def create_datasets_multi(args):
    train_mask = MaskFunc2(arc_masking_func, 15, (args.resolution, args.resolution),4)
    dev_mask = MaskFunc2(arc_masking_func, 15, (args.resolution, args.resolution),4)

    train_data = SliceData2(
        root=args.data_path / f'{args.challenge}_train2',
        transform=BasicMaskingTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )
    dev_data = SliceData2(
        root=args.data_path / f'{args.challenge}_val2',
        transform=BasicMaskingTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
    )
    return dev_data, train_data

# def create_datasets(args):
#     train_mask = MaskFunc(args.center_fractions, args.accelerations)
#     dev_mask = MaskFunc(args.center_fractions, args.accelerations)

#     train_data = SliceData(
#         root=args.data_path / f'{args.challenge}_train',
#         transform=SquareDataTransformC3_multi(train_mask, args.resolution, args.challenge),
#         sample_rate=args.sample_rate,
#         challenge=args.challenge
#     )
#     dev_data = SliceData(
#         root=args.data_path / f'{args.challenge}_val',
#         transform=SquareDataTransformC3_multi(dev_mask, args.resolution, args.challenge, use_seed=True),
#         sample_rate=args.sample_rate,
#         challenge=args.challenge,
#     )
#     return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets_multi(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=6,
        pin_memory=True,
    )
    
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    
    return train_loader, dev_loader, display_loader


def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in (enumerate(tqdm(data_loader))):

        stacked_kspace_square, _, _, stacked_image_square = data
        # input = input.unsqueeze(1).to(args.device)
        # target = target.to(args.device)
        ksp_shifted_mc = transforms.ifftshift(stacked_kspace_square,dim=(-2,-1))
        output = model(ksp_shifted_mc.cuda()) #.squeeze(1)
        print("output",output.shape)
        out_chans = stack_to_chans(output)
        out_ksp = transforms.fft2(out_chans)
       
        print("out_chans",out_ksp.shape)


        
        loss = F.l1_loss(output, stacked_image_square.cuda(),reduction = 'sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        if writer is not None:
            writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            stacked_kspace_square, _, _, stacked_image_square = data
            # input = input.unsqueeze(1).to(args.device)
            
            output = model(stacked_kspace_square.cuda()) #.squeeze(1)
            
            # mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            # std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            
            # to be done only on magnitude
            # target = target * std + mean
            # output = output * std + mean

            # norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            # norm = 1 # can't divide directly with complex
            # loss = F.mse_loss(output , target / norm, size_average=False)

            loss = F.mse_loss(output,stacked_kspace_square.cuda(),reduction = 'sum')
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):

            stacked_kspace_square, _, _, stacked_image_square = data
            # ksp,input, target, mean, std, norm = data
            # input = input.unsqueeze(1).to(args.device)
            # target = target.to(args.device)
            # target = target.unsqueeze(1)
            # print("target",target.shape)
            output = model(stacked_kspace_square.cuda())
            # print("output",output.shape)
            output_rss = stack_to_rss(output)
            output_rss = output_rss.unsqueeze(1)
            target = stack_to_rss(stacked_image_square).unsqueeze(1)
            # print("output_rss",output_rss.shape)
            

            #FIXME - not ready yet
            save_image(target, 'Target')
            save_image(output_rss, 'Reconstruction')
            save_image((target - output_rss.cpu()), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


# def build_model(args):
#     model = UnetModel(
#         in_chans=1,
#         out_chans=1,
#         chans=args.num_chans,
#         num_pool_layers=args.num_pools,
#         drop_prob=args.drop_prob
#     ).to(args.device)
#     return model

def build_dautomap(args):
    # checkpoint = torch.load(checkpoint_file)
    # args = checkpoint['args']

    patch_size = args.resolution
    model_params = {
      'input_shape': (30, patch_size, patch_size),
      'output_shape': (1, patch_size, patch_size),
      'tfx_params': {
        'nrow': patch_size,
        'ncol': patch_size,
        'nch_in': 30,
        'kernel_size': 1,
        'nl': None,
        'init_fourier': False,
        'init': 'kaiming_normal_',  #'xavier_uniform_',
        'bias': False, #True,
        'share_tfxs': False,
        'learnable': True
      },
      'tfx_params2': {
        'nrow': patch_size,
        'ncol': patch_size,
        'nch_in': 30,
        'kernel_size': 1,
        'nl': 'relu',
        'init_fourier': False,
        'init': 'kaiming_normal_', # 'xavier_uniform_',
        'bias':True,
        'share_tfxs': False,
        'learnable': True
      },
      'depth': 2,
      'nl':'relu'
    }

    model = dAUTOMAP(model_params['input_shape'],model_params['output_shape'],model_params['tfx_params'],model_params['tfx_params2']).to(args.device)

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # model.load_state_dict(checkpoint['model'])
    return model

def build_model(args):
    model = build_dautomap(args)
    # unet_model = UnetModel(in_chans=2,out_chans=2,chans=args.num_chans,
    #     num_pool_layers=args.num_pools,
    #     drop_prob=args.drop_prob)
    # print(args)
    # if args.residual:
    #     model = ResidualForm(model)
    # if args.dcblock:
    #     model = ModelWithDC(model)

    # dualencoderunet_model = build_dualencoderunet(args)
    # model = dAUTOMAPDualEncoderUnet(dautomap_model,dualencoderunet_model).to(args.device)
    # model = dautomap_model
    mdl = model.domain_transform  ## taking the domain _transform part only!!

    return mdl.to(args.device)
    # return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    # optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)

    return optimizer


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)

    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = Args()
    # parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    # parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    # parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--residual', default=True, help='residual')
    parser.add_argument('--dcblock', default=False, help='data consistency block')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

