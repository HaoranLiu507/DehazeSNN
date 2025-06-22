import json
import os
import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
import functools
from collections import OrderedDict
import logging

from utils import AverageMeter, write_img, chw_to_hwc
from data.loader import PairLoader
from models import *

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DehazeSNN-M', type=str, help='model name')
parser.add_argument('--num_workers', default=2, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./datasets/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./output/results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='reside6k', type=str, help='dataset name')
parser.add_argument('--exp', default='reside6k', type=str, help='experiment setting')
parser.add_argument('--output', default='./output/', type=str, help='path to output')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
parser.add_argument('--last_20', default=False, type=bool, help='GPUs used for training')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

@functools.lru_cache()
def create_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'

    # create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handler
    file_handler = logging.FileHandler(os.path.join(output_dir + '/log/', 'log.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def single(save_dir):
    """
       Process the loaded state dictionary to remove the 'module.' prefix.
       This is necessary if the model was trained using DataParallel (multi-GPU), which adds 'module.' to the keys.
    """
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    """
        Test the network on the test dataset and record the results.

        Args:
            test_loader: DataLoader for the test dataset.
            network: The neural network model to evaluate.
            result_dir: Directory where output images and CSV results will be saved.

        Returns:
            Average PSNR and SSIM values computed over the test set.
    """
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # Convert both output and target from [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        # Log the current result (current and average values)
        logger.info('Test: [{0}]\t'
                    'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
                    'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
                    .format(idx, psnr=PSNR, ssim=SSIM))

        # Write the current image's results to the CSV file
        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(result_dir, 'imgs', filename), out_img)

    f_result.close()
    return PSNR.avg, SSIM.avg


if __name__ == '__main__':

    # Select and build the model based on the command-line argument
    if args.model == 'DehazeSNN-S':
        model = build_S_model()
    elif args.model == 'DehazeSNN-M':
        model = build_M_model()
    elif args.model == 'DehazeSNN-L':
        model = build_L_model()
    else:
        print('Model not found')
        exit()

    model.cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '_best.pth')
    logger = create_logger(output_dir=args.output, name=f"{args.model}")

    # If the saved model exists, load its state dictionary
    if os.path.exists(saved_model_dir):
        logger.info('==> Start testing, current model name: ' + args.model)
        model_state_dict = torch.load(saved_model_dir)
        model.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    # Prepare the test dataset and dataloader
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    # Define the directory where test results will be saved
    result_dir_1 = os.path.join(args.result_dir, args.dataset, args.model)

    # Run the test function to evaluate the model on the test set
    avg_psnr, avg_ssim = test(test_loader, model, result_dir_1)

    # Save the best model's test results to a CSV file
    os.makedirs(os.path.join(result_dir_1), exist_ok=True)
    with open(os.path.join(result_dir_1, 'allTest.csv'), 'a') as f1_result:
        f1_result.write(f'best,%.02f,%.04f\n' % (avg_psnr, avg_ssim))

    # Optionally, test the last 20 epochs if specified by the command-line argument
    if args.last_20:
        # Starting epoch number for the last 20 epochs
        start_epoch = 480
        for i in range(start_epoch, start_epoch + 20):
            saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + f'_epoch_{i}.pth')
            if os.path.exists(saved_model_dir):
                model_state_dict = torch.load(saved_model_dir)
                model.load_state_dict(single(saved_model_dir))
            else:
                logger.info(f'==> No existing {saved_model_dir}')
                continue
            result_dir = os.path.join(args.result_dir, args.dataset, args.model, str(i))
            logger.info('==> Start testing, current model path: ' + saved_model_dir)
            avg_psnr, avg_ssim = test(test_loader, model, result_dir)

            with open(os.path.join(result_dir_1, 'allTest.csv'), 'a') as f2_result:
                f2_result.write('%d,%.02f,%.03f\n' % (i, avg_psnr, avg_ssim))



