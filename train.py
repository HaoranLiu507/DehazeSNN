import datetime
import gc
import inspect
import os
import argparse
import json
import sys
import time
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import functools
import logging
from tqdm import tqdm
from timm.scheduler.cosine_lr import CosineLRScheduler
from data.loader import PairLoader
from utils import AverageMeter
from models import *

# seed = 7
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DehazeSNN-M', type=str, help='model name')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--data_dir', default='./datasets/', type=str, help='path to dataset')
parser.add_argument('--dataset', default='reside6k', type=str, help='dataset name')
parser.add_argument('--exp', default='reside6k', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPUs used for training')
parser.add_argument('--output', default='./output/', type=str, help='path to output')
parser.add_argument('--resume', default=False, type=bool, help='resume train from checkpoint')
parser.add_argument('--fine_tuning', default=False, type=bool, help='use checkpoint to fine-tuning')
parser.add_argument('--loss', default='L1', type=str, help='loss function')
parser.add_argument('--weight', default=0.5, type=float, help='loss weight')
parser.add_argument('--accumulation_steps', default=1, type=int, help='train accumulation_steps')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# Logger Creation Function
@functools.lru_cache()
def create_logger(output_dir, name=''):
    """
    create logger
    """
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
    file_handler = logging.FileHandler(os.path.join(output_dir + '/log/', 'log.txt'), mode='a+')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


# Learning Rate Scheduler Builder
def build_scheduler(setting, optimizer, n_iter_per_epoch):
    num_steps = int(setting['epochs'] * n_iter_per_epoch)
    warmup_steps = int(setting['TRAIN_WARMUP_EPOCHS'] * n_iter_per_epoch)
    lr_scheduler = None
    if setting['TRAIN_LR_SCHEDULER'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=setting['TRAIN_MIN_LR'],
            warmup_lr_init=setting['TRAIN_WARMUP_LR'],
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    return lr_scheduler


# Training Function for One Epoch
def train(train_loader, model, criterion, weight, scaler, optimizer, optimizer_lif, epoch, lr_scheduler,
          lr_scheduler_lif, last_loss, best_psnr, best_epoch):
    losses = AverageMeter()

    torch.cuda.empty_cache()
    num_steps = len(train_loader)
    model.train()

    optimizer.zero_grad()
    if optimizer_lif is not None:
        optimizer_lif.zero_grad()

    start = time.time()

    for idx, batch in tqdm(enumerate(train_loader), total=num_steps,
                           desc="epoch " + str(epoch) + " loss " + str(round(last_loss, 4)) +
                                " best_psnr " + str(round(best_psnr, 3)) + " best_epoch " +
                                str(best_epoch), unit="batch", ncols=130):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()
        with autocast():
            output = model(source_img).clamp_(-1, 1)
            #
            loss = criterion[0](output, target_img)
            if args.loss == 'LPIPS':
                loss = torch.mean(loss)
            if args.loss == 'weight':
                loss = weight[0] * loss + (1 - weight[0]) * torch.mean(criterion[1](output, target_img))

        if args.accumulation_steps > 1:

            losses.update(loss.item())
            # print(losses.avg)
            loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (idx + 1) % args.accumulation_steps == 0 or (idx + 1) == num_steps:
                scaler.step(optimizer)
                if optimizer_lif is not None:
                    scaler.step(optimizer_lif)
                scaler.update()

                optimizer.zero_grad()
                if optimizer_lif is not None:
                    optimizer_lif.zero_grad()

                lr_scheduler.step_update(epoch * num_steps + idx)
                lr_scheduler_lif.step_update(epoch * num_steps + idx)
        else:

            losses.update(loss.item())

            optimizer.zero_grad()
            if optimizer_lif is not None:
                optimizer_lif.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            if optimizer_lif is not None:
                scaler.step(optimizer_lif)
            scaler.update()

            lr_scheduler.step_update(epoch * num_steps + idx)
            lr_scheduler_lif.step_update(epoch * num_steps + idx)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    return losses.avg, weight


# Validation Function
def valid(val_loader, network, criterion, weight):
    PSNR = AverageMeter()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    num_steps = len(val_loader)
    network.eval()
    start = time.time()

    for idx, batch in tqdm(enumerate(val_loader), total=num_steps, desc="test", unit="batch"):
        source_img = batch['source'].cuda()
        target_img = batch['target'].cuda()

        with torch.no_grad():
            output = network(source_img).clamp_(-1, 1)
            loss = criterion[0](output, target_img)
            if args.loss == 'LPIPS':
                loss = torch.mean(loss)
            if args.loss == 'weight':
                loss = weight[0] * loss + (1 - weight[0]) * torch.mean(criterion[1](output, target_img))
            losses.update(loss.item())

        mse_loss = F.mse_loss(output * 0.5 + 0.5, target_img * 0.5 + 0.5, reduction='none').mean((1, 2, 3))
        psnr = 10 * torch.log10(1 / mse_loss).mean()
        PSNR.update(psnr.item(), source_img.size(0))

    test_time = time.time() - start
    logger.info(f"Validation takes {datetime.timedelta(seconds=int(test_time))}")
    return PSNR.avg, losses.avg


# Data Loader Builder
def build_loader(setting, data_dir, dataset):
    num_workers = min([os.cpu_count(), setting['batch_size'] if setting['batch_size'] > 1 else 0, 24])
    dataset_dir = os.path.join(data_dir, dataset)
    train_dataset = PairLoader(dataset_dir, 'train', 'train',
                               setting['patch_size'], setting['edge_decay'], setting['only_h_flip'])

    train_loader = DataLoader(train_dataset,
                              batch_size=setting['batch_size'],
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_dataset = PairLoader(dataset_dir, 'test', setting['valid_mode'],
                             setting['patch_size'])
    val_loader = DataLoader(val_dataset,
                            batch_size=setting['batch_size'],
                            num_workers=num_workers,
                            pin_memory=True)

    return train_dataset, val_dataset, train_loader, val_loader


# Optimizer Configuration
def configure_optimizers(setting, main_params, lif_params):
    decay_params = [p for p in main_params if p.dim() >= 2]
    nodecay_params = [p for p in main_params if p.dim() < 2]
    main_optim_groups = [
        {'params': decay_params, 'weight_decay': setting['TRAIN_WEIGHT_DECAY']},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    # num_decay_params = sum(p.numel() for p in decay_params)
    # num_nodecay_params = sum(p.numel() for p in nodecay_params)

    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    optimizer = None
    if len(main_params) != 0:
        optimizer = torch.optim.AdamW(main_optim_groups,
                                      eps=setting['TRAIN_OPTIMIZER_EPS'],
                                      betas=eval(setting['TRAIN_OPTIMIZER_BETAS']),
                                      lr=setting['TRAIN_BASE_LR'],
                                      fused=use_fused)

    optimizer_lif = None
    if len(lif_params) != 0:
        optimizer_lif = torch.optim.AdamW(lif_params,
                                          eps=setting['TRAIN_OPTIMIZER_EPS'],
                                          betas=eval(setting['TRAIN_OPTIMIZER_BETAS']),
                                          lr=setting['TRAIN_LIF_LR'],
                                          weight_decay=setting['LIF_WEIGHT_DECAY'],
                                          fused=use_fused)

    return optimizer, optimizer_lif


if __name__ == '__main__':

    # Import Settings

    setting_filename = os.path.join('configs', args.exp, args.model + '.json')
    if not os.path.exists(setting_filename):
        setting_filename = os.path.join('configs', args.exp, 'default.json')
    with open(setting_filename, 'r') as f:
        setting = json.load(f)

    logger = create_logger(output_dir=args.output, name=f"{args.model}")

    # Build Model

    if args.model == 'DehazeSNN-S':
        model = build_S_model()
    elif args.model == 'DehazeSNN-M':
        model = build_M_model()
    elif args.model == 'DehazeSNN-L':
        model = build_L_model()
    else:
        print('Model not found')
        exit()
    model = nn.DataParallel(model).cuda()

    # Initialization

    logger.info(str(model))

    # Loss function

    criterion = []
    weight = []

    if args.loss == 'L1':
        criterion.append(nn.L1Loss())
    elif args.loss == 'LPIPS':
        criterion.append(lpips.LPIPS(net='alex').cuda())
    elif args.loss == 'weight':
        criterion.append(nn.L1Loss())
        criterion.append(lpips.LPIPS(net='alex').cuda())
        weight.append(args.weight)
    else:
        print('Loss function not found')
        exit()

    main_params = []
    lif_params = []
    for k, v in model.named_parameters():
        if k.find("tau") != -1 or k.find("Vth") != -1:
            lif_params.append(v)
        else:
            main_params.append(v)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    if args.fine_tuning:
        setting['TRAIN_BASE_LR'] = setting['TRAIN_BASE_LR'] / 10.0
        setting['TRAIN_LIF_LR'] = setting['TRAIN_LIF_LR'] / 10.0
        setting['TRAIN_WARMUP_EPOCHS'] = setting['TRAIN_WARMUP_EPOCHS'] / 2
        setting['epochs'] = setting['epochs'] / 5

    # Build Optimizer

    optimizer, optimizer_lif = configure_optimizers(setting, main_params, lif_params)

    # Build loader & scheduler

    dataset_train, dataset_val, train_loader, val_loader = build_loader(setting, args.data_dir, args.dataset)

    lr_scheduler = build_scheduler(setting, optimizer, len(train_loader))

    lr_scheduler_lif = None
    if optimizer_lif is not None:
        lr_scheduler_lif = build_scheduler(setting, optimizer_lif, len(train_loader))

    scaler = GradScaler()

    save_dir = os.path.join(args.save_dir, args.exp)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"===============================parameters=====================================================\n"
                f"args: {args}\n"
                f"setting: {setting}\n"
                f"==============================================================================================\n")
    # train

    if not os.path.exists(os.path.join(save_dir, args.model + '_current.pth')) or args.resume or args.fine_tuning:
        logger.info('==> Start training, current model name: ' + args.model)
        loss_list = []
        val_loss_list = []
        psnr_list = []
        best_psnr = 0
        best_epoch = 0
        start_epoch = 0
        last_loss = 0.0

        # resume

        if args.resume:
            logger.info(f"==============> Resuming form {save_dir, args.model + '_current.pth'}....................")
            saved_model_dir = os.path.join(save_dir, args.model + '_current.pth')
            checkpoint = torch.load(saved_model_dir)

            model.load_state_dict(checkpoint['state_dict'], strict=False)

            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_lif.load_state_dict(checkpoint['optimizer_lif'])

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            lr_scheduler_lif.load_state_dict(checkpoint['lr_scheduler_lif'])

            start_epoch = checkpoint['epoch'] + 1
            best_psnr = checkpoint['best_psnr']
            best_epoch = checkpoint['best_epoch']
            weight = checkpoint['weight']

        # fine_tuning

        if args.fine_tuning:
            logger.info(f"==============> fine_tuning use {save_dir, args.model + '_base.pth'}....................")
            saved_model_dir = os.path.join(save_dir, args.model + '_base.pth')
            checkpoint = torch.load(saved_model_dir)

            model.load_state_dict(checkpoint['state_dict'], strict=False)

        for epoch in range(start_epoch, setting['epochs']):
            loss, weight = train(train_loader, model, criterion, weight, scaler, optimizer,
                                 optimizer_lif, epoch, lr_scheduler,
                                 lr_scheduler_lif, last_loss, best_psnr, best_epoch)

            last_loss = loss

            gc.collect()
            torch.cuda.empty_cache()

            with open(os.path.join(save_dir, 'train.csv'), 'a') as f_result:
                f_result.write('%d,%.05f\n' % (epoch, loss))

            logger.info(f'train_loss: {loss}, epoch: {epoch}')

            # valid

            if (epoch % setting['eval_freq'] == 0) or (epoch > 0.9 * setting['epochs']):
                avg_psnr, val_loss = valid(val_loader, model, criterion, weight)

                logger.info(f'valid_psnr: {avg_psnr}, epoch: {epoch}')
                psnr_list.append(avg_psnr)

                logger.info(f'valid_loss: {val_loss}, epoch: {epoch}')

                with open(os.path.join(save_dir, 'valid.csv'), 'a') as f2_result:
                    f2_result.write('%d,%.05f,%.05f\n' % (epoch, val_loss, avg_psnr))

                # save checkpoint

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_epoch = epoch
                    torch.save({'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'optimizer_lif': optimizer_lif.state_dict() if optimizer_lif is not None else None,
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'lr_scheduler_lif': lr_scheduler_lif.state_dict() if lr_scheduler_lif is not None else None,
                                'epoch': epoch,
                                'best_psnr': best_psnr,
                                'best_epoch': best_epoch,
                                'weight': weight},
                               os.path.join(save_dir, args.model + '_best.pth'))

                if epoch == setting['epochs'] - 1:
                    torch.save({'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'optimizer_lif': optimizer_lif.state_dict() if optimizer_lif is not None else None,
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'lr_scheduler_lif': lr_scheduler_lif.state_dict() if lr_scheduler_lif is not None else None,
                                'epoch': epoch,
                                'best_psnr': best_psnr,
                                'best_epoch': best_epoch,
                                'weight': weight},
                               os.path.join(save_dir, args.model + '_last.pth'))

                torch.save({'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'optimizer_lif': optimizer_lif.state_dict() if optimizer_lif is not None else None,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'lr_scheduler_lif': lr_scheduler_lif.state_dict() if lr_scheduler_lif is not None else None,
                            'epoch': epoch,
                            'best_psnr': best_psnr,
                            'best_epoch': best_epoch,
                            'weight': weight},
                           os.path.join(save_dir, args.model + '_current.pth'))

                if setting['epochs'] - epoch <= 20:
                    torch.save({'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'optimizer_lif': optimizer_lif.state_dict() if optimizer_lif is not None else None,
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'lr_scheduler_lif': lr_scheduler_lif.state_dict() if lr_scheduler_lif is not None else None,
                                'epoch': epoch,
                                'best_psnr': best_psnr,
                                'best_epoch': best_epoch,
                                'weight': weight},
                               os.path.join(save_dir, args.model + f'_epoch_{epoch}.pth'))

                logger.info(f'best_psnr: {best_psnr}, best_epoch: {best_epoch}, current epoch: {epoch}')

    else:
        print('==> Existing trained model')
        exit(1)
