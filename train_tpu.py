import argparse
import gc

import torch
import logging
import os
import time
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
from data.dataset import SETIDataset
from torch.utils.data import DataLoader
from utils import (seed_everything, get_train_file_path, get_scheduler, AverageMeter,
                   split_data_kfold, print_result, save_checkpoint, mix_up_data, loss_mix_up)
from config import cfg
from metric_logger import MetricLogger
from sklearn.metrics import roc_auc_score
from models.pretrained_models import get_model

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train model on TPU')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--model_name', dest='model_name', help='train specified model', default=None, type=str)
    parser.add_argument('--ckpt', dest='ckpt', help='path to model statedict.pth.tar', default=None, type=str)
    parser.add_argument('--version_name', dest='version_name', help='Version name for wandb', default=None, type=str)
    parser.add_argument('--eff_ver', dest='eff_ver', help='Efficient version', default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle, '
                                                              'input api key', default=None, type=str)
    parser.add_argument('--test_epoch', dest='test_epoch', help='train one epoch for test', action='store_true')
    parser.add_argument('--num_epoch', dest='num_epoch', help='number of epochs', default=None, type=int)
    parser.add_argument('--n_fold', dest='n_fold', help='start from fold', default=None, type=int)
    parser.add_argument('--n_epoch', dest='n_epoch', help='start from epoch', default=None, type=int)

    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, metric_logger, epoch):
    """
    :param model: model
    :param optimizer: optimizer
    :param criterion: loss
    :param dataloader: train data dataloader
    :param device: tpu
    :return: average loss
    """
    model.train()
    losses = AverageMeter()
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        batch_size = label.size(0)
        # TODO: add mixup
        # img, label_a, label_b, lam = mix_up_data(img, label, use_cuda=False)
        # loss = loss_mix_up(criterion, y_preds.view(-1), label_a, label_b, lam)
        optimizer.zero_grad()
        y_preds = model(img)
        loss = criterion(y_preds.view(-1), label)
        losses.update(loss, batch_size)
        loss.backward()
        xm.optimizer_step(optimizer)

        if batch_idx % cfg.LOG_FREQ == 0:
            metric_logger.train_loss_batch(loss, epoch, len(dataloader), batch_idx)

        loop.set_postfix(
            loss=loss
        )

    return losses.avg


def eval_one_epoch(model, criterion, dataloader, metric_logger, epoch):
    model.eval()
    losses = AverageMeter()
    preds = []
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        batch_size = label.size(0)
        with torch.no_grad():
            y_preds = model(img)
        loss = criterion(y_preds.view(-1), label)
        losses.update(loss, batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())

        if batch_idx % cfg.LOG_FREQ == 0:
            metric_logger.val_loss_batch(loss, epoch, len(dataloader), batch_idx)

        loop.set_postfix(
            loss=loss
        )

    predictions = np.concatenate(preds)
    gc.collect()
    return losses.avg, predictions


def run_tpu(rank, train_df, mx_model, model_name, start_epoch):
    torch.set_default_tensor_type('torch.FloatTensor')
    device = xm.xla_device()
    xm.set_rng_state(cfg.SEED, device)

    logger.info(f"Running on device: {device}")

    train_idxs = train_df[train_df['fold'] != fold].index
    val_idxs = train_df[train_df['fold'] == fold].index

    train_folds = train_df.loc[train_idxs].reset_index(drop=True)
    val_folds = train_df.loc[val_idxs].reset_index(drop=True)
    val_labels = val_folds['target'].values

    train_dataset = SETIDataset(train_folds, transform=True)
    val_dataset = SETIDataset(val_folds)

    # special sampler needed for distributed/multi-core (divides dataset among the replicas/cores/devices)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.TPU_WORKER,
                                  drop_last=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.TPU_WORKER,
                                drop_last=False, sampler=valid_sampler)

    train_dataloader = pl.MpDeviceLoader(train_dataloader, device)  # puts the train data onto the current TPU core
    val_dataloader = pl.MpDeviceLoader(val_dataloader, device)    # puts the valid data onto the current TPU core

    model = mx_model.to(device)     # put model onto the current TPU core

    # scale the learning rate by number of cores
    optimizer = optim.Adam(model.parameters(model), lr=cfg.LEARNING_RATE * xm.xrt_world_size(), betas=cfg.BETAS,
                           weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)
    criterion = nn.BCEWithLogitsLoss()

    gc.collect()

    metric_logger = MetricLogger(model_name)

    best_score = 0.
    best_loss = np.inf
    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        avg_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, metric_logger, epoch)
        avg_val_loss, preds = eval_one_epoch(model, criterion, val_dataloader, metric_logger, epoch)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        auc_score = roc_auc_score(val_labels, preds)

        metric_logger.log(avg_loss, avg_val_loss, auc_score)
        logger.info(f"Epoch:{epoch} | avg_train_loss:{avg_loss:.4f} | avg_val_loss:{avg_val_loss:.4f}")
        logger.info(f"------ROC_AUC_SCORE: {auc_score}")

        gc.collect()

        xm.rendezvous('save_model')

        if auc_score > best_score:
            best_score = auc_score
            save_path = cfg.OUTPUT_DIR + f"XLA_{model_name}_fold_{fold}_best_roc_auc.pth.tar"
            xm.save(model.state_dict(), save_path)
            logger.info(f"Found the best roc_auc_score, save model to {save_path}")
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = cfg.OUTPUT_DIR + f"XLA_{model_name}_fold_{fold}_best_val_loss.pth.tar"
            xm.save(model.state_dict(), save_path)
            logger.info(f"Found the best validation loss, save model to {save_path}")


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    logger = logging.getLogger('train')

    assert args.data_path, 'data path not specified'
    assert args.model_name in ['efficient', 'nfnet'], 'incorrect model name'

    cfg.DATA_ROOT = args.data_path

    if args.version_name:
        project_version = args.version_name
        cfg.PROJECT_VERSION_NAME = project_version
    else:
        project_version = 'unnamed model'

    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    if args.wandb_id:
        cfg.WANDB_ID = args.wandb_id

    if args.out_dir:
        cfg.OUTPUT_DIR = args.out_dir

    if args.test_epoch:
        cfg.NUM_EPOCHS = 1

    if args.num_epoch:
        cfg.NUM_EPOCHS = args.num_epoch

    if args.n_fold:
        start_fold = args.n_fold
    else:
        start_fold = 0

    if args.n_epoch:
        start_epoch = args.n_epoch
    else:
        start_epoch = 0

    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        import torch_xla.utils.serialization as xser

        os.environ['XLA_USE_BF16'] = "1"
        os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
    except ImportError as e:
        logger.error(f"cannot import XLA libraries. try reinstall")

    if args.model_name:
        if args.model_name == 'efficient':
            model_name = 'efficientnet'
            if args.eff_ver:
                eff_ver = args.eff_ver
            else:
                eff_ver = 'b0'
            model = get_model(model_name=model_name, version=eff_ver, pretrained=True)

        elif args.model_name == 'nfnet_l0':
            model_name = 'nfnet_l0'
            get_model(model_name=model_name)
        else:
            raise ValueError('no model name')

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')


    mx_model = xmp.MpModelWrapper(model)

    # define dataset
    data_root = args.data_path
    train_path = os.path.join(data_root, 'train_labels.csv')
    train_df = pd.read_csv(train_path)
    train_df['file_path'] = train_df['id'].apply(get_train_file_path)
    # Split KFold
    train_df = split_data_kfold(train_df)

    for fold in range(start_fold, cfg.N_FOLD):
        xmp.spawn(run_tpu, args=(train_df, mx_model, model_name, start_epoch,), nprocs=8, start_method='fork')
        logger.info("done")
