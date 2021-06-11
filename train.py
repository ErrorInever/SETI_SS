import argparse
import torch
import logging
import os
import time
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
from models.effecient import EfficientNet
from data.dataset import SETIDataset
from torch.utils.data import DataLoader
from utils import (seed_everything, get_train_file_path, get_scheduler, AverageMeter,
                   split_data_kfold)
from config import cfg
from metric_logger import MetricLogger
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser(description='SLE-GC-GAN')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--load_model', dest='load_model', help='Path to model.pth.tar', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or tpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--version_name', dest='version_name', help='Version name for wandb', default=None, type=str)
    parser.add_argument('--model_version', dest='model_version', help='Specified version of model', default=None,
                        type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle, '
                                                              'input api key', default=None, type=str)
    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, device):
    """
    :param model: model
    :param optimizer: optimizer
    :param criterion: loss
    :param dataloader: train data dataloader
    :param device: gpu or tpu
    :return: average loss
    """
    model.train()
    losses = AverageMeter()
    if cfg.USE_APEX:
        scaler = GradScaler()
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        batch_size = label.size(0)
        img = img.to(device)
        label = label.to(device)
        if cfg.USE_APEX:
            with torch.cuda.amp.autocast():
                y_preds = model(img)
                loss = criterion(y_preds.view(-1), label)
        else:
            y_preds = model(img)
            loss = criterion(y_preds.view(-1), label)

        losses.update(loss.item(), batch_size)

        if cfg.USE_APEX:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        if batch_idx % cfg.LOG_FREQ == 0:
            MetricLogger.train_loss_batch(loss.item())

        loop.set_postfix(
            loss=loss
        )

    return losses.avg


def eval_one_epoch(model, criterion, dataloader, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        batch_size = label.size(0)
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_preds = model(img)
        loss = criterion(y_preds.view(-1), label)
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())

        if batch_idx % cfg.LOG_FREQ == 0:
            MetricLogger.val_loss_batch(loss.item())

        loop.set_postfix(
            loss=loss
        )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.device in ['gpu', 'tpu'], 'incorrect device type'

    cfg.DATA_ROOT = args.data_path
    logger = logging.getLogger('main')

    if args.version_name:
        project_version = args.version_name
    else:
        project_version = 'efficient_net'

    if args.model_version:
        assert args.model_version in cfg.EFFICIENT_VERSIONS, 'incorrect model version'
        cfg.EFFICIENT_VERSIONS = [args.model_version]

    if args.load_model:
        if not args.model_version:
            raise ValueError('load model version not specified')
        else:
            cfg.LOAD_MODEL = True
            model_file_name = args.load_model
            cfg.EFFICIENT_VERSIONS = [args.model_version]

    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    if args.wandb_id:
        cfg.WANDB_ID = args.wandb_id

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')

    if args.device == 'gpu':
        device = torch.device('cuda')

    # if args.device == 'tpu':
    #     try:
    #         import torch_xla.core.xla_model as xm
    #         import torch_xla.distributed.parallel_loader as pl
    #         import torch_xla.distributed.xla_multiprocessing as xmp
    #         import torch_xla.utils.serialization as xser
    #
    #         os.environ['XLA_USE_BF16'] = "1"
    #         os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'
    #
    #     except ImportError:
    #         logger.error('cannot import xla')

    logger.info(f'Using device:{args.device}')

    # define dataset
    data_root = args.data_path
    train_path = os.path.join(data_root, 'train_labels.csv')
    train_df = pd.read_csv(train_path)
    train_df['file_path'] = train_df['id'].apply(get_train_file_path)
    # Split KFold
    train_df = split_data_kfold(train_df)

    for version in cfg.EFFICIENT_VERSIONS:
        cfg.PROJECT_VERSION_NAME = f'{project_version}_{version}'
        # defining model
        if cfg.LOAD_MODEL:
            model = EfficientNet(version, num_classes=cfg.NUM_CLASSES, in_channels=cfg.IMG_CHANNELS).to(device)
            cp = torch.load(model_file_name, map_location=device)
            model.load_state_dict(cp['model'])
            logger.info(f"model loaded from {model_file_name}")
        else:
            model = EfficientNet(version, num_classes=cfg.NUM_CLASSES, in_channels=cfg.IMG_CHANNELS).to(device)
            logger.info(f'init default model version {version}')

        for fold in range(cfg.N_FOLD):
            logger.info(f'--------------------[{fold}-of-{cfg.N_FOLD}--------------------[')
            train_idxs = train_df[train_df['fold'] != fold].index
            val_idxs = train_df[train_df['fold'] == fold].index

            train_folds = train_df.loc[train_idxs].reset_index(drop=True)
            val_folds = train_df.loc[val_idxs].reset_index(drop=True)
            # y targets
            val_labels = val_folds['target'].values

            train_dataset = SETIDataset(train_folds, transform=True)
            val_dataset = SETIDataset(val_folds)

            train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2,
                                          pin_memory=True, drop_last=True)
            val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True,
                                        drop_last=False)

            # defining optimizer, scheduler, loss
            optimizer = optim.Adam(model.parameters(model), lr=cfg.LEARNING_RATE, betas=cfg.BETAS,
                                   weight_decay=cfg.WEIGHT_DECAY)
            scheduler = get_scheduler(optimizer)
            criterion = nn.BCEWithLogitsLoss()

            metric_logger = MetricLogger(version)

            best_score = 0.
            best_loss = np.inf
            for epoch in range(cfg.NUM_EPOCHS):
                # train model
                avg_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, device)
                # evaluate model
                avg_val_loss, preds = eval_one_epoch(model, criterion, val_dataloader, device)

                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                elif isinstance(scheduler, CosineAnnealingLR):
                    scheduler.step()
                elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step()
                # metric
                auc_score = roc_auc_score(val_labels, preds)
                metric_logger.log(avg_loss, avg_val_loss, auc_score)
                # save model
                if auc_score > best_score:
                    best_score = auc_score
                    logger.info(f"Fined the best score {best_score:.4f}: EPOCH {epoch}")
                    logger.info(f"Save model to {cfg.OUTPUT_DIR}")
                    torch.save({
                        'model': model.state_dict(),
                        'preds': preds
                    }, cfg.OUTPUT_DIR + f"efficient_{version}_best_auc.pth.tar")
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    logger.info(f"Fined the best evaluation loss {best_loss:.4f}: EPOCH {epoch}")
                    logger.info(f"Save model to {cfg.OUTPUT_DIR}")
                    torch.save({
                        'model': model.state_dict(),
                        'preds': preds
                    }, cfg.OUTPUT_DIR + f"efficient_{version}_best_val_loss.pth.tar")

                if epoch % cfg.SAVE_EPOCH_FREQ == 0:
                    logger.info(f"Save model epoch {epoch}")
                    torch.save({
                        'model': model.state_dict(),
                        'preds': preds
                    }, cfg.OUTPUT_DIR + f"efficient_{version}_epoch_{epoch}.pth.tar")
