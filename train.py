import argparse
import gc

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
from data.dataset import SETIDataset
from torch.utils.data import DataLoader
from utils import (seed_everything, get_train_file_path, get_scheduler, AverageMeter,
                   split_data_kfold, print_result, save_checkpoint, mix_up_data, loss_mix_up)
from config import cfg
from metric_logger import MetricLogger
from sklearn.metrics import roc_auc_score
from models.pretrained_models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='Pretrained model experminet')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or tpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--eff_ver', dest='eff_ver', help='version of efficient', default=None, type=str)
    parser.add_argument('--ckpt', dest='ckpt', help='path to model statedict.pth.tar', default=None, type=str)
    parser.add_argument('--version_name', dest='version_name', help='Version name for wandb', default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle, '
                                                              'input api key', default=None, type=str)
    parser.add_argument('--test_epoch', dest='test_epoch', help='train one epoch for test', action='store_true')
    parser.add_argument('--num_epoch', dest='num_epoch', help='number of epochs', default=None, type=int)
    parser.add_argument('--n_fold', dest='n_fold', help='start from fold', default=None, type=int)
    parser.add_argument('--n_epoch', dest='n_epoch', help='start from epoch', default=None, type=int)

    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, metric_logger, device, epoch):
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
        img, label_a, label_b, lam = mix_up_data(img, label, use_cuda=True)
        img = img.to(device)
        label_a = label_a.to(device)
        label_b = label_b.to(device)

        if cfg.USE_APEX:
            with torch.cuda.amp.autocast():
                y_preds = model(img)
                loss = loss_mix_up(criterion, y_preds.view(-1), label_a, label_b, lam)
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
            metric_logger.train_loss_batch(loss.item(), epoch, len(dataloader), batch_idx)

        loop.set_postfix(
            loss=loss.item()
        )

    return losses.avg


def eval_one_epoch(model, criterion, dataloader, metric_logger, device, epoch):
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
            metric_logger.val_loss_batch(loss.item(), epoch, len(dataloader), batch_idx)

        loop.set_postfix(
            loss=loss.item()
        )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.device in ['gpu', 'tpu'], 'incorrect device type'
    assert args.eff_ver in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], 'incorrect version'

    cfg.DATA_ROOT = args.data_path
    logger = logging.getLogger('main')

    if args.version_name:
        project_version = args.version_name
        cfg.PROJECT_VERSION_NAME = project_version
    else:
        project_version = 'unnamed model'

    if args.eff_ver:
        name_model = f'efficient_{args.eff_ver}'
        model_version = args.eff_ver

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

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')

    if args.device == 'gpu':
        device = torch.device('cuda')

    logger.info(f'Using device:{args.device}')

    # define dataset
    data_root = args.data_path
    train_path = os.path.join(data_root, 'train_labels.csv')
    train_df = pd.read_csv(train_path)
    train_df['file_path'] = train_df['id'].apply(get_train_file_path)
    # Split KFold
    train_df = split_data_kfold(train_df)

    start_epoch = 0
    oof_df = pd.DataFrame()
    for fold in range(start_fold, cfg.N_FOLD):
        train_idxs = train_df[train_df['fold'] != fold].index
        val_idxs = train_df[train_df['fold'] == fold].index
        train_folds = train_df.loc[train_idxs].reset_index(drop=True)
        val_folds = train_df.loc[val_idxs].reset_index(drop=True)
        val_labels = val_folds['target'].values
        train_dataset = SETIDataset(train_folds, transform=True)
        val_dataset = SETIDataset(val_folds)
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2,
                                      pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True,
                                    drop_last=False)
        # defining optimizer, scheduler, loss
        if args.ckpt:
            try:
                model = get_model('efficientnet', version=model_version).to(device)
                state = torch.load(args.ckpt)
                model.load_state_dict(state['model'])
                logger.info(f"resume training")
            except Exception:
                logger.error("incorrect type or path of model")
                raise ValueError
        else:
            model = get_model('efficientnet', version=model_version).to(device)
            logger.info(f"load default weights")

        optimizer = optim.Adam(model.parameters(model), lr=cfg.LEARNING_RATE, betas=cfg.BETAS,
                               weight_decay=cfg.WEIGHT_DECAY, amsgrad=False)
        scheduler = get_scheduler(optimizer)
        criterion = nn.BCEWithLogitsLoss()
        metric_logger = MetricLogger(name_model)

        best_score = 0.
        best_loss = np.inf

        for epoch in range(start_epoch, cfg.NUM_EPOCHS):
            # train model
            avg_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, metric_logger,  device, epoch)
            # evaluate model
            avg_val_loss, preds = eval_one_epoch(model, criterion, val_dataloader, metric_logger, device, epoch)
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            # epoch roc_auc score
            auc_score = roc_auc_score(val_labels, preds)
            metric_logger.log(avg_loss, avg_val_loss, auc_score)
            logger.info(f"Epoch:{epoch} | avg_train_loss:{avg_loss:.4f} | avg_val_loss:{avg_val_loss:.4f}")
            logger.info(f"------ROC_AUC_SCORE: {auc_score}")

            if auc_score > best_score:
                best_score = auc_score
                save_path = cfg.OUTPUT_DIR + f"{name_model}_fold_{fold}_best_roc_auc.pth.tar"
                save_checkpoint(save_path, model, preds, epoch)
                logger.info(f"Found the best roc_auc_score, save model to {save_path}")
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                save_path = cfg.OUTPUT_DIR + f"{name_model}_fold_{fold}_best_val_loss.pth.tar"
                save_checkpoint(save_path, model, preds, epoch)
                logger.info(f"Found the best validation loss, save model to {save_path}")

            val_folds['preds'] = torch.load(cfg.OUTPUT_DIR + f"{name_model}_fold_{fold}_best_val_loss.pth.tar",
                                            map_location=torch.device("cpu"))['preds']

        _oof_df = val_folds
        oof_df = pd.concat([oof_df, _oof_df])

        logger.info(f'--------------------[{fold}-of-{cfg.N_FOLD}--------------------[')
        # Best epoch result
        print_result(_oof_df)

        del model
        gc.collect()

    # best from all folds
    logger.info("=========== CROSS-VALIDATION SCORE ===========")
    print_result(oof_df)
    oof_df.to_csv(cfg.OUTPUT_DIR + 'oof_df.csv', index=False)
