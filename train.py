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
from data.dataset import SETIDataset
from torch.utils.data import DataLoader
from utils import (seed_everything, get_train_file_path, get_scheduler, AverageMeter,
                   split_data_kfold, print_result, save_checkpoint, mix_up_data, loss_mix_up)
from config import cfg
from metric_logger import MetricLogger
from sklearn.metrics import roc_auc_score
from models.pretrained_models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='SETI E.T.')
    parser.add_argument('--data_path', dest='data_path', help='Path to root dataset', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or cpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--ckpt', dest='ckpt', help='path to model ckpt.pth.tar', default=None, type=str)
    parser.add_argument('--run_name', dest='run_name', help='Run name of wandb', default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume train', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle notebook, '
                                                              'input api key wandb', default=None, type=str)
    parser.add_argument('--one_epoch', dest='one_epoch', help='Train one epoch', action='store_true')
    parser.add_argument('--one_fold', dest='one_fold', help='Train one_fold', action='store_true')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Number of epochs', default=None, type=int)
    parser.add_argument('--num_folds', dest='num_folds', help='Number of folds', default=None, type=int)
    parser.add_argument('--model_type', dest='model_type', help='Name model', default='nf_net', type=str)
    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, metric_logger, device):
    """
    Train one epoch
    :param model: ``instance of nn.Module``, model
    :param optimizer: ``instance of optim.object``, optimizer
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
    :param metric_logger: ``instance of MetricLogger``, helper class
    :param device: ``str``, cpu or gpu
    :return: ``float``, average loss on epoch
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

        if batch_idx % cfg.LOSS_FREQ == 0:
            metric_logger.train_loss(losses.val)

        loop.set_postfix(
            loss=losses.val
        )

    return losses.avg


def eval_one_epoch(model, criterion, dataloader, metric_logger, device):
    """
    Evaluate one epoch
    :param model: ``instance of nn.Module``, model
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
    :param metric_logger: ``instance of MetricLogger``, helper class
    :param device: ``str``, cpu or gpu
    :return: ``List([float, list])``, average loss of epoch, list predictions
    """
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
        # record acc
        preds.append(y_preds.sigmoid().to('cpu').numpy())

        if batch_idx % cfg.LOSS_FREQ == 0:
            metric_logger.val_loss(losses.val)

        loop.set_postfix(
            loss=losses.val
        )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.device in ['gpu', 'cpu'], 'incorrect device type'
    assert args.model_type in ['nf_net', 'efficient'], 'incorrect model type, available models: [nf_net, efficient]'
    cfg.DATA_FOLDER = args.data_path
    logger = logging.getLogger('train')

    if args.out_dir:
        cfg.OUTPUT_DIR = args.out_dir
    if args.device == 'gpu':
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        cfg.DEVICE = 'cpu'
    if args.run_name:
        cfg.RUN_NAME = args.run_name
    if args.wandb_id:
        cfg.RESUME_ID = args.wandb_id
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    if args.num_epochs:
        cfg.NUM_EPOCHS = args.num_epochs
    if args.num_folds:
        cfg.NUM_FOLDS = args.num_folds
    if args.model_type:
        cfg.MODEL_TYPE = args.model_type
    if args.one_epoch:
        cfg.NUM_EPOCHS = 1
    if args.one_fold:
        cfg.NUM_FOLDS = 1

    logger.info(f'==> Start {__name__} at {time.ctime()}')
    logger.info(f'==> Called with args: {args.__dict__}')
    logger.info(f'==> Config params: {cfg.__dict__}')
    logger.info(f'==> Using device:{args.device}')

    # Paths and create DataFrames
    sub_path = os.path.join(cfg.DATA_FOLDER, 'sample_submission.csv')
    sub_train_labels = os.path.join(cfg.DATA_FOLDER, 'train_labels.csv')

    old_leaky_data = os.path.join(cfg.DATA_FOLDER, 'old_leaky_data')    # Full pre-relaunch data, including test labels
    train_data = os.path.join(cfg.DATA_FOLDER, 'train')                 # A training set of cadence snippet files
    test_data = os.path.join(cfg.DATA_FOLDER, 'test')                   # The test set cadence snippet files
    sample_sub = pd.read_csv(sub_path)                                  # A sample submission file in the correct format
    train_df = pd.read_csv(sub_train_labels)                            # Targets corresponding (by id)

    # Add img file paths to dataframe
    train_df['file_paths'] = train_df['id'].apply(get_train_file_path)
    # Stratified K-Fold, split train data to K folds
    train_df = split_data_kfold(train_df, cfg.NUM_FOLDS)
    # out of fold (predictions), for display results
    oof_df = pd.DataFrame()
    # Train loop
    for fold in range(cfg.NUM_FOLDS):
        logger.info(f'========== Fold: [{fold} of {len(cfg.FOLD_LIST)}] ==========')
        # Each fold divide on train and validation datasets
        train_idxs = train_df[train_df['fold'] != fold].index
        val_idxs = train_df[train_df['fold'] == fold].index
        train_folds = train_df.loc[train_idxs].reset_index(drop=True)
        val_folds = train_df.loc[val_idxs].reset_index(drop=True)
        val_labels = val_folds['target'].values     # list of validation dataset targets of current fold
        # Define dataset and dataloader
        train_dataset = SETIDataset(train_folds, transform=True)
        val_dataset = SETIDataset(val_folds, resize=True)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2,
                                      pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True,
                                    drop_last=False)
        # Define optimizer and pretrained model or load from previous checkpoint
        model = get_model(model_name=cfg.MODEL_TYPE, pretrained=True).to(cfg.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY,
                                   amsgrad=False)
        # Load checkpoint
        if args.ckpt:
            try:
                state = torch.load(args.ckpt, map_location=cfg.DEVICE)
                model.load_state_dict(state['model'])
                optimizer.load_state_dict(state['opt'])
                for param_group in optimizer.param_groups:
                    param_group["lr"] = state['lr']
                logger.info("Loaded checkpoint")
            except Exception:
                logger.error("Fail to load model")
                raise ValueError
        else:
            logger.info(f"==> Load default pretrained model: {cfg.MODEL_TYPE}")

        # Scheduler
        scheduler = get_scheduler(optimizer)
        # Losses
        criterion = nn.BCEWithLogitsLoss()
        # Metrics
        metric_logger = MetricLogger(fold, job_type='Train')

        best_score = 0.
        best_loss = np.inf
        for epoch in range(cfg.NUM_EPOCHS):
            # Train model
            train_avg_loss = train_one_epoch(model, optimizer,criterion, train_dataloader, metric_logger, cfg.DEVICE)
            # Evaluate model
            val_avg_loss, preds = eval_one_epoch(model, criterion, val_dataloader, metric_logger, cfg.DEVICE)
            # Scheduler step
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_avg_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()
            # ROC AUC score
            score = roc_auc_score(val_labels, preds)
            metric_logger.avg_log(train_avg_loss, val_avg_loss, score)

            logger.info(f"Epoch:{epoch} | train_avg_loss:{train_avg_loss:.4f} | val_avg_loss:{val_avg_loss:.4f}")
            logger.info(f"====== ROC_AUC_SCORE: {score} ======")

            # Found the best roc auc score on current fold
            if score > best_score:
                best_score = score
                save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_score.pth.tar")
                save_checkpoint(save_path, model, optimizer, cfg.LEARNING_RATE, preds)
                logger.info(f"==> Found the best ROC_AUC score, save model to {save_path}")
            if val_avg_loss < best_loss:
                best_loss = val_avg_loss
                save_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar")
                save_checkpoint(save_path, model, optimizer, cfg.LEARNING_RATE, preds)

        val_folds['preds'] = torch.load(os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar"),
                                        map_location=torch.device("cpu"))['preds']
        # save predictions for CV
        oof_df = pd.concat([oof_df, val_folds])
        # display roc auc score on current fold
        logger.info(f"========== Fold: {fold} Result ==========")
        print_result(val_folds)
        # Reinitializing metric run
        metric_logger.finish()

    # Cross Validation score
    logger.info("==> Train done")
    logger.info("Cross validation score")
    print_result(oof_df)
    # Save CV result to csv file
    oof_df.to_csv(cfg.OUTPUT_DIR + 'oof_df.csv', index=False)
