import argparse
import torch
import logging
import os
import time
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import torch.optim as optim
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from data.dataset import SETIDataset
from torch.utils.data import DataLoader
from utils import (seed_everything, get_train_file_path, AverageMeter,
                   split_data_kfold, mix_up_data, loss_mix_up)
from config import cfg
from models.pretrained_models import get_model


def parse_args():
    parser = argparse.ArgumentParser(description='SETI E.T. hyperparams')
    parser.add_argument('--data_path', dest='data_path', help='Path to root dataset', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or cpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--run_name', dest='run_name', help='Run name of wandb', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle notebook, '
                                                              'input api key wandb', default=None, type=str)
    parser.add_argument('--model_type', dest='model_type', help='Name model', default='nf_net', type=str)
    parser.add_argument('--nf_net_version', dest='nf_net_version', help='Version of NF_NET', default=None, type=str)
    parser.add_argument('--efficient_version', dest='efficient_version', help='Version of Efficient', default=None,
                        type=str)

    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, device):
    """
    Train one epoch
    :param model: ``instance of nn.Module``, model
    :param optimizer: ``instance of optim.object``, optimizer
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
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
            wandb.log({'Train loss': losses.val})

        loop.set_postfix(
            loss=losses.val
        )

    return losses.avg


def eval_one_epoch(model, criterion, dataloader, device):
    """
    Evaluate one epoch
    :param model: ``instance of nn.Module``, model
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
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
            wandb.log({'Val loss': losses.val})

        loop.set_postfix(
            loss=losses.val
        )

    predictions = np.concatenate(preds)
    return losses.avg, predictions


def train_fn():
    # Paths and create DataFrames
    sub_train_labels = os.path.join(cfg.DATA_FOLDER, 'train_labels.csv')
    train_df = pd.read_csv(sub_train_labels)                            # Targets corresponding (by id)

    # Add img file paths to dataframe
    train_df['file_paths'] = train_df['id'].apply(get_train_file_path)
    # Stratified K-Fold, split train data to K folds
    train_df = split_data_kfold(train_df, cfg.NUM_FOLDS)

    train_idxs = train_df[train_df['fold'] != 0].index
    val_idxs = train_df[train_df['fold'] == 0].index
    train_folds = train_df.loc[train_idxs].reset_index(drop=True)
    val_folds = train_df.loc[val_idxs].reset_index(drop=True)

    default_params = {
        'optimizer': 'adam',
        'learning_rate': 1e-2,
        'weight_decay': 1e-6
    }
    wandb_id = wandb.util.generate_id()
    wandb.init(id=wandb_id, project='SETI-Sweep', config=default_params, name=f'RUN : {wandb_id}')
    config = wandb.config

    optimizer_type = config.optimizer
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay

    train_dataset = SETIDataset(train_folds, transform=True)
    val_dataset = SETIDataset(val_folds, resize=True)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True,
                                drop_last=False)

    model = get_model(model_type=cfg.MODEL_TYPE, version='l0', pretrained=True).to(cfg.DEVICE)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay,
                               amsgrad=False)
    elif optimizer_type == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    else:
        raise ValueError('No optimizer type')

    criterion = nn.BCEWithLogitsLoss()

    logger.info(f"=Data=\ntrain dataset length: {len(train_dataset)}\nval dataset length: {len(val_dataset)}")
    logger.info(f"=Config=\noptimizer_type: {optimizer_type}\nlearning_rate: {learning_rate}\nbatch_size: {batch_size}")
    # train one epoch
    train_avg_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, cfg.DEVICE)
    # eval one epoch
    val_avg_loss, preds = eval_one_epoch(model, criterion, val_dataloader, cfg.DEVICE)
    # log score
    wandb.log({'Epoch_train_loss': train_avg_loss})
    wandb.log({'Epoch_val_loss': val_avg_loss})


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.device in ['gpu', 'cpu'], 'incorrect device type'
    assert args.model_type in ['nf_net', 'efficient', 'eca_nfnet'], 'incorrect model type, available models: ' \
                                                                    '[nf_net, efficient]'

    cfg.DATA_FOLDER = args.data_path
    logger = logging.getLogger('hyperparam')

    if args.out_dir:
        cfg.OUTPUT_DIR = args.out_dir
    if args.device == 'gpu':
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        cfg.DEVICE = 'cpu'
    if args.run_name:
        cfg.RUN_NAME = args.run_name
    if args.wandb_key:
        os.environ["WANDB_API_KEY"] = args.wandb_key
    if args.model_type:
        cfg.MODEL_TYPE = args.model_type

    if args.nf_net_version:
        model_version = args.nf_net_version
    elif args.efficient_version:
        model_version = args.efficient_version
    else:
        model_version = 'b0'

    logger.info(f'==> Start {__name__} at {time.ctime()}')
    logger.info(f'==> Called with args: {args.__dict__}')
    logger.info(f'==> Config params: {cfg.__dict__}')
    logger.info(f'==> Using device:{args.device}')

    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'Epoch_val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'optimizer': {
                'values': ['adam', 'adamW']
            },
            'learning_rate': {
                "distribution": "uniform",
                "min": 0.00001,
                "max": 0.001
            },
            'weight_decay': {
                "distribution": "uniform",
                "min": 1e-6,
                "max": 1e-2
            },
        }
    }

    # Initialize the sweep and run
    sweep_id = wandb.sweep(sweep_config, project='SETI-Sweep')
    wandb.agent(sweep_id, train_fn, count=20)
