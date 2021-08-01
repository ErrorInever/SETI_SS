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
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.serialization as xser
from tqdm import tqdm
from data.dataset import SETIDataset
from torch.utils.data import DataLoader
from utils import (seed_everything, get_train_file_path, get_scheduler, AverageMeter,
                   split_data_kfold, mix_up_data, loss_mix_up)
from config import cfg
from sklearn.metrics import roc_auc_score
from models.pretrained_models import get_model

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='SETI E.T.')
    parser.add_argument('--data_path', dest='data_path', help='Path to root dataset', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--ckpt', dest='ckpt', help='path to model ckpt.pth.tar', default=None, type=str)
    parser.add_argument('--run_name', dest='run_name', help='Run name of wandb', default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='Wand metric id for resume train', default=None, type=str)
    parser.add_argument('--wandb_key', dest='wandb_key', help='Use this option if you run it from kaggle notebook, '
                                                              'input api key wandb', default=None, type=str)
    parser.add_argument('--one_epoch', dest='one_epoch', help='Train one epoch', action='store_true')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Number of epochs', default=None, type=int)
    parser.add_argument('--num_folds', dest='num_folds', help='Number of folds', default=None, type=int)
    parser.add_argument('--model_type', dest='model_type', help='Name model', default='nf_net', type=str)
    parser.add_argument('--nf_net_version', dest='nf_net_version', help='Version of NF_NET', default=None, type=str)
    parser.add_argument('--efficient_version', dest='efficient_version', help='Version of Efficient', default=None,
                        type=str)

    parser.print_help()
    return parser.parse_args()


def train_one_epoch(model, optimizer, criterion, dataloader, scheduler, device):
    """
    Train one epoch
    :param model: ``instance of nn.Module``, model
    :param optimizer: ``instance of optim.object``, optimizer
    :param criterion: ``nn.Object``, loss function
    :param dataloader: ``instance of Dataloader``, dataloader on train data
    :param metric_logger: ``instance of MetricLogger``, helper class
    :param device: ``str``, XLA device
    :return: ``float``, average loss on epoch
    """
    model.train()
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        img, label_a, label_b, lam = mix_up_data(img, label, use_cuda=False)
        img = img.to(device)
        label_a = label_a.to(device)
        label_b = label_b.to(device)

        optimizer.zero_grad()
        y_preds = model(img)
        loss = loss_mix_up(criterion, y_preds.view(-1), label_a, label_b, lam)
        loss.backward()
        xm.optimizer_step(optimizer)
        scheduler.step()

        del img, label_a, label_b, y_preds # delete for memory conservation
        gc.collect()

    # since the loss is on all 8 cores, reduce the loss values and print the average
    loss_reduced = xm.mesh_reduce('loss_reduce', loss, lambda x: sum(x) / len(x))
    # master_print will only print once (not from all 8 cores)
    xm.master_print(f'bi={batch_idx}, train loss={loss_reduced}')
    model.eval()


def eval_one_epoch(model, criterion, dataloader, device):
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

    fin_labels = []
    fin_preds = []
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (img, label) in enumerate(loop):
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            y_preds = model(img)
        loss = criterion(y_preds.view(-1), label)

        targets_np = label.cpu().detach().numpy().tolist()
        outputs_np = y_preds.cpu().detach().numpy().tolist()
        fin_labels.extend(targets_np)
        fin_preds.extend(outputs_np)
        del img, label, y_preds, targets_np, outputs_np
        gc.collect()    # delete for memory conservation

    preds, labels = np.array(fin_preds), np.array(fin_labels)
    loss = criterion(torch.tensor(preds), torch.tensor(labels).unsqueeze(1).float())
    loss_reduced = xm.mesh_reduce('loss_reduce', loss, lambda x: sum(x) / len(x))
    # master_print will only print once (not from all 8 cores)
    xm.master_print(f'val. loss={loss_reduced}')

    o_reduced = xm.mesh_reduce('preds_reduce', torch.tensor(preds).to(device), torch.cat)
    t_reduced = xm.mesh_reduce('targets_reduce', torch.tensor(labels).to(device), torch.cat)

    # metric calculation
    auc = roc_auc_score(t_reduced.cpu(), o_reduced.cpu())
    xm.master_print(f'val. auc = {auc}')



def train_fn(rank, params):
    fold = params['fold']
    mx = params['model']
    train_df = params['df']

    xm.master_print(f'========== Fold: [{fold + 1} of {len(cfg.FOLD_LIST)}] ==========')
    # Each fold divide on train and validation datasets
    train_idxs = train_df[train_df['fold'] != fold].index
    val_idxs = train_df[train_df['fold'] == fold].index
    train_folds = train_df.loc[train_idxs].reset_index(drop=True)
    val_folds = train_df.loc[val_idxs].reset_index(drop=True)
    val_labels = val_folds['target'].values     # list of validation dataset targets of current fold
    # Define devices, they will be different for each core on the TPU
    device = xm.xla_device()
    xm.set_rng_state(cfg.SEED, device)
    # Define dataset
    train_dataset = SETIDataset(train_folds, transform=True)
    val_dataset = SETIDataset(val_folds, resize=True)
    # Special samplers needed for distributed/multi-core
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),   # divide dataset among this many replicas
        rank=xm.get_ordinal(),              # which replica/device/core
        shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    # Define dataloaders with samplers
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, num_workers=4,
                                  sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, num_workers=4, sampler=val_sampler)
    # Puts the data onto the current TPU core
    train_loader = pl.MpDeviceLoader(train_dataloader, device)
    val_loader = pl.MpDeviceLoader(val_dataloader, device)
    # Clear memory 
    del train_sampler, val_sampler
    gc.collect()
    # Put model to current TPU core
    model = mx.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # often a good idea to scale the learning rate by number of cores
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE * xm.xrt_world_size(),
                           weight_decay=cfg.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer)

    xm.master_print('Start Training now...')
    for epoch in range(cfg.NUM_EPOCHS):
        # Train model
        train_one_epoch(model, optimizer, criterion, train_loader, scheduler, device)
        # Evaluate model
        eval_one_epoch(model, criterion, val_loader, device)

        gc.collect()

    xm.rendezvous('save_model')
    xm.master_print('save model')
    xm.save(model.state_dict(), f'xla_trained_model_{epoch}_epochs_fold_{fold}.pth')


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.model_type in ['nf_net', 'efficient', 'eca_nfnet'], 'incorrect model type, available models: ' \
                                                                    '[nf_net, efficient]'
    cfg.DATA_FOLDER = args.data_path
    logger = logging.getLogger('tpu train')

    # Use float16 for TPU
    os.environ['XLA_USE_BF16'] = "1"
    os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'

    if args.out_dir:
        cfg.OUTPUT_DIR = args.out_dir
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

    if args.nf_net_version:
        model_version = args.nf_net_version
    elif args.efficient_version:
        model_version = args.efficient_version
    else:
        model_version = 'b0'

    logger.info(f'==> Start {__name__} at {time.ctime()}')
    logger.info(f'==> Called with args: {args.__dict__}')
    logger.info(f'==> Config params: {cfg.__dict__}')

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
    MX = xmp.MpModelWrapper(get_model(model_type=cfg.MODEL_TYPE, version=model_version, pretrained=True))
    oof_df = pd.DataFrame()
    # Train loop
    PARAMS = {
        'model': MX,
        'df': train_df,
        'fold': None
    }
    for fold in range(cfg.NUM_FOLDS):
        PARAMS['fold'] = fold
        xmp.spawn(train_fn, args=(PARAMS,), nprocs=8, start_method='fork')


