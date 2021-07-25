import random
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from config import cfg

logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def show_cadence(idx, path, label):
    cadence = np.load(path).astype(np.float32)

    fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(16, 10))
    fig.suptitle(f'CADENCE_ID:{idx}   TARGET:{label}', fontsize=18)
    for i in range(6):
        ax[i].imshow(cadence[i], interpolation='nearest', aspect='auto')
        ax[i].text(5, 100, ["ON", "OFF"][i % 2], bbox={'facecolor': 'white'})
        ax[i].get_xaxis().set_visible(False)
    plt.show()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_data_kfold(df, k=4):
    """
    Split data: Stratified K-Fold
    :param df: DataFrame object
    :param k: ``int``, How many folds the dataset is going to be divided
    :return: Divided DataFrame object
    """
    fold = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.SEED)
    for n, (train_idx, val_idx) in enumerate(fold.split(df, df['target'])):
        df.loc[val_idx, 'fold'] = int(n)
    df['fold'] = df['fold'].astype(int)
    logger.info(f'==> Split K-Fold')
    logger.info(df.groupby(['fold', 'target']).size())
    return df


def get_train_file_path(image_id):
    """
    Add new column with file path to train images
    :param image_id: img id
    """
    return "{}/train/{}/{}.npy".format(cfg.DATA_ROOT, image_id[0], image_id)


def get_test_file_path(image_id):
    """
    Add new column with file path to test images
    :param image_id: img id
    """
    return "{}/test/{}/{}.npy".format(cfg.DATA_ROOT, image_id[0], image_id)


def get_scheduler(optimizer):
    if cfg.SCHEDULER_VERSION == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.FACTOR, patience=cfg.PATIENCE, verbose=True,
                                      eps=cfg.EPS)
    elif cfg.SCHEDULER_VERSION == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_MAX, eta_min=cfg.MIN_LR, last_epoch=-1)
    elif cfg.SCHEDULER_VERSION == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.MIN_LR, last_epoch=-1)
    else:
        raise ValueError('SCHEDULER WAS NOT DEFINED')
    return scheduler


def print_result(result_df):
    preds = result_df['preds'].values
    labels = result_df['target'].values
    score = roc_auc_score(labels, preds)
    logger.info(f'Score: {score:<.4f}')


def save_checkpoint(save_path, model, optimizer, lr, preds):
    torch.save({
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': lr,
        'preds': preds,
    }, save_path)


def mix_up_data(x, y, alpha=0.1, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.shape[0]
    if use_cuda:
        idx = torch.randperm(batch_size).cuda()
    else:
        idx = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[idx, :]
    y_a, y_b = y,  y[idx]

    return mixed_x, y_a, y_b, lam


def loss_mix_up(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
