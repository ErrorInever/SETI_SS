import logging
import wandb
from config import cfg


logger = logging.getLogger(__name__)


class MetricLogger:
    """Metric class"""
    def __init__(self, version):
        self.train_step = 0
        self.val_step = 0
        if cfg.RESUME_ID:
            wandb_id = cfg.WANDB_ID
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, project=cfg.PROJECT_NAME, name=cfg.PROJECT_VERSION_NAME, resume=True)
        wandb.config.update({
            'model_version': version,
            'apex': cfg.USE_APEX,
            'learning_rate': cfg.LEARNING_RATE,
            'batch_sizes': cfg.BATCH_SIZE,
            'weight_decay': cfg.WEIGHT_DECAY,
            'betas': cfg.BETAS,
            'scheduler_type': cfg.SCHEDULER_VERSION,
            'factor': cfg.FACTOR,
            'patience': cfg.PATIENCE,
            'eps': cfg.EPS,
            't_max': cfg.T_MAX,
            't_0': cfg.T_0,
            'min_lr': cfg.MIN_LR,
        })

    def log(self, avg_train_loss, avg_val_loss, auc_score):
        wandb.log({
            'avg_epoch_train_loss': avg_train_loss,
            'avg_epoch_val_loss': avg_val_loss,
            'roc_auc_score': auc_score
        })

    def train_loss_batch(self, t_loss, epoch, num_batches, batch_idx):
        self.train_step += MetricLogger._step(epoch, num_batches, batch_idx)
        wandb.log({
            'batch_train_loss': t_loss,
            't_step': self.train_step
        })

    def val_loss_batch(self, v_loss, epoch, num_batches, batch_idx):
        self.val_step += MetricLogger._step(epoch, num_batches, batch_idx)
        wandb.log({
            'batch_val_loss': v_loss,
            'v_step': self.val_step
        })

    @staticmethod
    def _step(epoch, num_batches, batch_idx):
        return epoch * num_batches + batch_idx
