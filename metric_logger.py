import logging
import wandb
from config import cfg


logger = logging.getLogger(__name__)


class MetricLogger:
    """Metric class"""
    def __init__(self, fold, group_name='Default group name'):
        params = {
            'model_type': cfg.MODEL_TYPE,
            'seed': cfg.SEED,
            'epochs': cfg.NUM_EPOCHS,
            'folds': cfg.NUM_FOLDS,
            'img_size': cfg.IMG_SIZE,
            'apex': cfg.USE_APEX,
            'lr': cfg.LEARNING_RATE,
            'batch_size': cfg.BATCH_SIZE,
            'weight_decay': cfg.WEIGHT_DECAY,
            'betas': cfg.BETAS,
            'scheduler_type': cfg.SCHEDULER_VERSION,
            'factor': cfg.FACTOR,
            'patience': cfg.PATIENCE,
            'eps': cfg.EPS,
            't_max': cfg.T_MAX,
            't_0': cfg.T_0,
            'min_lr': cfg.MIN_LR,
        }

        if cfg.RESUME_ID:
            wandb_id = cfg.WANDB_ID
        else:
            wandb_id = wandb.util.generate_id()

        self._run = wandb.init(
            id=wandb_id,
            project=cfg.PROJECT_NAME,
            config=params,
            group=group_name,
            name=f'Fold: {fold}',
            resume=True)

    def avg_log(self, train_avg_loss, val_avg_loss, score):
        wandb.log({
            'Epoch_train_loss': train_avg_loss,
            'Epoch_val_loss': val_avg_loss,
            'ROC_AUC': score
        })

    def train_loss(self, loss):
        wandb.log({'Train loss': loss})

    def val_loss(self, loss):
        wandb.log({'Val loss': loss})

    def finish(self):
        self._run.finish()
