import logging
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# NAMES
__C.PROJECT_NAME = "SETI E.T."
__C.RUN_NAME = "SETI_BASELINE"
__C.MODEL_TYPE = None
# GLOBAL
__C.DEVICE = None
__C.IMG_CHANNELS = 1
__C.SEED = 42
__C.USE_APEX = True
__C.NUM_CLASSES = 1
__C.IMG_SIZE = 224
__C.NUM_EPOCHS = 10
__C.LEARNING_RATE = 0.00003366
__C.BATCH_SIZE = 64
# OPTIMIZER ADAM
__C.WEIGHT_DECAY = 1e-6
__C.BETAS = (0.0, 0.99)
# SCHEDULER
# ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
__C.SCHEDULER_VERSION = 'CosineAnnealingWarmRestarts'
__C.FACTOR = 0.2
__C.PATIENCE = 4
__C.EPS = 1e-6
__C.T_MAX = 6
__C.MIN_LR = 1e-6
__C.T_0 = 10     # scheduler restarts after Ti epochs.
# DATA
__C.NUM_FOLDS = 4
__C.FOLD_LIST = [i for i in range(__C.NUM_FOLDS)]
# DISPLAY RESULTS
__C.LOSS_FREQ = 10
# WANDB
__C.WANDB_ID = None
__C.RESUME_ID = None
# PATHS
__C.DATA_FOLDER = None
__C.OUTPUT_DIR = './'
__C.MODEL_DIR = None

# Init logger
logger = logging.getLogger()
c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)
