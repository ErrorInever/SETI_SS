import logging
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# NAMES
__C.PROJECT_NAME = "SETI"
__C.PROJECT_VERSION_NAME = "BASELINE"

# GLOBAL
__C.IMG_CHANNELS = 1
__C.SEED = 44
__C.USE_APEX = True
__C.NUM_CLASSES = 1
__C.EFFICIENT_VERSIONS = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']
__C.IMG_SIZE = 224
__C.NUM_EPOCHS = 4
__C.LEARNING_RATE = 1e-4
__C.BATCH_SIZE = 64

# OPTIMIZER ADAM
__C.WEIGHT_DECAY = 1e-6
__C.BETAS = (0.0, 0.99)

# SCHEDULER
__C.SCHEDULER_VERSION = 'CosineAnnealingWarmRestarts'   # [ReduceLROnPlateau, # CosineAnnealingLR, # CosineAnnealingWarmRestarts]
__C.FACTOR = 0.2
__C.PATIENCE = 4
__C.EPS = 1e-6
__C.T_MAX = 6
__C.MIN_LR = 1e-6
__C.T_0 = 10     # scheduler restarts after Ti epochs.
# DATA
__C.N_FOLD = 4
__C.TRN_FOLD = [i for i in range(__C.N_FOLD)]

__C.SAVE = True
# DISPLAY RESULTS
__C.WANDB_ID = None
__C.RESUME_ID = None
__C.SAVE_EPOCH_FREQ = 10
__C.LOAD_MODEL = False
__C.PRETRAINED_MODEL = False
__C.LOG_FREQ = 20

__C.DATA_ROOT = None
__C.OUTPUT_DIR = './'

# Init logger
logger = logging.getLogger()
c_handler = logging.StreamHandler()

c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)
logger.setLevel(logging.INFO)