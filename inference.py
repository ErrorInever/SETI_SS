import argparse
import os

import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import SETIDataset
from utils import seed_everything, get_test_file_path
from config import cfg
from models.effecient import EfficientNet


def parse_args():
    parser = argparse.ArgumentParser(description='SLE-GC-GAN')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--load_model', dest='load_model', help='Path to model.pth.tar', default=None, type=str)
    parser.add_argument('--model_version', dest='model_version', help='Specified version of model', default=None,
                    type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or tpu. Default CPU', default='cpu', type=str)
    parser.print_help()
    return parser.parse_args()


def inference(model, dataloader, device):
    model.eval()
    probs = []
    loop = tqdm(dataloader, leave=True)
    for batch_idx, (image, _) in enumerate(loop):
        image = image.to(device)
        with torch.no_grad():
            y_pred = model(image)
        probs.append(y_pred.sigmoid().cpu().numpy())
    probs = np.concatenate(probs)
    return probs


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.device in ['cpu', 'gpu', 'tpu'], 'incorrect device type'
    assert args.load_model, 'load_model not specified'
    assert args.model_version, 'model_version not specified'
    assert args.model_version in cfg.EFFICIENT_VERSIONS, 'incorrect model version'

    cfg.DATA_ROOT = args.data_path

    logger = logging.getLogger('main')

    if args.device == 'gpu':
        device = torch.device('cuda')
    model = EfficientNet(args.model_version, num_classes=cfg.NUM_CLASSES, in_channels=cfg.IMG_CHANNELS).to(device)
    cp = torch.load(args.load_model, map_location=args.device)
    model.load_state_dict(cp['model'])
    logger.info(f"model loaded from {args.load_model}")

    data_root = args.data_path
    test_path = os.path.join(data_root, 'sample_submission.csv')
    test_df = pd.read_csv(test_path)
    test_df['file_path'] = test_df['id'].apply(get_test_file_path)

    test_dataset = SETIDataset(test_df, resize=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True)

    predictions = inference(model, test_dataloader, device)

    test_df['target'] = predictions
    test_df[['id', 'target']].to_csv('submission.csv', index=False)
