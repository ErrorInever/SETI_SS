import argparse
import os

import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import SETIDataset
from models.pretrained_models import get_model
from utils import seed_everything, get_test_file_path, print_result
from config import cfg
from models.effecient import EfficientNet


def parse_args():
    parser = argparse.ArgumentParser(description='SLE-GC-GAN')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--cur_model', dest='cur_model', help='inference specified model', default=None, type=str)
    parser.add_argument('--model_dir', dest='model_dir', help='dir where states stored', default=None, type=str)
    parser.add_argument('--load_model', dest='load_model', help='Path to model.pth.tar', default=None, type=str)
    parser.add_argument('--model_version', dest='model_version', help='Specified version of model', default=None,
                    type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or tpu. Default CPU', default='cpu', type=str)
    parser.print_help()
    return parser.parse_args()


def inference(model, states, dataloader, device):
    """Average probabilities"""
    model.to(device)
    loop = tqdm(dataloader, leave=True)
    probs = []
    for batch_idx, (img, _) in enumerate(loop):
        img = img.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state['model'])
            model.eval()
            with torch.no_grad():
                y_preds = model(img)
            avg_preds.append(y_preds.sigmoid().cpu().numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs


if __name__ == '__main__':
    seed_everything(cfg.SEED)
    args = parse_args()

    assert args.data_path, 'data path not specified'
    assert args.device in ['cpu', 'gpu', 'tpu'], 'incorrect device type'
    assert args.cur_model in ['wide_resnet50_2', 'efficientnet', 'nfnet_l0'], 'incorrect model name'

    if args.cur_model:
        model_name = args.cur_model

    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = cfg.OUTPUT_DIR

    cfg.DATA_ROOT = args.data_path

    logger = logging.getLogger('inference')

    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')

    oof = pd.read_csv(cfg.OUTPUT_DIR + 'oof_df.csv')
    logger.info("Loaded oof with score")
    print_result(oof)

    data_root = args.data_path
    test_path = os.path.join(data_root, 'sample_submission.csv')
    test_df = pd.read_csv(test_path)
    test_df['file_path'] = test_df['id'].apply(get_test_file_path)

    if args.cur_model:
        model_list = [model_name]
    else:
        model_list = ['wide_resnet50_2', 'efficientnet', 'nfnet_l0']

    for i, name_model in enumerate(model_list):
        logger.info(f"Start inference №{i} of {len(model_list)} - Current model name {name_model}")

        model = get_model(name_model, pretrained=False)
        states = [torch.load(model_dir+f"{name_model}_fold_{fold}_best_val_loss.pth.tar") for fold in cfg.TRN_FOLD]

        test_dataset = SETIDataset(test_df, resize=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True)

        predictions = inference(model, states, test_dataloader, device)

        # make submission
        test_df['target'] = predictions
        test_df[['id', 'target']].to_csv('submission.csv', index=False)
        test_df[['id', 'target']].head()

        logger.info(f"Inference done, save submission.csv")
