import argparse
import os
import time
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


def parse_args():
    parser = argparse.ArgumentParser(description='SETI E.T. Inference')
    parser.add_argument('--data_path', dest='data_path', help='Path to dataset folder', default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='Path where to save files', default=None, type=str)
    parser.add_argument('--model_dir', dest='model_dir', help='path where models stores', default=None, type=str)
    parser.add_argument('--model_type', dest='model_type', help='Name model', default='nf_net', type=str)
    parser.add_argument('--device', dest='device', help='Use device: gpu or cpu. Default use gpu if available',
                        default='gpu', type=str)
    parser.add_argument('--oof', dest='oof', help='path to oof score', default=None, type=str)
    parser.add_argument('--efficient_version', dest='efficient_version', help='Version model of efficient',
                        default='b0', type=str)
    parser.add_argument('--nf_net_version', dest='nf_net_version', help='Version of NF_NET', default=None, type=str)
    parser.print_help()
    return parser.parse_args()

def inference(model, states, dataloader, device):
    """
    Get average probabilities of all models
    :param model: ``instance of nn.Module``, model
    :param states: ``List([instance of nn.Module])``, list states of models
    :param dataloader: ``instance of Dataloader``, dataloader on train data
    :param device: ``str``, cpu or gpu
    :return: ``List([Float])``, average probabilities of all models
    """
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
    assert args.model_dir, 'model path not specified'
    assert args.device in ['gpu', 'cpu'], 'incorrect device type'
    assert args.model_type in ['nf_net', 'efficient'], 'incorrect model type, available models: [nf_net, efficient]'

    cfg.DATA_FOLDER = args.data_path
    cfg.MODEL_DIR = args.model_dir
    logger = logging.getLogger('inference')

    if args.out_dir:
        cfg.OUTPUT_DIR = args.out_dir
    if args.device == 'gpu':
        cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif args.device == 'cpu':
        cfg.DEVICE = 'cpu'
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
    logger.info(f'==> Using device:{args.device}')

    # Display cross validation score
    if args.oof:
        oof = pd.read_csv(args.oof)
        logger.info(f"==> Loaded cross validation score")
        print_result(oof)

    # Paths and create DataFrames
    test_path = os.path.join(cfg.DATA_FOLDER, 'sample_submission.csv')
    test_df = pd.read_csv(test_path)
    test_df['file_paths'] = test_df['id'].apply(get_test_file_path)

    # Load model
    model = get_model(model_type=cfg.MODEL_TYPE, version=model_version, pretrained=False).to(cfg.DEVICE)
    # Load states of each fold
    states = [torch.load(
        os.path.join(cfg.MODEL_DIR, f"{cfg.MODEL_TYPE}_fold_{fold}_best_val_loss.pth.tar")) for fold in cfg.FOLD_LIST]
    # Define test dataset & dataloader
    test_dataset = SETIDataset(test_df, resize=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, num_workers=2, pin_memory=True)
    # Get predictions on test dataset
    predictions = inference(model, states, test_dataloader, cfg.DEVICE)
    # Make submission
    test_df['target'] = predictions
    test_df[['id', 'target']].to_csv('submission.csv', index=False)

    logger.info(f"==> Test done. Save submission")
