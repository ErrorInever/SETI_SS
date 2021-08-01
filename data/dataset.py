import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from config import cfg


class SETIDataset(Dataset):
    """SETI dataset"""
    def __init__(self, df, transform=None, resize=None):
        self.df = df
        self.cad_ids = df['id'].values
        self.cad_paths = df['file_paths'].values
        self.cad_labels = df['target'].values

        self.train_transform = transform
        self.resize = resize

        self._transform = A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7),
            A.RandomBrightness(limit=0.6, p=0.5),
            A.Cutout(
                num_holes=10, max_h_size=12, max_w_size=12,
                fill_value=0, always_apply=False, p=0.5
            ),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.1, rotate_limit=0),
            # A.Normalize(mean=-0.0001, std=0.9055),
            ToTensorV2()
        ])
        self._resize_to_tensor = A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
            # A.Normalize(mean=-0.0002, std=0.8453),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        :param idx: ``int``
        :return: Tuple(Tensor([N, C, H, W]), Tensor([``float``]))
        """
        cad_path = self.cad_paths[idx]
        image = np.load(cad_path).astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        if self.train_transform:
            image = self.transform(image=image)['image']
        elif self.resize:
            image = self.resize_to_tensor(image=image)['image']
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
        label = torch.tensor(self.cad_labels[idx]).float()
        return image, label

    @property
    def transform(self):
        return self._transform

    @property
    def resize_to_tensor(self):
        return self._resize_to_tensor
