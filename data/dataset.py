import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from config import cfg


class SETIDataset(Dataset):
    """SETI dataset"""
    def __init__(self, df, transform=None):
        self.df = df
        self.cad_ids = df['id'].values
        self.cad_paths = df['file_path'].values
        self.cad_labels = df['target'].values

        self._transform = A.Compose([
            A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
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
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
        label = torch.tensor(self.cad_labels[idx]).float()
        return image, label

    @property
    def transform(self):
        return self._transform
