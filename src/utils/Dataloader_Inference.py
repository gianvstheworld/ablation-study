import os
import cv2
import numpy as np
from torch.utils.data import Dataset

class TravNetDataset(Dataset):
    def __init__(self, rgb_path, depth_path, transform=None):
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.transform = transform

    def __len__(self):
        return 1  # Retorna 1 para indicar que há apenas uma amostra

    def __getitem__(self, idx):
        # Carrega a imagem RGB
        rgb_img = cv2.imread(self.rgb_path.filename, -1)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (424, 240), interpolation=cv2.INTER_AREA)
        rgb_img = self.transform(rgb_img)

        # Carrega a imagem de profundidade
        depth_img = cv2.imread(self.depth_path.filename, -1)
        depth_img = cv2.resize(depth_img, (424, 240), interpolation=cv2.INTER_AREA)
        depth_img = np.uint16(depth_img)
        depth_img = depth_img*10**-3
        depth_img = (depth_img-3.5235)/10.6645
        depth_img = np.expand_dims(depth_img, axis=2)
        depth_img = np.transpose(depth_img, (2, 0, 1))

        return rgb_img, depth_img