from torchvision.datasets import Caltech101 as tCaltech101
import numpy as np
import os

TRAIN_SPLIT_PERCENT = 80
TEST_SPLIT_PERCENT = 20

class Caltech101(tCaltech101):
    def __init__(self, root,
                    target_type='category',
                    transform=None,
                    target_transform=None,
                    split='train',
                    download=False,
                    split_path='/scratch/caltech101/'):
        super(Caltech101, self).__init__(root,
                    target_type,
                    transform,
                    target_transform,
                    download=True)
        
        self.index = np.load(os.path.join(split_path, f"caltech101_{split}_index.npy"))
        self.y = np.load(os.path.join(split_path, f"caltech101_{split}_y.npy"))
        if split == 'train':
            self.index = self.index[:1000]
            self.y = self.y[:1000]
