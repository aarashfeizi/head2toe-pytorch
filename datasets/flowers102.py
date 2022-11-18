from torchvision.datasets import Flowers102 as tFlowers102

class Flowers102(tFlowers102):
    def __init__(self, root, split='train',
                    transform=None,
                    target_transform=None,
                    download=False):
        if split != 'train':
            split = 'test'
        super(Flowers102, self).__init__(root, split,
                    transform,
                    target_transform,
                    download)
        if split == 'train':
            self._image_files = self._image_files[:1000]
            self._labels = self._labels[:1000]
