from torchvision.datasets import SUN397 as tSUN397

class SUN397(tSUN397):
    def __init__(self, root, split='train',
                    transform=None,
                    target_transform=None,
                    download=False):
        if split != 'train':
            split = 'test'
        super(SUN397, self).__init__(root, split,
                    transform,
                    target_transform,
                    download=True)
        if split == 'train':
            self._image_files = self._image_files[:1000]
            self._labels = self._labels[:1000]