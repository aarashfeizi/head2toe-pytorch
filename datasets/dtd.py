from torchvision.datasets import DTD as tDTD

class DTD(tDTD):
    def __init__(self, root, split='train',
                    partition=1,
                    transform=None,
                    target_transform=None,
                    download=False):
        if split != 'train':
            split = 'test'
        super(DTD, self).__init__(root, split,
                    partition,
                    transform,
                    target_transform,
                    download=True)
        if split == 'train':
            self._image_files = self._image_files[:1000]
            self._labels = self._labels[:1000]
