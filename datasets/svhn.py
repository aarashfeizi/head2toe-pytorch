from torchvision.datasets import SVHN as tSVHN

class SVHN(tSVHN):
    def __init__(self, root, split='train',
                    transform=None,
                    target_transform=None,
                    download=False):
        super(SVHN, self).__init__(root, split,
                    transform,
                    target_transform,
                    download=True)
        if split == 'train':
            self.data = self.data[:1000]
            self.labels = self.labels[:1000]