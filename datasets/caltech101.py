from torchvision.datasets import Caltech101 as tCaltech101

TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20

class Caltech101(tCaltech101):
    def __init__(self, root,
                    target_type='category',
                    transform=None,
                    target_transform=None,
                    download=False):
        super(Caltech101, self).__init__(root,
                    target_type,
                    transform,
                    target_transform,
                    download=True)
                    
        self.index = self.index[:1000]
        self.y = self.y[:1000]
