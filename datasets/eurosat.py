from torchvision.datasets import EuroSAT as tEuroSAT

TRAIN_SPLIT_PERCENT = 60
VALIDATION_SPLIT_PERCENT = 20
TEST_SPLIT_PERCENT = 20

class EuroSAT(tEuroSAT):
    def __init__(self, root,
                    transform=None,
                    target_transform=None,
                    download=False):
        super(EuroSAT, self).__init__(root,
                    transform,
                    target_transform,
                    download)
        self.samples = self.samples[:1000]
