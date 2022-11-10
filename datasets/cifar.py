from torchvision.datasets import CIFAR100 as tCIFAR100
from torchvision.datasets import CIFAR10 as tCIFAR10

class CIFAR100(tCIFAR100):
    def __init__(self,
        root: str,
        transforms=None,
        transform=None,
        target_transform=None):
        super(CIFAR100, self).__init__(root=root, 
                                    transforms=transforms, 
                                    transform=transform, 
                                    target_transform=target_transform)

        self.data = self.data[:1000]
        self.targets = self.targets[:1000] 


class CIFAR10(tCIFAR10):
    def __init__(self,
        root: str,
        transforms=None,
        transform=None,
        target_transform=None):
        super(CIFAR10, self).__init__(root=root, 
                                    transforms=transforms, 
                                    transform=transform, 
                                    target_transform=target_transform)

        self.data = self.data[:1000]
        self.targets = self.targets[:1000] 