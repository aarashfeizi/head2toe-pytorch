from torchvision.datasets import CIFAR100 as tCIFAR100
from torchvision.datasets import CIFAR10 as tCIFAR10

class CIFAR100(tCIFAR100):
    def __init__(self,
        root: str,
        train,
        transform=None):
        super(CIFAR100, self).__init__(root=root, 
                                    train=train, 
                                    transform=transform)
        
        if train:
            self.limit = 1000
            self.data = self.data[:self.limit]
            self.targets = self.targets[:self.limit] 


class CIFAR10(tCIFAR10):
    def __init__(self,
        root: str,
        train,
        transform=None):
        super(CIFAR10, self).__init__(root=root, 
                                    train=train, 
                                    transform=transform)

        if train:
            self.limit = 1000
            self.data = self.data[:self.limit]
            self.targets = self.targets[:self.limit] 

        