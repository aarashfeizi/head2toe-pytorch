import torchvision.datasets as tdatasets
from .cifar import CIFAR100, CIFAR10
from .eurosat import EuroSAT
from .flowers102 import Flowers102
from .dtd import DTD
from .caltech101 import Caltech101
from .svhn import SVHN

DATASETS = {'cifar10': CIFAR10,
            'cifar100': CIFAR100,
            'caltech101': Caltech101,
            'svhn': SVHN,
            'flowers102': Flowers102,
            'kitti': Kitti,
            'dtd': DTD,
            }


def get_dataset(args, transform=None, mode='train'): 
    """
        args: 
        mode: train or test
    """
    dataset_name = args.dataset
    dataset_root = args.dataset_path
    if "cifar" in dataset_name:
        d = DATASETS[dataset_name](root=dataset_root, train=(mode == 'train'), transform=transform)
    elif dataset_name == 'caltech101':
        d = DATASETS[dataset_name](root=dataset_root, transform=transform)
    elif dataset_name == 'svhn':
        if mode != 'train':
            mode = 'test'
        d = DATASETS[dataset_name](root=dataset_root, split=mode, transform=transform)

    return d