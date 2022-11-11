import torchvision.datasets as tdatasets
from .cifar import CIFAR100, CIFAR10

DATASETS = {'cifar10': CIFAR10,
            'cifar100': CIFAR100,
            'caltech101': tdatasets.Caltech101,
            'svhn': tdatasets.SVHN}


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