import torchvision.datasets as datasets

DATASETS = {'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'caltech101': datasets.Caltech101,
            'svhn': datasets.SVHN}


def get_dataset(args, trainsform=None, mode='train'): 
    """
        args: 
        mode: train or test
    """
    dataset_name = args.data.dataset
    dataset_root = args.data.dataset_path
    if "cifar" in dataset_name:
        d = DATASETS[dataset_name](root=dataset_root, train=(mode == 'train'), transform=trainsform)
    elif dataset_name == 'caltech101':
        d = DATASETS[dataset_name](root=dataset_root, transform=trainsform)
    elif dataset_name == 'svhn':
        if mode != 'train':
            mode = 'test'
        d = DATASETS[dataset_name](root=dataset_root, split=mode, transform=trainsform)

    return d