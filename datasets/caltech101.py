from torchvision.datasets import Caltech101 as tCaltech101
from PIL import Image
import numpy as np
import os

TRAIN_SPLIT_PERCENT = 80
TEST_SPLIT_PERCENT = 20

class Caltech101(tCaltech101):
    def __init__(self, root,
                    target_type='category',
                    transform=None,
                    target_transform=None,
                    split='train',
                    download=False,
                    split_path='/scratch/caltech101/'):
        super(Caltech101, self).__init__(root,
                    target_type,
                    transform,
                    target_transform,
                    download=True)
        
        self.index = np.load(os.path.join(split_path, f"caltech101_{split}_index.npy"))
        self.y = np.load(os.path.join(split_path, f"caltech101_{split}_y.npy"))
        if split == 'train':
            self.index = self.index[:1000]
            self.y = self.y[:1000]


    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where the type of target specified by target_type.
        """
        import scipy.io

        img = Image.open(
            os.path.join(
                self.root,
                "101_ObjectCategories",
                self.categories[self.y[index]],
                f"image_{self.index[index]:04d}.jpg",
            )
        ).convert('RGB')

        target = []
        for t in self.target_type:
            if t == "category":
                target.append(self.y[index])
            elif t == "annotation":
                data = scipy.io.loadmat(
                    os.path.join(
                        self.root,
                        "Annotations",
                        self.annotation_categories[self.y[index]],
                        f"annotation_{self.index[index]:04d}.mat",
                    )
                )
                target.append(data["obj_contour"])
        target = tuple(target) if len(target) > 1 else target[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target