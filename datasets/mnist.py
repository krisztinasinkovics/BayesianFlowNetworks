import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def get_basic_transform(train=True):
    return transforms.Compose([#transforms.ToTensor(),
                            transforms.RandomHorizontalFlip() if train else nn.Identity()])


class DynamicallyBinarizedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(DynamicallyBinarizedMNIST, self).__init__(root, train=train, transform=transform,
                                                        target_transform=target_transform, download=download)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def collate_dynamic_binarize(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function that samples a binarization probability for each batch.

    Args:
        batch (list[tuple[torch.Tensor, int]]): list of samples to collate.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: resulting batch.
    """
    images, targets = zip(*batch)
    binarization_probs = torch.rand(len(images))
    binarized_images = []
    for img, prob in zip(images, binarization_probs):
        binarized_img = (img > prob).float()
        binarized_images.append(binarized_img)
    return torch.stack(binarized_images)[:, None, ...].to(torch.int64), torch.tensor(targets)


# Create the dynamically binarized MNIST dataset
def get_mnist_datasets(root='./datasets/') -> tuple[Dataset, Dataset, Dataset]:
    # create datasets, train and val are the same (apart from transform), test is official cifar10 test set
    train_set = DynamicallyBinarizedMNIST(root=root, train=True, download=True, transform=get_basic_transform(train=True))
    val_set = DynamicallyBinarizedMNIST(root=root, train=True, download=True, transform=get_basic_transform(train=False))
    test_set = DynamicallyBinarizedMNIST(root=root, train=False, download=True, transform=get_basic_transform(train=False))
    return train_set, val_set, test_set


def get_mnist_dataloaders(batch_size:int = 32, root='./datasets/', valid_size=0.01, seed=7) -> tuple[DataLoader, DataLoader, DataLoader]:

    # get datasets
    train_set, val_set, test_set = get_mnist_datasets(root)

    # train and val are the same by default, split them up using SubsetRandomSampler
    indices = list(range(len(train_set)))
    split = int(valid_size*len(train_set))
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # create dataloaders, using samplers
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size, drop_last=True, collate_fn=collate_dynamic_binarize)
    val_loader = DataLoader(val_set, sampler=valid_sampler, batch_size=batch_size, collate_fn=collate_dynamic_binarize)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_dynamic_binarize)
    return train_loader, val_loader, test_loader
