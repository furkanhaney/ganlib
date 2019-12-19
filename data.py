import h5py
import numpy as np
from torchvision import datasets
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Pad
from torch.utils.data import DataLoader, Dataset


def get_fashion_loader(batch_size=32, num_workers=2):
    transforms = Compose([
        Pad(2),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    dataset = FashionMNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms
    )
    loader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True
    )
    return loader


def get_mnist_loader(batch_size=32, num_workers=2):
    transforms = Compose([
        Pad(2),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    dataset = MNIST(
        '../data',
        train=True,
        download=True,
        transform=transforms
    )
    loader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=True
    )
    return loader


def get_flickr_loader(batch_size=32, num_workers=2):
    dataset = SudokuDataset(in_memory=True)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    return loader


class Flickr128(Dataset):
    def __init__(self, in_memory=True):
        self.data_path = "data/flickr128.hdf5"
        self.in_memory = in_memory
        with h5py.File(self.data_path, "r") as file:
            self.length = file["images"].shape[0]
        if self.in_memory:
            with h5py.File(self.data_path, "r") as file:
                self.images = np.array(file["images"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.in_memory:
            image = self.images[idx]
        else:
            with h5py.File(self.data_path, "r") as file:
                image = file["images"][idx]
        image = image.astype(np.float32)
        return image
