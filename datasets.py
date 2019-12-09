from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Pad
from torch.utils.data import DataLoader


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
