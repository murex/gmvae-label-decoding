# src/weak_supervision_labeling/data/mnist.py

from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

from weak_supervision_labeling.paths import RAW_DATA_DIR



def load_mnist_flat(
    normalize=True
):
    data_dir = RAW_DATA_DIR / "mnist"

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir / "mnist",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir / "mnist",
        train=False,
        download=True,
        transform=transform
    )

    # Flatten
    Xtr = train_dataset.data.numpy().reshape(len(train_dataset), -1)
    Xte = test_dataset.data.numpy().reshape(len(test_dataset), -1)

    ytr = train_dataset.targets.numpy()
    yte = test_dataset.targets.numpy()

    # Merge train + test
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)

    if normalize:
        X = X.astype("float32") / 255.0

    return X, y