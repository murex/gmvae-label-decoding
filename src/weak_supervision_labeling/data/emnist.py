# src/weak_supervision_labeling/data/emnist.py

from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import DataLoader

from weak_supervision_labeling.paths import RAW_DATA_DIR


def load_emnist_letters_flat(normalize=True):
    data_dir = RAW_DATA_DIR / "emnist"

    # Correction EMNIST: rotate + mirror
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.rot90(x, -1, [1, 2])),
        transforms.Lambda(lambda x: torch.flip(x, [2])),
    ])

    train_dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.EMNIST(
        root=data_dir,
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )

    def _to_flat_numpy(ds):
        loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
        xs, ys = [], []
        for x, y in loader:
            x = x.view(x.size(0), -1)
            xs.append(x.cpu().numpy())
            ys.append(y.cpu().numpy())
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0) - 1
        return X, y

    Xtr, ytr = _to_flat_numpy(train_dataset)
    Xte, yte = _to_flat_numpy(test_dataset)

    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)

    X = X.astype("float32") if normalize else (X * 255.0).astype("float32")

    return X, y