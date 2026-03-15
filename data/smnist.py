"""Sequential MNIST (sMNIST) and Permuted Sequential MNIST (psMNIST) datasets.

Images are presented one row at a time: seq_len=28, input_size=28.
For psMNIST, a fixed random permutation is applied to the 784 pixels before
reshaping into sequences.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class SequentialMNIST(Dataset):
    """MNIST presented as a sequence of rows: (seq_len=28, input_size=28).

    Args:
        root: Directory to store/load the MNIST data.
        train: If True, use the training split; otherwise the test split.
        permutation: Optional fixed permutation tensor of shape (784,) for psMNIST.
            If None, pixels are presented in natural row order.
        download: Whether to download MNIST if not already present.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        permutation: torch.Tensor | None = None,
        download: bool = True,
    ):
        self.permutation = permutation

        mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )
        # Pre-load all data into memory — MNIST fits easily in RAM
        data, targets = zip(*[(img, label) for img, label in mnist], strict=True)
        self.data = torch.stack(data)  # (N, 1, 28, 28)
        self.targets = torch.tensor(targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.data[idx].squeeze(0)  # (28, 28)
        flat = img.reshape(784)

        if self.permutation is not None:
            flat = flat[self.permutation]

        sequence = flat.reshape(28, 28)  # (seq_len=28, input_size=28)
        return sequence, self.targets[idx].item()


def make_permutation(seed: int = 42) -> torch.Tensor:
    """Return a fixed random permutation of 784 indices for psMNIST."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randperm(784, generator=rng)
