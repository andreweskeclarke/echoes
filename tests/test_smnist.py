"""Tests for sMNIST and psMNIST dataset implementations."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from unittest.mock import patch

import torch

from data.smnist import SequentialMNIST, make_permutation

MNIST_PIXELS = 784
MNIST_ROWS = 28
NUM_CLASSES = 10


class FakeMNIST:
    """Minimal MNIST substitute for testing without downloading data."""

    def __init__(self, size=100):
        shape = (1, MNIST_ROWS, MNIST_ROWS)
        self._data = [(torch.rand(shape), i % NUM_CLASSES) for i in range(size)]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def make_dataset(train=True, permutation=None, size=100):
    with patch("data.smnist.datasets.MNIST", return_value=FakeMNIST(size)):
        return SequentialMNIST(
            root="/tmp/fake", train=train, permutation=permutation, download=False
        )


class TestSequentialMNIST:
    def test_sequence_shape(self):
        ds = make_dataset()
        seq, _label = ds[0]
        expected = (MNIST_ROWS, MNIST_ROWS)
        assert seq.shape == expected, f"Expected {expected}, got {seq.shape}"

    def test_label_is_int(self):
        ds = make_dataset()
        _, label = ds[0]
        assert isinstance(label, int)
        assert 0 <= label < NUM_CLASSES

    def test_len(self):
        size = 50
        ds = make_dataset(size=size)
        assert len(ds) == size

    def test_pixel_values_in_range(self):
        ds = make_dataset()
        seq, _ = ds[0]
        assert seq.min() >= 0.0
        assert seq.max() <= 1.0

    def test_natural_order_preserved(self):
        """Without permutation, row i of the sequence equals row i of the image."""
        ds = make_dataset(permutation=None)
        img = ds.data[0].squeeze(0)  # (28, 28)
        seq, _ = ds[0]
        assert torch.allclose(seq, img)

    def test_permutation_changes_order(self):
        perm = make_permutation(seed=42)
        ds_plain = make_dataset(permutation=None)
        ds_perm = make_dataset(permutation=perm)

        shared_data = torch.rand(100, 1, MNIST_ROWS, MNIST_ROWS)
        ds_plain.data = shared_data
        ds_perm.data = shared_data

        seq_plain, _ = ds_plain[0]
        seq_perm, _ = ds_perm[0]
        assert not torch.allclose(seq_plain, seq_perm)

    def test_permutation_is_lossless(self):
        """All pixel values survive permutation — just reordered."""
        perm = make_permutation(seed=99)
        ds = make_dataset(permutation=perm)

        img = ds.data[0].squeeze(0).reshape(MNIST_PIXELS)
        seq, _ = ds[0]
        flat_seq = seq.reshape(MNIST_PIXELS)

        assert torch.allclose(img.sort().values, flat_seq.sort().values)


class TestMakePermutation:
    def test_shape(self):
        perm = make_permutation()
        assert perm.shape == (MNIST_PIXELS,)

    def test_is_valid_permutation(self):
        perm = make_permutation()
        assert perm.min() == 0
        assert perm.max() == MNIST_PIXELS - 1
        assert len(perm.unique()) == MNIST_PIXELS

    def test_reproducible(self):
        perm1 = make_permutation(seed=42)
        perm2 = make_permutation(seed=42)
        assert torch.equal(perm1, perm2)

    def test_different_seeds_differ(self):
        perm1 = make_permutation(seed=1)
        perm2 = make_permutation(seed=2)
        assert not torch.equal(perm1, perm2)
