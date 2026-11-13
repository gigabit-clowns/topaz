import numpy as np
import torch

from topaz import mrc
from topaz.denoising.datasets import PairedImages
from topaz.utils.data.loader import load_image


def _pairs(tmp_path, count=5):
    rng = np.random.default_rng(0)
    a, b = [], []
    for i in range(count):
        for paths, half in ((a, 'a'), (b, 'b')):
            path = str(tmp_path / f'{half}{i}.mrc')
            with open(path, 'wb') as f:
                mrc.write(f, (rng.standard_normal((70, 90 + i)) * 2 + 5).astype(np.float32)[np.newaxis])
            paths.append(path)
    return a, b


def _normalized(path, cutoff):
    x = load_image(path, make_image=False, return_header=False).astype(np.float32)
    x = (x - x.mean()) / x.std()
    if cutoff > 0:
        x[(x < -cutoff) | (x > cutoff)] = 0
    return x


def test_crops_are_those_of_the_normalized_micrographs(tmp_path):
    a, b = _pairs(tmp_path)
    for preload in (False, True):
        data = PairedImages(a, b, crop=32, xform=False, preload=preload, cutoff=1.5)
        np.random.seed(2)
        corners = data.crops().fields['x'].centres - 16
        np.random.seed(2)
        x, y = (torch.cat(parts).numpy() for parts in zip(*data.make_loader(2, False, 0)))
        for row, (i, j) in enumerate(corners):
            assert np.array_equal(x[row], _normalized(a[row], 1.5)[i:i+32, j:j+32])
            assert np.array_equal(y[row], _normalized(b[row], 1.5)[i:i+32, j:j+32])


def test_a_pass_crops_every_pair_once_and_anew(tmp_path):
    a, b = _pairs(tmp_path)
    data = PairedImages(a, b, crop=32, xform=True)
    batches = data.make_loader(2, True, 0)
    assert len(batches) == 3
    first, second = (torch.cat([x for x, _ in batches]) for _ in range(2))
    assert first.shape == second.shape == (5, 32, 32)
    assert not torch.equal(first, second)
    x, y = data[0]
    assert x.shape == y.shape == (32, 32)
