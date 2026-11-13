import numpy as np

from topaz import mrc
from topaz.utils.data.loader import load_mrc, read_ahead, read_mrc


def _write(path, array):
    with open(path, 'wb') as f:
        mrc.write(f, array[np.newaxis])


def test_read_mrc_is_load_mrc(tmp_path):
    for dtype in (np.float32, np.float16, np.int16):
        path = str(tmp_path / f'{np.dtype(dtype).name}.mrc')
        _write(path, (np.arange(12 * 17) % 251).astype(dtype).reshape(12, 17))
        expected, header, extended = load_mrc(path)
        image, read_header, read_extended = read_mrc(path)
        assert image.dtype == expected.dtype
        assert np.array_equal(image, expected)
        assert read_header == header and read_extended == extended


def test_read_ahead_keeps_the_order_and_every_image(tmp_path):
    paths = []
    for i in range(7):
        path = str(tmp_path / f'{i}.mrc')
        _write(path, np.full((4, 5 + i), i, dtype=np.float32))
        paths.append(path)
    read = list(read_ahead(iter(paths), depth=2))
    assert [path for path, *_ in read] == paths
    assert all(np.all(image == i) and image.shape == (4, 5 + i)
               for i, (_, image, _, _) in enumerate(read))
