import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hgc_avatar.codecs.smpl_huffman import decode, encode


class SmplHuffmanTest(unittest.TestCase):
    def test_round_trip_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "smpl_params.npz"
            stream = directory / "smpl_params.hgc"
            restored = directory / "restored.npz"
            np.savez(
                source,
                betas=np.arange(10, dtype=np.float32)[None],
                body_pose=np.linspace(-1, 1, 4 * 63, dtype=np.float32).reshape(4, 63),
            )
            encoded = encode(source, stream)
            decoded = decode(stream, restored)
            self.assertEqual(source.read_bytes(), restored.read_bytes())
            self.assertEqual(encoded["sha256"], decoded["sha256"])
            self.assertEqual(
                encoded["sha256"], hashlib.sha256(source.read_bytes()).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()

