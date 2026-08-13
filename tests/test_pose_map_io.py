import json
import tempfile
import unittest
from pathlib import Path

import torch

from utils.pose_map_io import export_pose_map, load_pose_map


class PoseMapIoTest(unittest.TestCase):
    def test_export_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pose_map = torch.linspace(-1.25, 0.4, 3 * 16 * 16).reshape(3, 16, 16)
            image_path, range_path = export_pose_map(pose_map, 7, directory)
            restored = load_pose_map(7, image_path.parent, range_path.parent, "cpu")
            self.assertEqual(restored.shape, pose_map.shape)
            quantization_step = (pose_map.max() - pose_map.min()).item() / 255
            self.assertLessEqual((restored - pose_map).abs().max().item(), quantization_step)
            self.assertEqual(json.loads(range_path.read_text())["min"], pose_map.min().item())


if __name__ == "__main__":
    unittest.main()

