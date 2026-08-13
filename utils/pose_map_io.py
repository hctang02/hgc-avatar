"""PoseMap image serialization used by the HGC encoder/decoder simulation.

Only the front three channels are transmitted, matching the original experiment.
Each frame is normalized independently and its floating-point range is stored as
JSON so that the decoder can restore the network input after video decoding.
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np
import torch


def _frame_index(value):
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return int(value.item())
    return int(value)


def export_pose_map(pose_map, frame_index, output_dir):
    """Write one 3xHxW PoseMap as a lossless PNG and a range sidecar."""
    frame_index = _frame_index(frame_index)
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    range_dir = output_dir / "ranges"
    frame_dir.mkdir(parents=True, exist_ok=True)
    range_dir.mkdir(parents=True, exist_ok=True)

    pose_min = float(pose_map.min().item())
    pose_max = float(pose_map.max().item())
    span = pose_max - pose_min
    if span == 0:
        normalized = torch.zeros_like(pose_map)
    else:
        normalized = (pose_map - pose_min) / span
    image = torch.round(normalized.clamp(0, 1) * 255).to(torch.uint8)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    image_path = frame_dir / f"pose_map_{frame_index:08d}.png"
    range_path = range_dir / f"pose_map_{frame_index:08d}.json"
    if not cv.imwrite(str(image_path), image):
        raise IOError(f"Cannot write PoseMap image: {image_path}")
    with range_path.open("w", encoding="utf-8") as file:
        json.dump({"min": pose_min, "max": pose_max}, file)
    return image_path, range_path


def _find_frame(frame_dir, frame_index):
    candidates = [
        f"pose_map_{frame_index:08d}.png",
        f"pose_map_{frame_index:08d}.jpg",
        f"pose_map_{frame_index}.png",
        f"pose_map_{frame_index}.jpg",
        f"im{frame_index + 1:05d}.png",
    ]
    for name in candidates:
        path = Path(frame_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Decoded PoseMap frame {frame_index} not found under {frame_dir}; "
        f"tried: {', '.join(candidates)}"
    )


def _find_range(range_dir, frame_index):
    candidates = [
        f"pose_map_{frame_index:08d}.json",
        f"pose_map_{frame_index}.json",
        f"pose_map_{frame_index:08d}_range.json",
        f"pose_map_{frame_index}_range.json",
    ]
    for name in candidates:
        path = Path(range_dir) / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"PoseMap range for frame {frame_index} not found under {range_dir}; "
        f"tried: {', '.join(candidates)}"
    )


def load_pose_map(frame_index, frame_dir, range_dir, device):
    """Restore one decoded PoseMap image to a float tensor on ``device``."""
    frame_index = _frame_index(frame_index)
    image_path = _find_frame(frame_dir, frame_index)
    range_path = _find_range(range_dir, frame_index)
    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Cannot read decoded PoseMap image: {image_path}")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = torch.from_numpy(image).to(torch.float32).permute(2, 0, 1) / 255.0
    with range_path.open("r", encoding="utf-8") as file:
        value_range = json.load(file)
    restored = image * (float(value_range["max"]) - float(value_range["min"]))
    restored += float(value_range["min"])
    return restored.to(device)
