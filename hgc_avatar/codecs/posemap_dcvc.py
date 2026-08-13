"""Run DCVC-DC on the exported PoseMap PNG sequence."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _pose_frames(frame_dir):
    frames = sorted(Path(frame_dir).glob("pose_map_*.png"))
    if not frames:
        raise FileNotFoundError(f"No pose_map_*.png frames found under {frame_dir}")
    return frames


def run_dcvc(
    frame_dir,
    work_dir,
    codec_dir,
    i_frame_model,
    p_frame_model,
    rate_num=2,
    rate_index=0,
    intra_period=32,
    cuda=True,
    cuda_device=None,
):
    """Encode/decode a sequence and return the selected decoded frame directory."""
    frames = _pose_frames(frame_dir)
    if not 0 <= rate_index < rate_num:
        raise ValueError("rate_index must be smaller than rate_num")

    work_dir = Path(work_dir).resolve()
    codec_dir = Path(codec_dir).resolve()
    source_dir = work_dir / "source" / "pose_map"
    stream_dir = work_dir / "bitstreams"
    all_decoded_dir = work_dir / "dcvc_decoded"
    selected_dir = work_dir / "decoded" / "frames"
    # These directories are owned by this codec invocation. Clearing them avoids
    # stale frames and DCVC's rename collision when a stage is intentionally rerun.
    for generated_dir in (source_dir, stream_dir, all_decoded_dir, selected_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    width = height = None
    for sequence_index, source in enumerate(frames, start=1):
        with Image.open(source) as image:
            current_width, current_height = image.size
            if width is None:
                width, height = current_width, current_height
            elif (width, height) != (current_width, current_height):
                raise ValueError("All PoseMap frames must have identical dimensions")
            image.save(source_dir / f"im{sequence_index:05d}.png", format="PNG")

    test_config = {
        "root_path": str(work_dir / "source"),
        "test_classes": {
            "PoseMap": {
                "test": 1,
                "base_path": ".",
                "src_type": "png",
                "sequences": {
                    "pose_map": {
                        "width": width,
                        "height": height,
                        "frames": len(frames),
                        "gop": intra_period,
                    }
                },
            }
        },
    }
    config_path = work_dir / "dcvc_dataset.json"
    config_path.write_text(json.dumps(test_config, indent=2), encoding="utf-8")
    output_json = work_dir / "dcvc_metrics.json"
    command = [
        sys.executable,
        str(codec_dir / "test_video.py"),
        "--i_frame_model_path",
        str(Path(i_frame_model).resolve()),
        "--p_frame_model_path",
        str(Path(p_frame_model).resolve()),
        "--rate_num",
        str(rate_num),
        "--test_config",
        str(config_path),
        "--yuv420",
        "0",
        "--cuda",
        "1" if cuda else "0",
        "--worker",
        "1",
        "--write_stream",
        "1",
        "--stream_path",
        str(stream_dir),
        "--output_path",
        str(output_json),
        "--force_intra_period",
        str(intra_period),
        "--force_frame_num",
        str(len(frames)),
        "--save_decoded_frame",
        "1",
        "--decoded_frame_path",
        str(all_decoded_dir),
        "--root_path",
        str(work_dir / "source"),
        "--verbose",
        "1",
    ]
    if cuda_device is not None:
        command.extend(["--cuda_device", str(cuda_device)])
    env = os.environ.copy()
    # Extensions compiled with the host GCC may require a newer GLIBCXX symbol
    # than the old Conda environment ships. Prefer the compatible host runtime.
    system_libstdcpp = Path("/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    if system_libstdcpp.is_file():
        current_preload = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = ":".join(
            value for value in (str(system_libstdcpp), current_preload) if value
        )
    subprocess.run(command, cwd=codec_dir, env=env, check=True)

    candidates = sorted((all_decoded_dir / "pose_map").glob(f"{rate_index}_*"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one decoded DCVC directory for rate {rate_index}, got {candidates}"
        )
    for output_index, source in enumerate(sorted(candidates[0].glob("im*.png"))):
        original_index = int(frames[output_index].stem.rsplit("_", 1)[1])
        shutil.copy2(source, selected_dir / f"pose_map_{original_index:08d}.png")

    selected_stream_dir = stream_dir / "pose_map" / str(rate_index)
    metrics = json.loads(output_json.read_text(encoding="utf-8"))
    return {
        "decoded_frame_dir": str(selected_dir),
        "bitstream_dir": str(selected_stream_dir),
        "metrics": metrics["PoseMap"]["pose_map"][f"{rate_index:03d}"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--codec-dir", default="compress_part/codec/DCVC-DC")
    parser.add_argument("--i-frame-model", required=True)
    parser.add_argument("--p-frame-model", required=True)
    parser.add_argument("--rate-num", type=int, default=2)
    parser.add_argument("--rate-index", type=int, default=0)
    parser.add_argument("--intra-period", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda-device")
    args = parser.parse_args()
    result = run_dcvc(
        args.frame_dir,
        args.work_dir,
        args.codec_dir,
        args.i_frame_model,
        args.p_frame_model,
        args.rate_num,
        args.rate_index,
        args.intra_period,
        not args.cpu,
        args.cuda_device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
