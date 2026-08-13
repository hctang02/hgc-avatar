#!/usr/bin/env python3
"""Run the reproducible multi-subject, multi-quantization HGC experiment matrix."""

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, allow_unicode=True, sort_keys=False)


def json_load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_count(path):
    return len(list(Path(path).glob("*.jpg")))


def bytes_below(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {path}")
    path.symlink_to(target, target_is_directory=target.is_dir())


def run_logged(command, log_path, cwd=PROJECT_DIR):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(map(str, command))}\n")
        log.flush()
        subprocess.run(
            list(map(str, command)), cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    return time.time() - started


def pipeline_options(matrix, subject, q_index, work_dir):
    avatar_config = Path(subject["avatar_config"])
    if not avatar_config.is_absolute():
        avatar_config = PROJECT_DIR / avatar_config
    return {
        "work_dir": str(work_dir),
        "source": {
            "subject_name": subject["name"],
            "dataset": subject["dataset"],
            "avatar_config": str(avatar_config),
            "data_dir": subject["data_dir"],
            "smpl_model_path": matrix["smpl_model_path"],
            "checkpoint": subject["checkpoint"],
            "frame_range": matrix["frame_range"],
            "view_setting": "front",
            "img_scale": 1.0,
            "cuda_device": subject["cuda_device"],
        },
        "compression": {"network_q_index": q_index, "posemap": matrix["posemap"]},
    }


def run_pipeline_stage(config_path, stage, log_path):
    return run_logged(
        [sys.executable, "-m", "hgc_avatar.pipeline", "--config", config_path, "--stage", stage],
        log_path,
    )


def network_only(subject, q_index, point_dir, log_path):
    network_dir = point_dir / "compression" / "network"
    checkpoint = point_dir / "decoder" / "checkpoint" / "net.pt"
    metrics = network_dir / "metrics.json"
    if metrics.is_file() and checkpoint.is_file():
        return
    run_logged(
        [
            sys.executable,
            PROJECT_DIR / "compress_part/codec/quantization/core_coder.py",
            "--input_ckpt",
            Path(subject["checkpoint"]) / "net.pt",
            "--output_ckpt",
            checkpoint,
            "--bitstream_dir",
            network_dir,
            "--q_index",
            q_index,
            "--metrics_path",
            metrics,
        ],
        log_path,
        PROJECT_DIR / "compress_part/codec/quantization",
    )


def run_subject(matrix, subject):
    name = subject["name"]
    root = Path(matrix["output_dir"]) / name
    root.mkdir(parents=True, exist_ok=True)
    expected_frames = len(range(*matrix["frame_range"]))
    base_q = matrix["quantization_levels"][0]
    base_dir = root / f"q{base_q}"
    base_config = base_dir / "pipeline.yaml"
    write_yaml(base_config, pipeline_options(matrix, subject, base_q, base_dir))
    print(f"[{name}] q{base_q}: encoder/shared streams", flush=True)
    if frame_count(base_dir / "encoder/render/vanilla/rgb_map") != expected_frames:
        run_pipeline_stage(base_config, "encode", root / "experiment.log")
    if not (base_dir / "compression_manifest.json").is_file():
        run_pipeline_stage(base_config, "compress", root / "experiment.log")
    if frame_count(base_dir / "decoder/render/vanilla/rgb_map") != expected_frames:
        run_pipeline_stage(base_config, "decode", root / "experiment.log")
    if not (base_dir / "verification.json").is_file():
        run_pipeline_stage(base_config, "verify", root / "experiment.log")

    for q_index in matrix["quantization_levels"][1:]:
        point_dir = root / f"q{q_index}"
        config_path = point_dir / "pipeline.yaml"
        write_yaml(config_path, pipeline_options(matrix, subject, q_index, point_dir))
        link(base_dir / "encoder", point_dir / "encoder")
        link(base_dir / "compression/posemap", point_dir / "compression/posemap")
        link(base_dir / "compression/smpl", point_dir / "compression/smpl")
        link(base_dir / "decoder/smpl_params.npz", point_dir / "decoder/smpl_params.npz")
        print(f"[{name}] q{q_index}: network + decoder", flush=True)
        network_only(subject, q_index, point_dir, root / "experiment.log")
        if frame_count(point_dir / "decoder/render/vanilla/rgb_map") != expected_frames:
            run_pipeline_stage(config_path, "decode", root / "experiment.log")
        if not (point_dir / "verification.json").is_file():
            run_pipeline_stage(config_path, "verify", root / "experiment.log")
    return name


def collect_rows(matrix):
    rows = []
    frame_num = len(range(*matrix["frame_range"]))
    for subject in matrix["subjects"]:
        base_dir = Path(matrix["output_dir"]) / subject["name"] / f"q{matrix['quantization_levels'][0]}"
        shared = json_load(base_dir / "compression_manifest.json")
        pose_bits = bytes_below(shared["posemap"]["bitstream_dir"])
        range_bytes = bytes_below(base_dir / "encoder/posemap/ranges")
        smpl_bytes = shared["smpl"]["output_bytes"]
        for q_index in matrix["quantization_levels"]:
            point_dir = Path(matrix["output_dir"]) / subject["name"] / f"q{q_index}"
            network = json_load(point_dir / "compression/network/metrics.json")
            verification = json_load(point_dir / "verification.json")["encoder_decoder_comparison"]
            network_bytes = network["header_bytes"] + network["bitstream_bytes"]
            total_bytes = network_bytes + smpl_bytes + pose_bits + range_bytes
            rows.append(
                {
                    "dataset_family": subject["family"],
                    "subject": subject["name"],
                    "frames": frame_num,
                    "q_index": q_index,
                    "q_step": network["q_step"],
                    "network_bytes": network_bytes,
                    "smpl_bytes": smpl_bytes,
                    "posemap_bitstream_bytes": pose_bits,
                    "posemap_range_bytes": range_bytes,
                    "total_transmission_bytes": total_bytes,
                    "total_kib_per_frame": total_bytes / frame_num / 1024,
                    "network_compression_ratio": network["source_checkpoint_bytes"] / network_bytes,
                    "network_mse": network["mse"],
                    "network_max_absolute_error": network["max_absolute_error"],
                    "reference_psnr_db": verification["mean_psnr_db"],
                    "reference_ssim": verification.get("mean_ssim"),
                    "reference_mae_8bit": verification["mean_absolute_error_8bit"],
                    "network_encode_seconds": network["encoding_seconds"],
                    "network_decode_seconds": network["decoding_seconds"],
                    "posemap_bpp": shared["posemap"]["metrics"]["ave_all_frame_bpp"],
                    "posemap_psnr_db": shared["posemap"]["metrics"]["ave_all_frame_psnr"],
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/compression_matrix.server.yaml")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--subjects", nargs="*", help="Only run the named subjects")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    matrix = load_yaml(config_path)
    if args.subjects:
        requested = set(args.subjects)
        matrix["subjects"] = [
            subject for subject in matrix["subjects"] if subject["name"] in requested
        ]
        found = {subject["name"] for subject in matrix["subjects"]}
        if found != requested:
            parser.error(f"Unknown subjects: {sorted(requested - found)}")
    output_dir = Path(matrix["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run_subject, matrix, subject) for subject in matrix["subjects"]]
        for future in concurrent.futures.as_completed(futures):
            print(f"Completed subject: {future.result()}", flush=True)
    rows = collect_rows(matrix)
    summary_json = output_dir / "experiment_summary.json"
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "experiment_summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "matrix_config": str(config_path),
        "elapsed_seconds": time.time() - started,
        "subjects": len(matrix["subjects"]),
        "operating_points": len(rows),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
