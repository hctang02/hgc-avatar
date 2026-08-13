"""End-to-end HGC-Avatar encoder/compression/decoder simulation."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity

from hgc_avatar.codecs.posemap_dcvc import run_dcvc
from hgc_avatar.codecs.smpl_huffman import decode as decode_smpl
from hgc_avatar.codecs.smpl_huffman import encode as encode_smpl


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_yaml(path):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, allow_unicode=True, sort_keys=False)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Pipeline:
    def __init__(self, config_path):
        self.pipeline_config_path = Path(config_path).resolve()
        self.options = _load_yaml(self.pipeline_config_path)
        self.work_dir = self._resolve(self.options["work_dir"])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.source = self.options["source"]
        self.compression = self.options["compression"]

    @staticmethod
    def _resolve(path):
        path = Path(path).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_DIR / path).resolve()

    @property
    def source_checkpoint(self):
        path = self._resolve(self.source["checkpoint"])
        return path / "net.pt" if path.is_dir() else path

    def _avatar_options(self, decoded):
        options = _load_yaml(self._resolve(self.source["avatar_config"]))
        data_dir = str(self._resolve(self.source["data_dir"]))
        smpl_model_path = str(self._resolve(self.source["smpl_model_path"]))
        frame_range = self.source.get("frame_range", [0, 1, 1])
        subject_name = self.source.get("subject_name", Path(data_dir).name)

        common_data = {
            "data_dir": data_dir,
            "frame_range": frame_range,
            "subject_name": subject_name,
            "smpl_model_path": smpl_model_path,
        }
        options["mode"] = "test"
        options["train"]["data"].update(
            {
                "data_dir": data_dir,
                "smpl_model_path": smpl_model_path,
                "frame_range": frame_range,
            }
        )
        options["test"].pop("pose_data", None)
        options["test"]["data"] = dict(common_data)
        options["test"]["dataset"] = self.source.get(
            "dataset", options["train"].get("dataset", "MvRgbDatasetAvatarReX")
        )
        options["test"]["n_pca"] = -1
        options["test"]["img_scale"] = float(self.source.get("img_scale", 1.0))
        options["test"]["view_setting"] = self.source.get("view_setting", "front")

        if decoded:
            options["test"]["data"]["smpl_path"] = str(
                self.work_dir / "decoder" / "smpl_params.npz"
            )
            options["test"]["prev_ckpt"] = str(self.work_dir / "decoder" / "checkpoint")
            options["test"]["output_dir"] = str(self.work_dir / "decoder" / "render")
            options["model"]["pose_map_io"] = {
                "mode": "decoded",
                "frame_dir": str(self.work_dir / "compression" / "posemap" / "decoded" / "frames"),
                "range_dir": str(self.work_dir / "encoder" / "posemap" / "ranges"),
            }
        else:
            options["test"]["prev_ckpt"] = str(self.source_checkpoint.parent)
            options["test"]["output_dir"] = str(self.work_dir / "encoder" / "render")
            options["model"]["pose_map_io"] = {
                "mode": "export",
                "output_dir": str(self.work_dir / "encoder" / "posemap"),
            }
        return options

    def _run_avatar(self, decoded):
        stage = "decoder" if decoded else "encoder"
        generated_dir = self.work_dir / stage / "render"
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
        if not decoded:
            posemap_dir = self.work_dir / "encoder" / "posemap"
            if posemap_dir.exists():
                shutil.rmtree(posemap_dir)
        config_path = self.work_dir / stage / "avatar.generated.yaml"
        _write_yaml(config_path, self._avatar_options(decoded))
        env = os.environ.copy()
        if self.source.get("cuda_device") is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.source["cuda_device"])
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "main_avatar.py"), "-c", str(config_path), "--mode", "test"],
            cwd=PROJECT_DIR,
            env=env,
            check=True,
        )

    def encode(self):
        start = time.time()
        self._run_avatar(decoded=False)
        return {"seconds": time.time() - start}

    def compress(self):
        start = time.time()
        compression_dir = self.work_dir / "compression"
        decoder_dir = self.work_dir / "decoder"
        smpl_stream = compression_dir / "smpl" / "smpl_params.hgc"
        decoded_smpl = decoder_dir / "smpl_params.npz"
        smpl_result = encode_smpl(self._resolve(self.source["data_dir"]) / "smpl_params.npz", smpl_stream)
        decoded_result = decode_smpl(smpl_stream, decoded_smpl)
        if smpl_result["sha256"] != decoded_result["sha256"]:
            raise RuntimeError("SMPL-X lossless round trip verification failed")

        network_stream_dir = compression_dir / "network"
        decoded_checkpoint = decoder_dir / "checkpoint" / "net.pt"
        network_q_index = self.compression.get("network_q_index", 5)
        network_bits = network_stream_dir / f"{network_q_index}_bits"
        network_header = network_stream_dir / f"{network_q_index}_header"
        network_metrics = network_stream_dir / "metrics.json"
        network_meta_path = network_stream_dir / "metadata.json"
        desired_network_meta = {
            "source": str(self.source_checkpoint),
            "source_size": self.source_checkpoint.stat().st_size,
            "source_mtime_ns": self.source_checkpoint.stat().st_mtime_ns,
            "q_index": network_q_index,
        }
        existing_network_meta = None
        if network_meta_path.is_file():
            existing_network_meta = json.loads(network_meta_path.read_text(encoding="utf-8"))
        if existing_network_meta != desired_network_meta or not all(
            path.is_file() for path in (decoded_checkpoint, network_bits, network_header, network_metrics)
        ):
            print("Compressing StyleUNet parameters...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "compress_part/codec/quantization/core_coder.py"),
                    "--input_ckpt",
                    str(self.source_checkpoint),
                    "--output_ckpt",
                    str(decoded_checkpoint),
                    "--bitstream_dir",
                    str(network_stream_dir),
                    "--q_index",
                    str(network_q_index),
                    "--metrics_path",
                    str(network_metrics),
                ],
                cwd=PROJECT_DIR / "compress_part/codec/quantization",
                check=True,
            )
            network_meta_path.write_text(
                json.dumps(desired_network_meta, indent=2), encoding="utf-8"
            )
        else:
            print("Reusing verified network compression outputs.", flush=True)

        posemap_options = self.compression["posemap"]
        posemap_result = run_dcvc(
            self.work_dir / "encoder" / "posemap" / "frames",
            compression_dir / "posemap",
            self._resolve(posemap_options.get("codec_dir", "compress_part/codec/DCVC-DC")),
            self._resolve(posemap_options["i_frame_model"]),
            self._resolve(posemap_options["p_frame_model"]),
            posemap_options.get("rate_num", 2),
            posemap_options.get("rate_index", 0),
            posemap_options.get("intra_period", 32),
            True,
            self.source.get("cuda_device"),
        )
        manifest = {
            "elapsed_seconds": time.time() - start,
            "smpl": smpl_result,
            "network": {
                "q_index": network_q_index,
                "source_bytes": self.source_checkpoint.stat().st_size,
                "decoded_bytes": decoded_checkpoint.stat().st_size,
                "header_bytes": network_header.stat().st_size,
                "bitstream_bytes": network_bits.stat().st_size,
                "metrics": json.loads(network_metrics.read_text(encoding="utf-8")),
            },
            "posemap": posemap_result,
        }
        (self.work_dir / "compression_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def decode(self):
        start = time.time()
        self._run_avatar(decoded=True)
        return {"seconds": time.time() - start}

    def verify(self):
        encoder_dir = self.work_dir / "encoder" / "render" / "vanilla" / "rgb_map"
        decoder_dir = self.work_dir / "decoder" / "render" / "vanilla" / "rgb_map"
        expected = [encoder_dir, decoder_dir]
        report = {}
        for path in expected:
            frames = sorted(path.glob("*.jpg"))
            if not frames:
                raise RuntimeError(f"No rendered frames found under {path}")
            report[str(path)] = {"frames": len(frames), "first_sha256": _sha256(frames[0])}
        encoder_frames = {path.name: path for path in encoder_dir.glob("*.jpg")}
        decoder_frames = {path.name: path for path in decoder_dir.glob("*.jpg")}
        common_names = sorted(encoder_frames.keys() & decoder_frames.keys())
        psnrs = []
        maes = []
        ssims = []
        for name in common_names:
            encoder_uint8 = cv.imread(str(encoder_frames[name]), cv.IMREAD_COLOR)
            decoder_uint8 = cv.imread(str(decoder_frames[name]), cv.IMREAD_COLOR)
            encoder_image = encoder_uint8.astype(np.float32)
            decoder_image = decoder_uint8.astype(np.float32)
            difference = encoder_image - decoder_image
            mse = float(np.mean(difference ** 2))
            psnrs.append(float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
            maes.append(float(np.mean(np.abs(difference))))
            encoder_gray = cv.cvtColor(encoder_uint8, cv.COLOR_BGR2GRAY)
            decoder_gray = cv.cvtColor(decoder_uint8, cv.COLOR_BGR2GRAY)
            ssims.append(
                float(structural_similarity(encoder_gray, decoder_gray, data_range=255))
            )
        report["encoder_decoder_comparison"] = {
            "matched_frames": len(common_names),
            "mean_psnr_db": float(np.mean(psnrs)),
            "mean_absolute_error_8bit": float(np.mean(maes)),
        }
        if ssims:
            report["encoder_decoder_comparison"]["mean_ssim"] = float(np.mean(ssims))
        report_path = self.work_dir / "verification.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage", choices=("all", "encode", "compress", "decode", "verify"), default="all"
    )
    args = parser.parse_args()
    pipeline = Pipeline(args.config)
    stages = ("encode", "compress", "decode", "verify") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"\n===== HGC stage: {stage} =====", flush=True)
        result = getattr(pipeline, stage)()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
