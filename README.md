# HGC-Avatar：面向传输的可驱动高斯人体头像

本仓库是在 [Animatable Gaussians（CVPR 2024）](https://github.com/lizhe00/AnimatableGaussians) 基础上实现的 HGC-Avatar 研究代码。目标是在编码端完成头像重建，分别传输人体姿态、StyleUNet 网络参数和 PoseMap，再用解码后的三类中间量执行第二次测试，模拟真实解码端重建。

当前流程依据 `ACMMM_2025_CamreaReady.pdf` 与 `hgc-avatar运行指南.pdf` 整理。原来需要反复修改 `pred.py`、`avatar.py` 并手动复制文件的操作，现已合并为一个配置文件和一个命令。

## 流程概览

```text
原始 SMPL-X + 原始 net.pt
             │
             ▼
     编码端 main_avatar test
       ├── 编码端 RGB 重建
       └── 逐帧导出 PoseMap PNG + min/max
             │
             ├── SMPL-X ── 无损 Huffman ──► 解码 smpl_params.npz
             ├── net.pt ── 量化 + Range Coding ──► 解码 net.pt
             └── PoseMap ── DCVC-DC 视频编解码 ──► 解码 PoseMap PNG
                                                    │
                                                    ▼
                                         解码端 main_avatar test
                                                    │
                                                    ▼
                                              解码端 RGB 重建
```

压缩逻辑保持论文思路不变：

- SMPL-X 姿态参数：无损 Huffman 编解码；新流包含完整码表，可以在独立进程中解码，并逐字节校验 SHA-256。
- StyleUNet 网络参数：按 `q_index` 选择量化步长，再进行基于 Laplace 分布的范围编码。
- PoseMap：逐帧归一化为 8-bit RGB 图像，用 DCVC-DC 做时序视频编解码，解码后结合原始逐帧 `min/max` 恢复网络输入。

旧版 SMPL 脚本实际会先量化浮点数，而且没有把码表、形状和数值范围写入 `.bin`，无法独立解码；本仓库已改成真正自包含的无损流。旧版还把 DCVC 输出再保存成 JPEG，本仓库保留 PNG，避免额外一次有损压缩。

## 已验证状态

2026-08-13 在 NVIDIA L20、CUDA 11.7、PyTorch 2.0.0 上完成 subject00 500 帧端到端实测：

| 项目 | 结果 |
|---|---:|
| 编码端 / 解码端重建帧数 | 500 / 500 |
| 原始网络 checkpoint | 903,341,752 bytes |
| q=5 网络码流 | 357,516,056 bytes |
| 网络参数 MSE / 最大误差 | 4.9963e-8 / 3.9673e-4 |
| SMPL-X 原始 / Huffman 流 | 1,722,078 / 1,614,722 bytes |
| SMPL-X 解码校验 | SHA-256 完全一致 |
| PoseMap GOP / I 帧 / P 帧 | 32 / 16 / 484 |
| PoseMap DCVC（rate 0） | 0.004069 bpp，41.9786 dB |
| PoseMap rate 0 码流 / range side info | 66,660 / 27,420 bytes |
| 编码端图像与模拟解码端图像 | 36.2299 dB，8-bit MAE 0.6064 |
| encode / compress / decode 耗时 | 135 / 426 / 127 秒 |

另外也保留了单帧冒烟配置，仅用于快速验证代码连通性；正式率失真结果应使用包含 P 帧的完整序列。

## 仓库与大文件边界

Git 仓库只保存核心代码和配置，不保存以下大文件：

- `data/` 下的图像、mask 和预处理数据；
- SMPL-X 授权模型；
- `.pt`、`.pth.tar` 等训练或 DCVC 权重；
- `results/`、`test_results/`、`compress_part/data/` 和运行输出；
- 本地编译的 `.so`。

因此其他人可以正常编辑、提交和运行核心代码，但必须自行准备数据、授权模型和 checkpoint。`.gitignore` 已防止这些大文件被误传到 GitHub。

## 统一环境

统一使用 `hgc-avatar` 环境，不再在 `anim-gaussian` 与 `compress` 之间切换。本机已验证环境位于：

```bash
/mnt/hdd2tC/tmp/haocheng/conda_envs/hgc-avatar
```

环境配置只使用下面这一种方案：

```bash
conda env create -f environment.yml
conda activate hgc-avatar
PYTHON="$(which python)" ./scripts/build_extensions.sh
```

这个环境同时包含头像重建、StyleUNet 网络压缩、SMPL-X Huffman 压缩和 DCVC-DC PoseMap 视频压缩所需的依赖。

关键版本为 Python 3.8.20、PyTorch 2.0.0、TorchVision 0.15.1、CUDA 11.7、PyTorch3D 0.7.3、NumPy 1.21.6、`constriction` 0.4.1 与 `pytorch-msssim` 1.0.0。

DCVC 扩展由系统 GCC 编译时，较旧 Conda 环境可能缺少 `GLIBCXX_3.4.30`。统一入口会在 DCVC 子进程中自动选择兼容的系统 `libstdc++.so.6`，不会改变头像渲染进程使用的库。

## 必需资源

以 THuman4.0 subject00 为例，需要：

```text
/path/to/data/thuman4.0/subject00/
├── calibration.json
├── smpl_params.npz
├── smpl_pos_map/
│   ├── cano_smpl_pos_map.exr
│   ├── cano_smpl_nml_map.exr
│   └── init_pts_lbs.npy
└── ...

/path/to/smpl_files/smplx/
└── SMPLX_NEUTRAL.npz（以及所需模型文件）

/path/to/avatar_checkpoint/
└── net.pt

/path/to/DCVC-checkpoints/
├── cvpr2023_image_psnr.pth.tar
└── cvpr2023_video_psnr.pth.tar
```

SMPL-X 需要从[官方页面](https://smpl-x.is.tue.mpg.de/)申请并遵守其许可证。数据预处理仍参考 [gen_data/GEN_DATA.md](gen_data/GEN_DATA.md)。

## 一条命令运行完整链路

先复制并修改示例配置：

```bash
cp configs/pipeline/subject00.example.yaml configs/pipeline/my_subject.yaml
```

必须检查这些字段：

- `work_dir`：运行结果目录，建议放在有足够空间的磁盘；
- `source.avatar_config`：原始头像 YAML；
- `source.data_dir`、`source.smpl_model_path`、`source.checkpoint`；
- `source.frame_range`：冒烟测试用 `[0, 1, 1]`，正式 subject00 用 `[0, 500, 1]`；
- `source.cuda_device`；
- 两个 DCVC checkpoint 路径；
- `compression.network_q_index`、`rate_num` 和 `rate_index`。

运行全部阶段：

```bash
python -m hgc_avatar.pipeline \
  --config configs/pipeline/my_subject.yaml \
  --stage all
```

这一条命令会自动完成：

1. 编码端原始 `test` 和 PoseMap 导出；
2. SMPL-X、StyleUNet 网络参数和 PoseMap 三路压缩/解压；
3. 使用三路解码中间量执行第二次 `test`；
4. 统计码流、帧数和编解码端图像差异。

因此，正常复现时不需要再手工运行任何 codec 命令。

仅当某个阶段中断、不希望从头执行时，才需要分阶段运行：

```bash
python -m hgc_avatar.pipeline --config configs/pipeline/my_subject.yaml --stage encode
python -m hgc_avatar.pipeline --config configs/pipeline/my_subject.yaml --stage compress
python -m hgc_avatar.pipeline --config configs/pipeline/my_subject.yaml --stage decode
python -m hgc_avatar.pipeline --config configs/pipeline/my_subject.yaml --stage verify
```

网络压缩结果若已经完整存在，重新执行 `compress` 会复用它；DCVC 的临时目录则会安全重建，避免混入旧帧。

## 输出目录

```text
work_dir/
├── encoder/
│   ├── avatar.generated.yaml
│   ├── render/vanilla/rgb_map/
│   └── posemap/
│       ├── frames/
│       └── ranges/
├── compression/
│   ├── smpl/smpl_params.hgc
│   ├── network/{q}_header, {q}_bits
│   └── posemap/
│       ├── bitstreams/
│       ├── dcvc_decoded/
│       └── decoded/frames/
├── decoder/
│   ├── smpl_params.npz
│   ├── checkpoint/net.pt
│   ├── avatar.generated.yaml
│   └── render/vanilla/rgb_map/
├── compression_manifest.json
└── verification.json
```

`compression_manifest.json` 记录三类码流大小、DCVC bpp/PSNR 和压缩参数；`verification.json` 记录两端帧数及编码端—解码端图像 PSNR/MAE。

## 只运行头像训练或测试

原有入口仍然保留：

```bash
# 训练
CUDA_VISIBLE_DEVICES=0 python main_avatar.py \
  -c configs/subject01/avatar.yaml --mode train

# 原始测试
CUDA_VISIBLE_DEVICES=0 python main_avatar.py \
  -c configs/subject00/avatar.yaml --mode test
```

新的 `model.pose_map_io` 配置有三种模式：

- `native`：直接使用实时生成的 PoseMap；
- `export`：渲染同时导出 PoseMap 和 range JSON；
- `decoded`：加载 DCVC 解码图像并反归一化，模拟接收端。

`test.data.smpl_path` 可以显式指向解码后的 `smpl_params.npz`，不再需要覆盖数据集原文件。

## 高级调试：单独运行 codec（可选）

`--stage all` 已经会自动调用下面三个 codec。本节不是正常流程的必做步骤，只在调试某一路码流、做消融实验或单独调整压缩参数时使用。

SMPL-X 无损流：

```bash
python -m hgc_avatar.codecs.smpl_huffman encode smpl_params.npz smpl_params.hgc
python -m hgc_avatar.codecs.smpl_huffman decode smpl_params.hgc restored.npz
```

网络参数：

```bash
python compress_part/codec/quantization/core_coder.py \
  --input_ckpt /path/to/net.pt \
  --output_ckpt /path/to/decoded/net.pt \
  --bitstream_dir /path/to/network_stream \
  --q_index 5
```

PoseMap DCVC：

```bash
python -m hgc_avatar.codecs.posemap_dcvc \
  --frame-dir /path/to/encoder/posemap/frames \
  --work-dir /path/to/compression/posemap \
  --i-frame-model /path/to/cvpr2023_image_psnr.pth.tar \
  --p-frame-model /path/to/cvpr2023_video_psnr.pth.tar \
  --rate-num 2 --rate-index 0 --cuda-device 0
```

## 测试

不依赖大数据的单元测试：

```bash
python -m unittest discover -s tests -v
```

当前覆盖 SMPL-X 字节级无损往返，以及 PoseMap PNG/range 的导出与恢复。

## 已知限制

- `q_index` 和 DCVC rate 点会改变码率与质量；不同点必须分别记录，不要覆盖后直接混合统计。
- PoseMap 的逐帧 min/max 目前作为小型 side information 保存，正式码率统计时应将 JSON 大小计入总码率。
- 500 帧及更长序列才包含 P 帧，单帧冒烟测试不能用于论文表格。
- 当前入口模拟发送端和接收端；若部署到两台机器，只需把 `compression/` 下码流及 PoseMap range side information 传到接收端，并分别执行解码与重建阶段。

## 致谢与引用

头像模型建立在 Animatable Gaussians、3D Gaussian Splatting、StyleAvatar 等项目之上；PoseMap 视频压缩使用 DCVC-DC。使用本代码时请同时引用 HGC-Avatar 论文、Animatable Gaussians 和对应 codec 工作。

```bibtex
@inproceedings{li2024animatablegaussians,
  title={Animatable Gaussians: Learning Pose-dependent Gaussian Maps for High-fidelity Human Avatar Modeling},
  author={Li, Zhe and Zheng, Zerong and Wang, Lizhen and Liu, Yebin},
  booktitle={CVPR},
  year={2024}
}
```

```bibtex
@inproceedings{tang2025hgcavatar,
  title={HGC-Avatar: Hierarchical Gaussian Compression for Streamable Dynamic 3D Avatars},
  author={Tang, Haocheng and Yan, Ruoke and Yin, Xinhui and Zhang, Qi and Zhang, Xinfeng and Ma, Siwei and Gao, Wen and Jia, Chuanmin},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  year={2025},
  doi={10.1145/3746027.3755317}
}
```
