# 服务器大文件目录说明

Git 仓库只保存代码、配置、汇总结果和小型文档。服务器原工程目录还保留以下不进入 Git 的资产：

```text
AnimatableGaussians/
├── data/                       # 原始/预处理数据（保留）
├── results/                    # 训练检查点（保留）
├── smpl_files/                 # SMPL/SMPL-X 授权模型（保留）
├── pose/                       # 姿态与 AMASS 资源（保留）
├── compress_part/
│   ├── data/before_compress/   # 四人物正式压缩源检查点（保留）
│   └── codec/DCVC-DC/checkpoints/ # DCVC 权重（保留）
└── docs/references/            # 论文、运行指南、N4120 参考文档（本地保留）
```

正式批量实验输出位于：

```text
/mnt/hdd2tC/tmp/haocheng/hgc-avatar-runs/compression-matrix-20260813/
```

已删除的内容仅包括可重新生成的旧 test 渲染、debug/cache、临时量化目录、重复旧数据副本和失效链接；具体范围记录在实验报告中。
