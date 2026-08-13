# 文档目录

- `reports/`：实测实验报告、`summaries/` 机器可读汇总和报告插图；
- `standards/`：三维数字人智能编码标准 WD1.0 草案及可复现生成脚本产生的插图；
- 论文、运行指南、N4120 参考文档属于本地参考资料，因版权和仓库体积原因不上传 Git，服务器副本统一放在原工程的 `docs/references/`。

Word 文档由以下命令生成：

```bash
python scripts/generate_digital_human_standard.py \
  --output docs/standards/3D_Digital_Human_Intelligent_Coding_Standard_WD1.0_CN.docx

python scripts/generate_experiment_report.py \
  --summary /path/to/experiment_summary.json \
  --run-dir /path/to/compression-matrix-run \
  --figure-dir docs/reports/figures \
  --output docs/reports/HGC_Avatar_Multi_Dataset_Compression_Report_CN.docx
```

完整渲染帧、模型、原始数据和压缩码流只保存在服务器实验目录，不进入 Git。

> 为避免 Windows/Linux、UTF-8/GBK 之间传输时文件名出现乱码，最终 Word 文档统一使用 ASCII 文件名。Word 内部封面、标题和正文仍为中文。
