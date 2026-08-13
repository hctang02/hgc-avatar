# 文档目录

- `reports/`：实测实验报告、机器可读汇总和报告插图；
- `standards/`：三维数字人智能编码标准 WD1.0 草案及可复现生成脚本产生的插图；
- 论文、运行指南、N4120 参考文档属于本地参考资料，因版权和仓库体积原因不上传 Git，服务器副本统一放在原工程的 `docs/references/`。

Word 文档由以下命令生成：

```bash
python scripts/generate_digital_human_standard.py \
  --output docs/standards/三维数字人智能编码标准_WD1.0.docx

python scripts/generate_experiment_report.py \
  --summary /path/to/experiment_summary.json \
  --run-dir /path/to/compression-matrix-run \
  --figure-dir docs/reports/figures \
  --output docs/reports/HGC-Avatar多数据集分层压缩实验报告.docx
```

完整渲染帧、模型、原始数据和压缩码流只保存在服务器实验目录，不进入 Git。
