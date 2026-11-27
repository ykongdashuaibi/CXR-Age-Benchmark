# CXR-Age-Benchmark

基于 NIH ChestX-ray 数据集的胸片年龄回归（age estimation）基线：CNN 骨干网络 + 回归头，以 L1 损失训练。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `preprocess_nih.py` | 数据预处理：读取标签、按患者拆分 train/val/test、生成统计报告与配置 |
| `train_DNN.py` | 模型训练与验证（MAE / MSE / RMSE / R² / Pearson r） |
| `run_nih_train.slurm` | SLURM 集群提交脚本示例 |

## 1. 准备数据

将下载的图片数据的 `images` 文件夹放到工程目录的 `NIH_data/` 下面。

请确保图片文件**直接**放在 `images/` 下，没有其他额外的层级目录：

```
NIH_data/
├── images/
│   ├── 00000001_000.png
│   └── ...
├── Data_Entry_2017_v2020.csv
├── train_val_list.txt
└── test_list.txt
```

## 2. 数据预处理

包括载入图片标签、按患者拆分数据等：

```bash
python preprocess_nih.py --data_root ./NIH_data
```

- `--data_root`：指定数据目录，即 `images/` 文件夹所在目录。预处理结果（`processed/`、`reports/`）也会保存在该目录。
- 默认使用 `NIH_data/` 目录，建议不要更改。

## 3. 模型训练

目前只支持用 NIH 训练和验证。Supplementary Table 1 里实际使用的也是验证集，而非 JSRT 的测试数据。

```bash
python train_DNN.py --data_root ./NIH_data
```

- `--data_root`：预处理数据所在目录，必须指定，与上一步保持同一个目录即可。

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--backbone` | 模型架构 | `resnet50` |
| `--save_dir` | 模型保存目录 | `checkpoints/` |
| `--batch_size` | 批次大小 | `32` |
| `--num_epochs` | 训练周期 | `100` |

关于 `--num_epochs`：论文中没有说明该参数取值，可自行调整。若运行较慢，可先设为 50 观察效果。

训练结束后，会打印验证集的相关指标，并说明模型保存的路径。

### 在 SLURM 集群上运行

编辑 `run_nih_train.slurm` 中的邮箱与路径后提交：

```bash
sbatch run_nih_train.slurm
```
