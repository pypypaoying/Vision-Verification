# 完整验证流程

这份流程从一台新服务器开始，最终输出完整报告表：

```text
save/vision_verification/summary.csv
save/vision_verification/report_mse.csv
save/vision_verification/report_mae.csv
save/vision_verification/report_mse_delta_vs_visionts_none.csv
save/vision_verification/report_mse_ratio_vs_visionts_none.csv
```

其中 `report_mse.csv` 和 `report_mae.csv` 是宽表，行是 `data,pred_len`，列是 `VisionTS/none`、各消融模式和 `SeasonalNaive`。

## 1. 准备代码

```bash
git clone https://github.com/pypypaoying/Vision-Verification.git
cd Vision-Verification
bash prepare_visionts_verification.sh ../VisionTS-verification
cd ../VisionTS-verification
```

这一步会 clone 官方 VisionTS，并创建 `vision-verification` 分支，自动应用验证补丁。

## 2. 配置 Python 环境

官方 README 说明论文结果基于 `python==3.8.18`、`torch==1.7.1`、`torchvision==0.8.2`、`timm==0.3.2`。为了尽量复现，推荐用 Python 3.8。

如果服务器有 conda 或 mamba：

```bash
conda create -n visionts-verification python=3.8.18 -y
conda activate visionts-verification
```

安装 PyTorch。下面默认使用 CUDA 11.0 wheel，适合多数较老/兼容驱动的服务器：

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

如果 CUDA/显卡太新导致 PyTorch 1.7.1 不可用，可以改装当前服务器匹配的新版 PyTorch，但结果可能与论文环境有轻微差异。

安装其余依赖：

```bash
cd ../VisionTS-verification
# 跳过 long_term_tsf/requirements.txt 里的 torch，避免覆盖刚才装好的 CUDA wheel。
grep -v '^torch==' long_term_tsf/requirements.txt > /tmp/visionts_ltsf_req_no_torch.txt
pip install -r /tmp/visionts_ltsf_req_no_torch.txt
pip install timm==0.3.2 pillow requests huggingface_hub gdown
```

如果你需要用 `dataset.7z` 镜像方式下载数据，还需要 7z：

```bash
conda install -c conda-forge p7zip -y
```

## 3. 准备 MAE checkpoint

默认脚本从 `long_term_tsf` 运行，checkpoint 目录是 `../ckpt/`，也就是仓库根目录下的 `ckpt/`。

```bash
cd ../VisionTS-verification
mkdir -p ckpt
wget -c -O ckpt/mae_visualize_vit_base.pth \
  https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth
```

如果不手动下载，VisionTS 首次运行时也会尝试自动下载；提前下载更利于排查问题。

## 4. 下载并放置数据集

完整验证套件需要 6 个 long-term forecasting 数据集：

```text
long_term_tsf/dataset/ETT-small/ETTm1.csv
long_term_tsf/dataset/ETT-small/ETTm2.csv
long_term_tsf/dataset/ETT-small/ETTh1.csv
long_term_tsf/dataset/ETT-small/ETTh2.csv
long_term_tsf/dataset/electricity/electricity.csv
long_term_tsf/dataset/weather/weather.csv
```

### 方式 A：官方 Time-Series-Library Google Drive

官方 long-term forecasting 数据在 Time-Series-Library README 指向的 Google Drive 文件夹中。

```bash
cd ../VisionTS-verification/long_term_tsf
mkdir -p dataset
gdown --folder "https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing" -O dataset
```

下载后检查路径：

```bash
find dataset -maxdepth 3 -type f | grep -E 'ETTm1.csv|ETTm2.csv|ETTh1.csv|ETTh2.csv|electricity.csv|weather.csv'
```

如果 `gdown` 多套了一层目录，把对应子目录移动到 `long_term_tsf/dataset/` 下即可。

### 方式 B：Hugging Face/GitHub 镜像包

这个镜像包通常下载更直接，但不是 VisionTS 官方仓库本身维护的链接。

```bash
cd ../VisionTS-verification
wget -c -O /tmp/tsf_dataset.7z \
  https://github.com/duyu09/TimeSeries-Forecasting-Dataset/releases/download/v1.0.0/dataset.7z
mkdir -p /tmp/tsf_dataset
7z x /tmp/tsf_dataset.7z -o/tmp/tsf_dataset
mkdir -p long_term_tsf/dataset
rsync -av /tmp/tsf_dataset/dataset/ long_term_tsf/dataset/
```

再次检查：

```bash
cd long_term_tsf
find dataset -maxdepth 3 -type f | grep -E 'ETTm1.csv|ETTm2.csv|ETTh1.csv|ETTh2.csv|electricity.csv|weather.csv'
```

## 5. 先跑快速冒烟测试

```bash
cd ../VisionTS-verification/long_term_tsf
CUDA_VISIBLE_DEVICES=0 PRED_LENS="96" bash scripts/vision_verification/run_ettm1_quick.sh
```

成功后会生成：

```text
save/vision_verification_quick/summary.csv
save/vision_verification_quick/report_mse.csv
save/vision_verification_quick/report_mae.csv
save/vision_verification_quick/report_mse_delta_vs_visionts_none.csv
save/vision_verification_quick/report_mse_ratio_vs_visionts_none.csv
```

## 6. 跑完整验证套件

```bash
cd ../VisionTS-verification/long_term_tsf
CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_official_ltsf_suite.sh
```

默认会跑：

```text
datasets: ETTm1 ETTm2 ETTh1 ETTh2 ECL Weather
pred_len: 96 192 336 720
models: SeasonalNaive, VisionTS/none, VisionTS 所有消融
```

如果显存不够，可以先降低 batch size：

```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 bash scripts/vision_verification/run_official_ltsf_suite.sh
```

如果想分批跑：

```bash
DATASETS="ETTm1 ETTm2" CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_official_ltsf_suite.sh
DATASETS="ETTh1 ETTh2" CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_official_ltsf_suite.sh
DATASETS="ECL Weather" CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_official_ltsf_suite.sh
```

脚本默认 `SKIP_DONE=1`，已经有 `metrics.npy` 的实验会跳过，适合断点续跑。

## 7. 重新生成完整报告表

如果中途分批跑，最后可以手动重新汇总：

```bash
cd ../VisionTS-verification/long_term_tsf
python scripts/vision_verification/summarize_verification.py \
  --root save/vision_verification \
  --out save/vision_verification/summary.csv

python scripts/vision_verification/make_report_tables.py \
  --summary save/vision_verification/summary.csv \
  --out-dir save/vision_verification
```

最终表：

```text
save/vision_verification/summary.csv
save/vision_verification/report_mse.csv
save/vision_verification/report_mae.csv
save/vision_verification/report_mse_delta_vs_visionts_none.csv
save/vision_verification/report_mse_ratio_vs_visionts_none.csv
```

## 8. 读表方式

`report_mse.csv`：直接比较绝对 MSE。重点看 `none`、`SeasonalNaive/period_*` 和各消融列。

`report_mse_delta_vs_visionts_none.csv`：每列都是 `variant_mse - VisionTS_none_mse`。大于 0 表示该消融更差，小于 0 表示该消融反而更好。

`report_mse_ratio_vs_visionts_none.csv`：每列都是 `variant_mse / VisionTS_none_mse`。例如 `1.20` 表示比官方 VisionTS MSE 高 20%。

判断逻辑：

- `SeasonalNaive` 接近 `none`：周期数值先验可能已经解释大量性能。
- `stats_only` 很强：均值/方差和水平延续是强 baseline。
- `time_permute` 接近 `none`：模型可能更多利用数值分布，而不是时间顺序。
- `row_shuffle`、`col_shuffle`、`image_patch_shuffle` 明显变差：二维渲染几何和 MAE 空间先验更可能是真贡献。
