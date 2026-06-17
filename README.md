# Vision Verification

这个仓库用于在另一台服务器上准备 VisionTS 验证实验环境。它不是完整 VisionTS fork，而是一个轻量补丁包：脚本会 clone 官方 VisionTS，然后应用本仓库里的验证补丁。

## 快速开始

```bash
git clone https://github.com/pypypaoying/Vision-Verification.git
cd Vision-Verification
bash prepare_visionts_verification.sh ../VisionTS-verification
```

准备完成后：

```bash
cd ../VisionTS-verification/long_term_tsf
```

按官方目录放数据和 MAE checkpoint：

```text
VisionTS-verification/
  ckpt/
    mae_visualize_vit_base.pth
  long_term_tsf/
    dataset/
      ETT-small/
        ETTm1.csv
        ETTm2.csv
        ETTh1.csv
        ETTh2.csv
      electricity/
        electricity.csv
      weather/
        weather.csv
```

先做一个轻量冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_ettm1_quick.sh
```

跑官方 zero-shot 验证套件：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/vision_verification/run_official_ltsf_suite.sh
```

## 默认验证矩阵

每个数据集和预测长度都会跑：

- `SeasonalNaive`: 直接重复最近一个周期，周期值使用 VisionTS 同一个 `--periodicity`。
- `VisionTS/none`: 官方 VisionTS 路径。
- `VisionTS/stats_only`: 只保留窗口均值/方差反归一化，归一化后的输入置零。
- `VisionTS/time_reverse`: 反转 lookback 时间顺序。
- `VisionTS/time_permute`: 固定随机置换 lookback 时间顺序。
- `VisionTS/row_shuffle`: 打乱渲染图的周期内行结构。
- `VisionTS/col_shuffle`: 打乱渲染图的周期列顺序。
- `VisionTS/image_patch_shuffle`: 打乱 MAE patch 级空间布局。

结果会汇总到：

```text
long_term_tsf/save/vision_verification/summary.csv
```

## 常用裁剪

只跑 ETTm1 和 Weather，且只跑两个预测长度：

```bash
DATASETS="ETTm1 Weather" PRED_LENS="96 192" CUDA_VISIBLE_DEVICES=0 \
  bash scripts/vision_verification/run_official_ltsf_suite.sh
```

只跑部分消融：

```bash
ABLATIONS="none stats_only time_permute col_shuffle" PRED_LENS="96" \
  bash scripts/vision_verification/run_ettm1_quick.sh
```

只跑 SeasonalNaive 或只跑 VisionTS：

```bash
RUN_VISIONTS=0 bash scripts/vision_verification/run_ettm1_quick.sh
RUN_SEASONAL_NAIVE=0 bash scripts/vision_verification/run_ettm1_quick.sh
```

## 证据解读

- 如果 `SeasonalNaive` 接近 `VisionTS/none`，周期数值先验可能贡献很大。
- 如果 `stats_only` 很强，要警惕均值/方差和水平延续带来的 baseline。
- 如果 `time_permute` 仍然强，说明时间顺序本身可能不是核心。
- 如果 `row_shuffle`、`col_shuffle`、`image_patch_shuffle` 显著变差，说明二维渲染几何和 MAE 空间先验确实重要。
